"""Build a bounded context table from the user's selected trajectories."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .actions import validate_action
from .features import FEATURE_NAMES, SCHEMA, extract_features
from .ram import parse_frame, read_frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(root: Path, outcome: str):
    episodes = {}
    for path in sorted(root.rglob("*.png")):
        frame = parse_frame(path)
        if outcome != "all" and frame.outcome != outcome:
            continue
        episodes.setdefault((frame.episode, frame.outcome), []).append(frame)
    if not episodes:
        raise ValueError(f"No {outcome} trajectory PNGs under {root}")
    for key, frames in episodes.items():
        frames.sort(key=lambda frame: frame.number)
        if len({frame.number for frame in frames}) != len(frames):
            raise ValueError(f"Duplicate frame numbers in {key}")
    return episodes


def prepare(root: Path, output: Path, *, outcome="win", stride=4,
            max_rows=8192, seed=0, label_offset=1, encoding="dataset-cr"):
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new path: {output}")
    if stride < 1 or max_rows < 1 or label_offset not in (0, 1):
        raise ValueError("stride/max_rows must be positive; label_offset must be 0 or 1")
    episodes = discover(root, outcome)
    # Reservoir sample candidate pairs BEFORE image I/O. No implicit train/test
    # split, no action balancing that would change the demonstrator's prior.
    rng = np.random.default_rng(seed)
    reservoir = []
    eligible = 0
    counts = Counter()
    for frames in episodes.values():
        by_number = {frame.number: frame for frame in frames}
        for source in frames[::stride]:
            target = by_number.get(source.number + label_offset)
            if target is None:
                counts["missing_target_frame"] += 1
                continue
            try:
                validate_action(target.action)
            except ValueError:
                counts["non_gameplay_action"] += 1
                continue
            eligible += 1
            item = (source, target)
            if len(reservoir) < max_rows:
                reservoir.append(item)
            else:
                index = int(rng.integers(eligible))
                if index < max_rows:
                    reservoir[index] = item
    reservoir.sort(key=lambda pair: (pair[0].episode, pair[0].number))
    rows, labels, source_paths, target_paths, episode_ids, levels, frames_out = [], [], [], [], [], [], []
    provenance = hashlib.sha256()
    for source, target in reservoir:
        ram = read_frame(source, encoding)
        if source.path != target.path:
            read_frame(target, encoding)  # validate the label's own BP1/outcome
        # Normal controllable game state; remove title/transition/ending frames.
        if int(ram[0x770]) != 1 or int(ram[0x0E]) != 8:
            counts["not_controllable"] += 1
            continue
        rows.append(extract_features(ram))
        labels.append(target.action)
        source_paths.append(source.path.relative_to(root).as_posix())
        target_paths.append(target.path.relative_to(root).as_posix())
        episode_ids.append(source.episode)
        levels.append(f"{source.world}-{source.level}")
        frames_out.append(source.number)
        for frame in (source, target):
            provenance.update(frame.path.relative_to(root).as_posix().encode())
            provenance.update(bytes.fromhex(sha256(frame.path)))
    if not rows:
        raise ValueError("No controllable samples survived. Check RAM encoding and selection.")
    metadata = {
        "schema": SCHEMA, "feature_names": list(FEATURE_NAMES),
        "source_root": str(root.resolve()), "outcome": outcome,
        "stride": stride, "label_offset": label_offset, "ram_encoding": encoding,
        "max_rows": max_rows, "seed": seed, "eligible_pairs": eligible,
        "selected_rows": len(rows), "trajectory_count": len(set(episode_ids)),
        "levels": sorted(set(levels)), "skipped": dict(counts),
        "action_counts": dict(sorted(Counter(labels).items())),
        "source_pairs_sha256": provenance.hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # A file handle avoids numpy silently appending another suffix.
    with output.open("xb") as stream:
        np.savez_compressed(stream, X=np.stack(rows), y=np.asarray(labels, dtype=np.int64),
                            source_paths=np.asarray(source_paths), target_paths=np.asarray(target_paths),
                            episodes=np.asarray(episode_ids), levels=np.asarray(levels),
                            frames=np.asarray(frames_out), metadata=np.asarray(json.dumps(metadata)))
    return metadata


def load_table(path: Path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        X, y = data["X"].copy(), data["y"].copy()
    if metadata.get("schema") != SCHEMA or metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Feature schema differs; prepare the table again with this version")
    if X.ndim != 2 or X.shape != (len(y), len(FEATURE_NAMES)) or len(y) == 0:
        raise ValueError("Invalid feature/label dimensions")
    if np.isinf(X).any():
        raise ValueError("Infinite feature values are invalid")
    for action in np.unique(y):
        validate_action(action)
    return X, y, metadata
