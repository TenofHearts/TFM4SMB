"""Build a bounded context table from the user's selected trajectories."""

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from .actions import validate_action
from .features import FEATURE_NAMES, SCHEMA, extract_features
from .metadata_cache import MANIFEST, MetadataCache
from .ram import parse_frame, read_frame


def discover(root: Path, outcome: str, cache=None, exclude_level=None):
    episodes = {}
    paths = cache.frames(outcome) if cache else sorted(root.rglob("*.png"))
    for item in paths:
        frame = item if cache else parse_frame(item)
        if not cache and outcome != "all" and frame.outcome != outcome:
            continue
        if exclude_level == f"{frame.world}-{frame.level}":
            continue
        episodes.setdefault((frame.episode, frame.outcome), []).append(frame)
    if not episodes:
        raise ValueError(f"No {outcome} trajectory PNGs under {root}")
    for key, frames in episodes.items():
        frames.sort(key=lambda frame: frame.number)
        if len({frame.number for frame in frames}) != len(frames):
            raise ValueError(f"Duplicate frame numbers in {key}")
    return episodes


def prepare(
    root: Path,
    output: Path,
    *,
    outcome="win",
    stride=4,
    max_rows=8192,
    seed=0,
    label_offset=1,
    encoding="dataset-cr",
    exclude_level=None,
    head_rows_per_trajectory=16,
):
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new path: {output}")
    if (
        stride < 1
        or max_rows < 1
        or head_rows_per_trajectory < 0
        or label_offset not in (0, 1)
    ):
        raise ValueError(
            "stride/max_rows must be positive; head rows must be nonnegative; "
            "label_offset must be 0 or 1"
        )
    if (
        exclude_level is not None
        and re.fullmatch(r"[1-8]-[1-4]", exclude_level) is None
    ):
        raise ValueError("exclude_level must look like 8-4")
    cache = MetadataCache(root) if (root / MANIFEST).is_file() else None
    if cache and cache.info.get("ram_encoding") != encoding:
        raise ValueError("Requested RAM encoding differs from the metadata cache")
    episodes = discover(root, outcome, cache, exclude_level)
    # Preserve an opening segment from every trajectory, then uniformly sample
    # the remainder. Pure global sampling loses the rare wait-to-move transition.
    rng = np.random.default_rng(seed)
    effective_head = min(head_rows_per_trajectory, max_rows // len(episodes))
    remainder_capacity = max_rows - effective_head * len(episodes)
    priority = []
    reservoir = []
    eligible = 0
    remainder_seen = 0
    counts = Counter()
    for frames in episodes.values():
        by_number = {frame.number: frame for frame in frames}
        episode_head = 0
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
            ram = cache.ram(source) if cache else read_frame(source, encoding)
            if int(ram[0x770]) != 1 or int(ram[0x0E]) != 8:
                counts["not_controllable"] += 1
                continue
            eligible += 1
            item = (source, target, ram)
            if episode_head < effective_head:
                priority.append(item)
                episode_head += 1
                continue
            remainder_seen += 1
            if len(reservoir) < remainder_capacity:
                reservoir.append(item)
            else:
                index = int(rng.integers(remainder_seen))
                if index < remainder_capacity:
                    reservoir[index] = item
    selected = priority + reservoir
    selected.sort(key=lambda item: (item[0].episode, item[0].number))
    rows, labels, source_paths, target_paths, episode_ids, levels, frames_out = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for source, target, ram in selected:
        if not cache and source.path != target.path:
            read_frame(target, encoding)  # validate the label's own BP1/outcome
        rows.append(extract_features(ram))
        labels.append(target.action)
        source_path = (
            source.path.as_posix()
            if cache
            else source.path.relative_to(root).as_posix()
        )
        target_path = (
            target.path.as_posix()
            if cache
            else target.path.relative_to(root).as_posix()
        )
        source_paths.append(source_path)
        target_paths.append(target_path)
        episode_ids.append(source.episode)
        levels.append(f"{source.world}-{source.level}")
        frames_out.append(source.number)
    if not rows:
        raise ValueError(
            "No controllable samples survived. Check RAM encoding and selection."
        )
    metadata = {
        "schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "source_root": str(root.resolve()),
        "source_format": "npz-cache" if cache else "png",
        "outcome": outcome,
        "stride": stride,
        "label_offset": label_offset,
        "excluded_level": exclude_level,
        "ram_encoding": encoding,
        "max_rows": max_rows,
        "head_rows_per_trajectory": head_rows_per_trajectory,
        "effective_head_rows_per_trajectory": effective_head,
        "priority_rows": len(priority),
        "seed": seed,
        "eligible_pairs": eligible,
        "selected_rows": len(rows),
        "trajectory_count": len(set(episode_ids)),
        "levels": sorted(set(levels)),
        "skipped": dict(counts),
        "action_counts": dict(sorted(Counter(labels).items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # A file handle avoids numpy silently appending another suffix.
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            X=np.stack(rows),
            y=np.asarray(labels, dtype=np.int64),
            source_paths=np.asarray(source_paths),
            target_paths=np.asarray(target_paths),
            episodes=np.asarray(episode_ids),
            levels=np.asarray(levels),
            frames=np.asarray(frames_out),
            metadata=np.asarray(json.dumps(metadata)),
        )
    return metadata


def load_table(path: Path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        X, y = data["X"].copy(), data["y"].copy()
    if metadata.get("schema") != SCHEMA or metadata.get("feature_names") != list(
        FEATURE_NAMES
    ):
        raise ValueError(
            "Feature schema differs; prepare the table again with this version"
        )
    if X.ndim != 2 or X.shape != (len(y), len(FEATURE_NAMES)) or len(y) == 0:
        raise ValueError("Invalid feature/label dimensions")
    if np.isinf(X).any():
        raise ValueError("Infinite feature values are invalid")
    for action in np.unique(y):
        validate_action(action)
    return X, y, metadata
