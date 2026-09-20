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


def _balanced_quotas(availability, total, max_action_share):
    """Allocate a proportional row budget with a hard per-action ceiling."""
    ceiling = int(np.floor(total * max_action_share))
    if ceiling < 1:
        raise ValueError("max_action_share is too small for the requested context")
    if sum(min(count, ceiling) for count in availability.values()) < total:
        dominant, count = max(availability.items(), key=lambda item: item[1])
        raise ValueError(
            f"Cannot select {total} rows with max_action_share={max_action_share}: "
            f"action {dominant} has {count} candidates and the other actions do "
            "not provide enough rows. Increase the share or reduce max_rows."
        )
    quotas = {action: 0 for action in availability}
    remaining = total
    active = set(availability)
    while remaining:
        weights = {action: availability[action] - quotas[action] for action in active}
        weight_total = sum(weights.values())
        if not weight_total:
            raise ValueError("Action-balanced selection exhausted its candidates")
        progress = False
        for action in sorted(active):
            room = min(availability[action], ceiling) - quotas[action]
            if room <= 0:
                continue
            grant = min(room, max(1, int(round(remaining * weights[action] / weight_total))))
            grant = min(grant, remaining)
            quotas[action] += grant
            remaining -= grant
            progress = True
            if not remaining:
                break
        active = {
            action
            for action in active
            if quotas[action] < min(availability[action], ceiling)
        }
        if not progress:
            raise ValueError("Action-balanced selection could not satisfy its cap")
    return quotas


def _select_trajectory_rows(candidates, max_rows, max_action_share, rng):
    """Action-stratified sampling, round-robin across trajectory IDs."""
    total = min(max_rows, sum(map(len, candidates.values())))
    availability = Counter(
        item[1].action
        for episode_candidates in candidates.values()
        for item in episode_candidates
    )
    quotas = _balanced_quotas(availability, total, max_action_share)
    selected = []
    for action, quota in sorted(quotas.items()):
        queues = {}
        for episode, episode_candidates in candidates.items():
            matches = [item for item in episode_candidates if item[1].action == action]
            if matches:
                # Keep opening transitions first, but randomize later rows.
                head = [item for item in matches if item[2]]
                tail = [item for item in matches if not item[2]]
                rng.shuffle(tail)
                queues[episode] = head + tail
        episode_order = sorted(queues)
        rng.shuffle(episode_order)
        while quota:
            progressed = False
            for episode in episode_order:
                if queues[episode]:
                    selected.append(queues[episode].pop(0))
                    quota -= 1
                    progressed = True
                    if not quota:
                        break
            if not progressed:
                raise ValueError(f"Insufficient candidates for action {action}")
    rng.shuffle(selected)
    return selected, availability, quotas


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
    outcome="all",
    stride=4,
    max_rows=8192,
    seed=0,
    label_offset=1,
    encoding="dataset-cr",
    exclude_level=None,
    head_rows_per_trajectory=16,
    max_action_share=0.50,
):
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new path: {output}")
    if (
        stride < 1
        or max_rows < 1
        or head_rows_per_trajectory < 0
        or label_offset not in (0, 1)
        or not 0 < max_action_share <= 1
    ):
        raise ValueError(
            "stride/max_rows must be positive; head rows must be nonnegative; "
            "label_offset must be 0 or 1; max_action_share must be in (0, 1]"
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
    # Build per-trajectory candidate queues. Selection below caps action share and
    # round-robins trajectory IDs, so neither long episodes nor modal actions can
    # consume the context merely because they have more frames.
    rng = np.random.default_rng(seed)
    effective_head = head_rows_per_trajectory
    candidates = {}
    frame_lookup = {}
    eligible = 0
    counts = Counter()
    for episode_key, frames in episodes.items():
        by_number = {frame.number: frame for frame in frames}
        frame_lookup[episode_key] = by_number
        episode_head = 0
        episode_candidates = []
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
            is_head = episode_head < effective_head
            # Keep only lightweight frame records until selection. Retaining RAM
            # for every candidate would scale to multiple gigabytes on the full
            # cache even though only max_rows samples can reach the output.
            episode_candidates.append((source, target, is_head))
            episode_head += 1
        if episode_candidates:
            candidates[episode_key] = episode_candidates
    selected, candidate_action_counts, selected_action_counts = _select_trajectory_rows(
        candidates, max_rows, max_action_share, rng
    )
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
    outcomes = []
    for source, target, _ in selected:
        if not cache and source.path != target.path:
            read_frame(target, encoding)  # validate the label's own BP1/outcome
        ram = cache.ram(source) if cache else read_frame(source, encoding)
        previous = frame_lookup[(source.episode, source.outcome)].get(source.number - 1)
        if previous is None:
            previous_ram = ram
            counts["padded_history"] += 1
        else:
            previous_ram = cache.ram(previous) if cache else read_frame(previous, encoding)
        success = int(source.outcome == "win")
        rows.append(extract_features(ram, previous_ram, success=success))
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
        outcomes.append(source.outcome)
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
        "priority_rows": sum(item[2] for item in selected),
        "selection": "action-balanced-trajectory-round-robin",
        "max_action_share": max_action_share,
        "candidate_action_counts": dict(sorted(candidate_action_counts.items())),
        "seed": seed,
        "eligible_pairs": eligible,
        "selected_rows": len(rows),
        "trajectory_count": len(set(episode_ids)),
        "outcome_counts": dict(sorted(Counter(outcomes).items())),
        "levels": sorted(set(levels)),
        "skipped": dict(counts),
        "action_counts": dict(sorted(selected_action_counts.items())),
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
            outcomes=np.asarray(outcomes),
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
