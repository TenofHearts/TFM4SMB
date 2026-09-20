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


def _player_x(ram):
    return int(ram[0x6D]) * 256 + int(ram[0x86])


def _decimal_counter(ram, start, length):
    value = 0
    for digit in ram[start : start + length]:
        value = value * 10 + int(digit)
    return value


def _is_death_state(ram):
    return int(ram[0x0E]) in {0x06, 0x0B} or int(ram[0xB5]) > 1


def _death_frame(frames, read_ram):
    """Find the start of the final contiguous death-state block, if present."""
    onset = None
    for frame in reversed(frames):
        if _is_death_state(read_ram(frame)):
            onset = frame.number
        elif onset is not None:
            break
    return onset


def _action_value(
    before,
    after,
    progress_max,
    effect_frame,
    death_frame,
    window,
    min_progress_delta=1,
):
    if (
        death_frame is not None
        and effect_frame <= death_frame
        and death_frame - effect_frame < window
    ):
        return -1
    before_x = _player_x(before)
    after_x = _player_x(after)
    useful_progress = (
        after_x - before_x >= min_progress_delta and after_x > progress_max
    )
    score_gain = _decimal_counter(after, 0x7DE, 6) > _decimal_counter(
        before, 0x7DE, 6
    )
    powerup_gain = int(after[0x756]) > int(before[0x756])
    return int(useful_progress or score_gain or powerup_gain)


def _select_trajectory_rows(candidates, max_rows, rng):
    """Keep action changes, then sample round-robin across trajectory IDs."""
    availability = Counter(
        item[1].action
        for episode_candidates in candidates.values()
        for item in episode_candidates
    )
    mandatory = [
        item
        for episode_candidates in candidates.values()
        for item in episode_candidates
        if item[3]
    ]
    if len(mandatory) > max_rows:
        raise ValueError(
            f"{len(mandatory)} mandatory action-change rows exceed max_rows={max_rows}"
        )
    selected = list(mandatory)
    queues = {}
    for episode, episode_candidates in candidates.items():
        ordinary = [item for item in episode_candidates if not item[3]]
        head = [item for item in ordinary if item[2]]
        tail = [item for item in ordinary if not item[2]]
        rng.shuffle(tail)
        queues[episode] = head + tail
    episode_order = sorted(queues)
    rng.shuffle(episode_order)
    remaining = min(max_rows, sum(availability.values())) - len(selected)
    while remaining:
        progressed = False
        for episode in episode_order:
            if queues[episode]:
                selected.append(queues[episode].pop(0))
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("Trajectory selection exhausted its candidates")
    rng.shuffle(selected)
    selected_counts = Counter(item[1].action for item in selected)
    return selected, availability, selected_counts


def discover(
    root: Path, outcome: str, cache=None, include_level=None, exclude_level=None
):
    episodes = {}
    paths = cache.frames(outcome) if cache else sorted(root.rglob("*.png"))
    for item in paths:
        frame = item if cache else parse_frame(item)
        if not cache and outcome != "all" and frame.outcome != outcome:
            continue
        level = f"{frame.world}-{frame.level}"
        if include_level is not None and include_level != level:
            continue
        if exclude_level == level:
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
    stride=2,
    max_rows=8192,
    seed=0,
    label_offset=1,
    encoding="dataset-cr",
    include_level=None,
    exclude_level=None,
    head_rows_per_trajectory=16,
    pre_death_frames=30,
    min_progress_delta=3,
):
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new path: {output}")
    if (
        stride < 1
        or max_rows < 1
        or head_rows_per_trajectory < 0
        or label_offset not in (0, 1)
        or pre_death_frames < 1
        or min_progress_delta < 1
    ):
        raise ValueError(
            "stride/max_rows must be positive; head rows must be nonnegative; "
            "label_offset must be 0 or 1; pre-death frames and minimum progress "
            "delta must be positive"
        )
    if include_level is not None and exclude_level is not None:
        raise ValueError("include_level and exclude_level are mutually exclusive")
    for name, level in (("include_level", include_level), ("exclude_level", exclude_level)):
        if level is not None and re.fullmatch(r"[1-8]-[1-4]", level) is None:
            raise ValueError(f"{name} must look like 1-1")
    cache = MetadataCache(root) if (root / MANIFEST).is_file() else None
    if cache and cache.info.get("ram_encoding") != encoding:
        raise ValueError("Requested RAM encoding differs from the metadata cache")
    episodes = discover(root, outcome, cache, include_level, exclude_level)
    # Build per-trajectory candidate queues. Selection below caps action share and
    # round-robins trajectory IDs, so neither long episodes nor modal actions can
    # consume the context merely because they have more frames.
    rng = np.random.default_rng(seed)
    effective_head = head_rows_per_trajectory
    candidates = {}
    frame_lookup = {}
    death_frames = {}
    eligible = 0
    counts = Counter()
    for episode_key, frames in episodes.items():
        by_number = {frame.number: frame for frame in frames}
        frame_lookup[episode_key] = by_number
        read_ram = (
            cache.ram if cache else lambda frame: read_frame(frame, encoding)
        )
        death_frames[episode_key] = (
            _death_frame(frames, read_ram) if episode_key[1] == "fail" else None
        )
        episode_head = 0
        episode_candidates = []
        progress_max = -1
        for frame_index, source in enumerate(frames):
            target = by_number.get(source.number + label_offset)
            if target is None:
                counts["missing_target_frame"] += 1
                continue
            previous_target = by_number.get(target.number - 1)
            is_action_change = (
                previous_target is not None
                and target.action != previous_target.action
            )
            if frame_index % stride != 0 and not is_action_change:
                counts["stride_dropped"] += 1
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
            progress_before = progress_max
            progress_max = max(progress_max, _player_x(ram))
            # Keep only lightweight frame records until selection. Retaining RAM
            # for every candidate would scale to multiple gigabytes on the full
            # cache even though only max_rows samples can reach the output.
            episode_candidates.append(
                (
                    source,
                    target,
                    is_head,
                    is_action_change,
                    progress_before,
                    progress_max,
                )
            )
            episode_head += 1
        if episode_candidates:
            candidates[episode_key] = episode_candidates
    selected, candidate_action_counts, selected_action_counts = _select_trajectory_rows(
        candidates, max_rows, rng
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
    action_values = []
    for (
        source,
        target,
        _,
        _,
        progress_before,
        progress_through_source,
    ) in selected:
        ram = cache.ram(source) if cache else read_frame(source, encoding)
        target_ram = cache.ram(target) if cache else read_frame(target, encoding)
        previous = frame_lookup[(source.episode, source.outcome)].get(source.number - 1)
        if previous is None:
            previous_ram = ram
            counts["padded_history"] += 1
        else:
            previous_ram = cache.ram(previous) if cache else read_frame(previous, encoding)
        if label_offset == 0:
            before, after = previous_ram, ram
            progress_reference = progress_before
            effect_frame = source.number
        else:
            before, after = ram, target_ram
            progress_reference = progress_through_source
            effect_frame = target.number
        action_value = _action_value(
            before,
            after,
            progress_reference,
            effect_frame,
            death_frames[(source.episode, source.outcome)],
            pre_death_frames,
            min_progress_delta,
        )
        rows.append(extract_features(ram, previous_ram, action_value=action_value))
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
        action_values.append(action_value)
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
        "included_level": include_level,
        "excluded_level": exclude_level,
        "ram_encoding": encoding,
        "max_rows": max_rows,
        "head_rows_per_trajectory": head_rows_per_trajectory,
        "effective_head_rows_per_trajectory": effective_head,
        "priority_rows": sum(item[2] for item in selected),
        "mandatory_action_change_rows": sum(item[3] for item in selected),
        "selection": "mandatory-changes-trajectory-round-robin",
        "pre_death_frames": pre_death_frames,
        "min_progress_delta": min_progress_delta,
        "action_value_rule": {
            "positive": (
                "minimum_forward_delta_and_new_progress_max_or_score_gain_or_"
                "powerup_gain"
            ),
            "neutral": "no_observed_transition_reward",
            "negative": "within_pre_death_window",
        },
        "candidate_action_counts": dict(sorted(candidate_action_counts.items())),
        "seed": seed,
        "eligible_pairs": eligible,
        "selected_rows": len(rows),
        "selection_discarded_rows": eligible - len(rows),
        "trajectory_count": len(set(episode_ids)),
        "outcome_counts": dict(sorted(Counter(outcomes).items())),
        "action_value_counts": dict(sorted(Counter(action_values).items())),
        "detected_death_trajectories": sum(
            frame is not None for frame in death_frames.values()
        ),
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
            action_values=np.asarray(action_values, dtype=np.int8),
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
