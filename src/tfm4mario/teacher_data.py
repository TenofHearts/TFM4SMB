"""Prepare imitation rows at the collected DQN teacher's decision boundaries."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .actions import validate_action
from .dataset import (
    _action_value,
    _player_x,
    _positive_reward_event,
    _rolling_action_value,
)
from .features import FEATURE_NAMES, SCHEMA, checked_ram, extract_features
from .metadata_cache import CACHE_SCHEMA, MetadataCache
from .teacher import ACTION_REPEAT, TEACHER_ENV_ID


def prepare_teacher_decisions(
    data: Path,
    output: Path,
    *,
    episodes=None,
    min_progress_delta=3,
    value_mode="student-judge",
    judge_horizon_frames=16,
    judge_min_progress_delta=16,
    death_lookback_actions=4,
):
    """Map reset and post-decision RAM to the teacher's next complete action."""
    data, output = Path(data), Path(output)
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new path: {output}")
    if min_progress_delta < 1:
        raise ValueError("minimum progress delta must be positive")
    if value_mode not in {"student-judge", "transition", "trajectory-positive"}:
        raise ValueError(
            "value mode must be student-judge, transition, or trajectory-positive"
        )
    if (
        judge_horizon_frames < 1
        or judge_min_progress_delta < 1
        or death_lookback_actions < 1
    ):
        raise ValueError(
            "student judge horizon, progress delta, and death lookback must be positive"
        )
    manifest = MetadataCache(data).info
    if (
        manifest.get("source") != "roclark/super-mario-bros-dqn"
        or manifest.get("environment") != TEACHER_ENV_ID
        or manifest.get("teacher_action_repeat") != ACTION_REPEAT
        or manifest.get("ram_encoding") != "raw"
    ):
        raise ValueError("Expected a complete 1-3 DQN teacher cache with raw RAM")

    selected_episodes = None if episodes is None else set(episodes)
    found = set()
    columns = {key: [] for key in (
        "X", "y", "source_paths", "target_paths", "episodes", "levels",
        "frames", "outcomes", "action_values", "previous_actions",
    )}
    for shard in sorted(data.glob("*.win.npz")):
        with np.load(shard, allow_pickle=False) as saved:
            info = json.loads(str(saved["metadata"]))
            episode = info["episode"]
            if selected_episodes is not None and episode not in selected_episodes:
                continue
            if (
                info.get("schema") != CACHE_SCHEMA
                or info.get("outcome") != "win"
                or info.get("teacher_action_repeat") != ACTION_REPEAT
                or "reset_ram" not in saved
            ):
                raise ValueError(f"Teacher shard lacks aligned reset/decision data: {shard}")
            reset_ram = checked_ram(saved["reset_ram"])
            ram = saved["ram"]
            actions = saved["actions"]
            frames = saved["frames"]
            paths = saved["paths"]
            count = len(actions)
            if (
                ram.shape != (count, 2048)
                or frames.shape != (count,)
                or paths.shape != (count,)
                or not np.array_equal(frames, np.arange(1, count + 1))
                or count == 0
            ):
                raise ValueError(f"Invalid teacher frame sequence: {shard}")
            for action in np.unique(actions):
                validate_action(int(action))
            for start in range(0, count, ACTION_REPEAT):
                if np.any(actions[start : start + ACTION_REPEAT] != actions[start]):
                    raise ValueError(f"Teacher action changed inside a decision: {shard}")

            found.add(episode)
            if "rewards" in saved:
                rewards = saved["rewards"]
                if rewards.shape != (count,) or not np.all(np.isfinite(rewards)):
                    raise ValueError(f"Invalid teacher rewards: {shard}")
                raw_reward_events = rewards > 0
            else:
                # Older teacher shards predate per-frame reward persistence.
                raw_reward_events = []
                reward_progress_max = _player_x(reset_ram)
                reward_before = reset_ram
                for index, state in enumerate(ram):
                    raw_reward_events.append(
                        _positive_reward_event(
                            reward_before,
                            state,
                            reward_progress_max,
                            completion=index == count - 1,
                        )
                    )
                    reward_progress_max = max(reward_progress_max, _player_x(state))
                    reward_before = state
            progress_max = _player_x(reset_ram)
            entries = []
            for source_frame in range(0, count, ACTION_REPEAT):
                ending_frame = min(source_frame + ACTION_REPEAT, count) - 1
                decision_progress_max = max(
                    progress_max,
                    *(_player_x(state) for state in ram[source_frame : ending_frame + 1]),
                )
                if source_frame == 0:
                    current = reset_ram
                    previous = None
                    previous_action = 0
                    source_path = f"teacher/{episode}/reset"
                else:
                    current = ram[source_frame - 1]
                    previous = ram[source_frame - 2]
                    previous_action = int(actions[source_frame - 1])
                    source_path = str(paths[source_frame - 1])
                    if int(current[0x770]) != 1 or int(current[0x0E]) != 8:
                        progress_max = decision_progress_max
                        continue
                after = ram[ending_frame]
                entries.append(
                    {
                        "current": current,
                        "previous": previous,
                        "previous_action": previous_action,
                        "source_path": source_path,
                        "target_path": str(paths[source_frame]),
                        "source_frame": source_frame,
                        "action": int(actions[source_frame]),
                        "after": after,
                        "progress_before": progress_max,
                        "effect_frame": ending_frame + 1,
                    }
                )
                progress_max = decision_progress_max

            if value_mode == "trajectory-positive":
                values = [1] * len(entries)
            elif value_mode == "transition":
                values = [
                    _action_value(
                        entry["current"],
                        entry["after"],
                        entry["progress_before"],
                        effect_frame=entry["effect_frame"],
                        death_frame=None,
                        window=1,
                        min_progress_delta=min_progress_delta,
                    )
                    for entry in entries
                ]
            else:
                values = []
                for entry in entries:
                    start = entry["source_frame"]
                    end = min(start + judge_horizon_frames, count) - 1
                    values.append(
                        _rolling_action_value(
                            entry["current"],
                            ram[end],
                            positive_reward=any(raw_reward_events[start : end + 1]),
                            min_progress_delta=judge_min_progress_delta,
                        )
                    )

            for entry, action_value in zip(entries, values, strict=True):
                columns["X"].append(extract_features(
                    entry["current"],
                    entry["previous"],
                    previous_action=entry["previous_action"],
                    action_value=action_value,
                ))
                columns["y"].append(entry["action"])
                columns["source_paths"].append(entry["source_path"])
                columns["target_paths"].append(entry["target_path"])
                columns["episodes"].append(episode)
                columns["levels"].append("1-3")
                columns["frames"].append(entry["source_frame"])
                columns["outcomes"].append("win")
                columns["action_values"].append(action_value)
                columns["previous_actions"].append(entry["previous_action"])

    if selected_episodes is not None and found != selected_episodes:
        raise ValueError(f"Missing teacher episodes: {sorted(selected_episodes - found)}")
    if not columns["y"]:
        raise ValueError("No teacher decision rows were found")
    labels = np.asarray(columns["y"], dtype=np.int64)
    if value_mode == "trajectory-positive":
        value_rule = {
            "positive": "all_decisions_in_accepted_winning_teacher_trajectory"
        }
    elif value_mode == "student-judge":
        value_rule = {
            "positive": (
                "the_actions_own_future_horizon_contains_any_positive_reward_"
                "component_or_its_endpoint_advances_by_minimum_delta"
            ),
            "neutral": "no_positive_reward_or_sufficient_endpoint_progress_in_own_horizon",
            "negative": "not_present_in_success_only_teacher_trajectories",
        }
    else:
        value_rule = {
            "positive": (
                "four_frame_forward_delta_and_new_progress_max_or_score_gain_or_"
                "powerup_gain"
            ),
            "neutral": "no_observed_transition_reward",
            "negative": "not_present_in_success_only_teacher_trajectories",
        }
    metadata = {
        "schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "source_root": str(data.resolve()),
        "source_format": "teacher-npz-cache",
        "outcome": "win",
        "included_level": "1-3",
        "levels": ["1-3"],
        "label_offset": 1,
        "teacher_action_repeat": ACTION_REPEAT,
        "selection": "teacher-decision-boundaries-with-reset",
        "action_value_mode": value_mode,
        "min_progress_delta": (
            judge_min_progress_delta
            if value_mode == "student-judge"
            else min_progress_delta
        ),
        "judge_horizon_frames": (
            judge_horizon_frames if value_mode == "student-judge" else None
        ),
        "judge_capacity": None,
        "death_lookback_actions": (
            death_lookback_actions if value_mode == "student-judge" else None
        ),
        "action_value_rule": value_rule,
        "selected_rows": len(labels),
        "trajectory_count": len(found),
        "reset_state_rows": sum(frame == 0 for frame in columns["frames"]),
        "action_value_counts": dict(
            sorted(Counter(map(int, columns["action_values"])).items())
        ),
        "action_counts": dict(sorted(Counter(map(int, labels)).items())),
    }
    arrays = {
        "X": np.stack(columns["X"]),
        "y": labels,
        "source_paths": np.asarray(columns["source_paths"]),
        "target_paths": np.asarray(columns["target_paths"]),
        "episodes": np.asarray(columns["episodes"]),
        "levels": np.asarray(columns["levels"]),
        "frames": np.asarray(columns["frames"], dtype=np.int32),
        "outcomes": np.asarray(columns["outcomes"]),
        "action_values": np.asarray(columns["action_values"], dtype=np.int8),
        "previous_actions": np.asarray(columns["previous_actions"], dtype=np.uint8),
        "metadata": np.asarray(json.dumps(metadata)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare aligned DQN teacher decisions")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", action="append", dest="episodes")
    parser.add_argument("--min-progress-delta", type=int, default=3)
    parser.add_argument(
        "--value-mode",
        choices=("student-judge", "transition", "trajectory-positive"),
        default="student-judge",
    )
    parser.add_argument("--judge-horizon-frames", type=int, default=16)
    parser.add_argument("--judge-min-progress-delta", type=int, default=16)
    parser.add_argument("--death-lookback-actions", type=int, default=4)
    args = parser.parse_args(argv)
    result = prepare_teacher_decisions(
        args.data,
        args.output,
        episodes=args.episodes,
        min_progress_delta=args.min_progress_delta,
        value_mode=args.value_mode,
        judge_horizon_frames=args.judge_horizon_frames,
        judge_min_progress_delta=args.judge_min_progress_delta,
        death_lookback_actions=args.death_lookback_actions,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
