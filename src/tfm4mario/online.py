"""Batch-delayed online context for between-episode TabPFN adaptation."""

from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np

from .actions import validate_action
from .dataset import _is_death_state, _rolling_action_value, load_table
from .features import FEATURE_NAMES, SCHEMA, checked_ram, extract_features


ONLINE_SCHEMA = "tfm4mario-online-rolling-context-v5-per-action-frame-horizon"


class OnlineReplay:
    """Label each action independently after a fixed raw-frame horizon."""

    def __init__(
        self,
        policy,
        base_context: Path,
        path: Path,
        *,
        capacity=256,
        min_progress_delta=3,
        death_lookback_actions=4,
    ):
        if capacity < 1 or min_progress_delta < 1 or death_lookback_actions < 1:
            raise ValueError(
                "online horizon, progress delta, and death lookback must be positive"
            )
        self.policy = policy
        self.base_context = Path(base_context)
        self.path = Path(path)
        self.capacity = int(capacity)
        self.death_lookback_actions = int(death_lookback_actions)
        self.min_progress_delta = int(min_progress_delta)
        self.base_X, self.base_y, self.base_metadata = load_table(self.base_context)
        self.context_X = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        self.context_y = np.empty(0, dtype=np.int64)
        self.context_values = np.empty(0, dtype=np.int8)
        self.pending = []
        self.last_after_ram = None
        self.episode_death = False
        self.episode_flushed_rows = 0
        self.episode_flushed_batches = 0
        self.flushed_batches = 0
        self.updates = 0
        self.last_refit_seconds = None
        self._load()
        # Persisted rows come only from earlier episodes/invocations. Current
        # episode rows are never fitted by observe() or _flush_pending().
        if len(self.context_y):
            self.last_refit_seconds = self._refit()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"]))
                X = data["X"].copy()
                y = data["y"].copy()
                values = data["action_values"].copy()
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid online context: {self.path}") from error
        if (
            metadata.get("schema") != ONLINE_SCHEMA
            or metadata.get("feature_schema") != SCHEMA
            or metadata.get("feature_names") != list(FEATURE_NAMES)
            or metadata.get("context") != self._context_identity()
            or metadata.get("horizon_frames") != self.capacity
            or metadata.get("horizon_unit") != "raw_frames"
            or metadata.get("death_lookback_actions") != self.death_lookback_actions
            or metadata.get("min_progress_delta") != self.min_progress_delta
            or X.shape != (len(y), len(FEATURE_NAMES))
            or values.shape != (len(y),)
        ):
            raise ValueError(
                f"Incompatible online context: {self.path}. Choose a fresh "
                "online_cache path for batch-delayed adaptation."
            )
        for action in np.unique(y):
            validate_action(action)
        if not set(map(int, np.unique(values))).issubset({-1, 0, 1}):
            raise ValueError(f"Invalid action values in online context: {self.path}")
        self.context_X = X
        self.context_y = y
        self.context_values = values
        self.flushed_batches = int(metadata.get("flushed_batches", 0))

    def begin_episode(self, initial_ram):
        if self.pending:
            raise RuntimeError("Previous online cache was not flushed")
        initial = checked_ram(initial_ram).astype(np.uint8)
        self.last_after_ram = initial
        self.episode_death = False
        self.episode_flushed_rows = 0
        self.episode_flushed_batches = 0

    def observe(
        self,
        previous_ram,
        current_ram,
        action,
        after_ram,
        *,
        previous_action=0,
        death=False,
        elapsed_frames=1,
        positive_reward=False,
    ):
        if self.last_after_ram is None:
            raise RuntimeError("begin_episode must be called before observe")
        if elapsed_frames < 1:
            raise ValueError("elapsed_frames must be positive")
        previous = checked_ram(previous_ram).astype(np.uint8)
        current = checked_ram(current_ram).astype(np.uint8)
        after = checked_ram(after_ram).astype(np.uint8)
        action = validate_action(action)
        previous_action = validate_action(previous_action)
        self.pending.append({
            "previous": previous,
            "current": current,
            "previous_action": previous_action,
            "action": action,
            "elapsed_frames": 0,
            "positive_reward": False,
        })
        for item in self.pending:
            item["elapsed_frames"] += int(elapsed_frames)
            item["positive_reward"] = bool(
                item["positive_reward"] or positive_reward
            )
        self.last_after_ram = after
        transition_death = bool(death or _is_death_state(after))
        self.episode_death = self.episode_death or transition_death
        if transition_death:
            split = max(0, len(self.pending) - self.death_lookback_actions)
            mature = [
                item for item in self.pending[:split]
                if item["elapsed_frames"] >= self.capacity
            ]
            negative = self.pending[split:]
            self.pending.clear()
            results = []
            if mature:
                results.append(self._commit(mature, after))
            if negative:
                results.append(self._commit(negative, after, forced_value=-1))
            return self._merge_results(results)
        mature_count = 0
        for item in self.pending:
            if item["elapsed_frames"] < self.capacity:
                break
            mature_count += 1
        if not mature_count:
            return None
        mature = self.pending[:mature_count]
        del self.pending[:mature_count]
        return self._commit(mature, after)

    def end_episode(self, *, death=False, success=False, refit=True):
        if self.last_after_ram is None:
            return None
        death = bool(death or self.episode_death)
        if self.pending and death:
            negative = self.pending[-self.death_lookback_actions:]
            self._commit(negative, self.last_after_ram, forced_value=-1)
        elif self.pending and success:
            self._commit(self.pending, self.last_after_ram)
        self.pending.clear()
        self.last_after_ram = None
        self.episode_death = False
        should_refit = bool(refit and self.episode_flushed_rows)
        self.last_refit_seconds = self._refit() if should_refit else None
        self.updates += 1
        return {
            "committed_rows": self.episode_flushed_rows,
            "accumulated_context_rows": len(self.context_y),
            "pending_rows": len(self.pending),
            "flushed_batches": self.episode_flushed_batches,
            "total_flushed_batches": self.flushed_batches,
            "action_value_counts": dict(
                sorted(Counter(map(int, self.context_values)).items())
            ),
            "action_counts": dict(sorted(Counter(map(int, self.context_y)).items())),
            "refit_seconds": self.last_refit_seconds,
            "refit_performed": should_refit,
            "updates": self.updates,
        }

    def _commit(self, entries, ending_ram, *, forced_value=None):
        """Commit resolved rolling actions using their shared current endpoint."""
        if not entries:
            return None
        ending = checked_ram(ending_ram).astype(np.uint8)
        values = np.asarray([
            forced_value if forced_value is not None else _rolling_action_value(
                item["current"],
                ending,
                positive_reward=item["positive_reward"],
                min_progress_delta=self.min_progress_delta,
            )
            for item in entries
        ], dtype=np.int8)
        X = np.stack(
            [
                extract_features(
                    item["current"],
                    item["previous"],
                    previous_action=item["previous_action"],
                    action_value=int(value),
                )
                for item, value in zip(entries, values, strict=True)
            ]
        )
        y = np.asarray([item["action"] for item in entries], dtype=np.int64)
        self.context_X = np.concatenate((self.context_X, X))
        self.context_y = np.concatenate((self.context_y, y))
        self.context_values = np.concatenate((self.context_values, values))
        flushed_rows = len(entries)
        self.episode_flushed_rows += flushed_rows
        self.episode_flushed_batches += 1
        self.flushed_batches += 1
        # Durably append the resolved batch, but leave the current policy frozen.
        self._write()
        return {
            "rows": flushed_rows,
            "assigned_action_values": values.tolist(),
            "assigned_action_value": (
                int(values[0]) if np.all(values == values[0]) else None
            ),
            "accumulated_context_rows": len(self.context_y),
        }

    @staticmethod
    def _merge_results(results):
        results = [result for result in results if result is not None]
        if not results:
            return None
        values = [value for result in results for value in result["assigned_action_values"]]
        return {
            "rows": sum(result["rows"] for result in results),
            "assigned_action_values": values,
            "assigned_action_value": values[0] if len(set(values)) == 1 else None,
            "accumulated_context_rows": results[-1]["accumulated_context_rows"],
        }

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": ONLINE_SCHEMA,
            "feature_schema": SCHEMA,
            "feature_names": list(FEATURE_NAMES),
            "context": self._context_identity(),
            "horizon_frames": self.capacity,
            "horizon_unit": "raw_frames",
            "death_lookback_actions": self.death_lookback_actions,
            "min_progress_delta": self.min_progress_delta,
            "rows": len(self.context_y),
            "flushed_batches": self.flushed_batches,
            "action_value_counts": dict(
                sorted(Counter(map(int, self.context_values)).items())
            ),
            "action_counts": dict(sorted(Counter(map(int, self.context_y)).items())),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                X=self.context_X,
                y=self.context_y,
                action_values=self.context_values,
                metadata=np.asarray(json.dumps(metadata)),
            )
        for attempt in range(8):
            try:
                os.replace(temporary, self.path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.1 * (2 ** attempt))

    def _context_identity(self):
        return {
            "included_level": self.base_metadata.get("included_level"),
            "excluded_level": self.base_metadata.get("excluded_level"),
            "outcome": self.base_metadata.get("outcome"),
            "label_offset": self.base_metadata.get("label_offset"),
            "min_progress_delta": self.base_metadata.get("min_progress_delta"),
            "stride": self.base_metadata.get("stride"),
            "seed": self.base_metadata.get("seed"),
            "action_value_mode": self.base_metadata.get("action_value_mode"),
            "teacher_action_repeat": self.base_metadata.get("teacher_action_repeat"),
            "judge_horizon_frames": self.base_metadata.get("judge_horizon_frames"),
        }

    def _refit(self):
        X = np.concatenate((self.base_X, self.context_X))
        y = np.concatenate((self.base_y, self.context_y))
        return self.policy.refit_context(X, y)

    def summary(self):
        return {
            "enabled": True,
            "cache": str(self.path.resolve()),
            "horizon_frames": self.capacity,
            "death_lookback_actions": self.death_lookback_actions,
            "min_progress_delta": self.min_progress_delta,
            "pending_rows": len(self.pending),
            "accumulated_context_rows": len(self.context_y),
            "flushed_batches": self.flushed_batches,
            "updates": self.updates,
            "last_refit_seconds": self.last_refit_seconds,
            "action_value_counts": dict(
                sorted(Counter(map(int, self.context_values)).items())
            ),
            "action_counts": dict(sorted(Counter(map(int, self.context_y)).items())),
        }
