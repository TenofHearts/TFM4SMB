"""Batch-delayed online context for between-episode TabPFN adaptation."""

from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from .actions import validate_action
from .dataset import _action_value, _is_death_state, _player_x, load_table
from .features import FEATURE_NAMES, SCHEMA, checked_ram, extract_features


ONLINE_SCHEMA = "tfm4mario-online-batch-context-v3-progress-threshold"


class OnlineReplay:
    """Label complete transition batches from their ending state."""

    def __init__(
        self,
        policy,
        base_context: Path,
        path: Path,
        *,
        capacity=256,
        min_progress_delta=3,
    ):
        if capacity < 1 or min_progress_delta < 1:
            raise ValueError(
                "online capacity and minimum progress delta must be positive"
            )
        self.policy = policy
        self.base_context = Path(base_context)
        self.path = Path(path)
        self.capacity = int(capacity)
        self.min_progress_delta = int(min_progress_delta)
        self.base_X, self.base_y, self.base_metadata = load_table(self.base_context)
        self.context_X = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        self.context_y = np.empty(0, dtype=np.int64)
        self.context_values = np.empty(0, dtype=np.int8)
        self.pending = []
        self.batch_start_ram = None
        self.last_after_ram = None
        self.progress_max = None
        self.last_progress_reference = None
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
            or metadata.get("batch_capacity") != self.capacity
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
        self.batch_start_ram = initial
        self.last_after_ram = initial
        self.progress_max = _player_x(initial)
        self.last_progress_reference = self.progress_max
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
    ):
        if self.batch_start_ram is None:
            raise RuntimeError("begin_episode must be called before observe")
        previous = checked_ram(previous_ram).astype(np.uint8)
        current = checked_ram(current_ram).astype(np.uint8)
        after = checked_ram(after_ram).astype(np.uint8)
        action = validate_action(action)
        previous_action = validate_action(previous_action)
        self.pending.append((previous, current, previous_action, action))
        self.last_after_ram = after
        self.last_progress_reference = self.progress_max
        transition_death = bool(death or _is_death_state(after))
        self.episode_death = self.episode_death or transition_death
        if len(self.pending) == self.capacity:
            result = self._flush_pending(
                after,
                death=transition_death,
                progress_reference=self.last_progress_reference,
            )
            self.progress_max = max(self.progress_max, _player_x(after))
            return result
        self.progress_max = max(self.progress_max, _player_x(after))
        return None

    def end_episode(self, *, death=False, refit=True):
        if self.batch_start_ram is None:
            return None
        death = bool(death or self.episode_death)
        if self.pending:
            self._flush_pending(
                self.last_after_ram,
                death=death,
                progress_reference=self.last_progress_reference,
            )
        self.batch_start_ram = None
        self.last_after_ram = None
        self.progress_max = None
        self.last_progress_reference = None
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

    def _flush_pending(self, ending_ram, *, death, progress_reference):
        """Assign one ending-state value to every action in the full cache."""
        if not self.pending:
            return None
        ending = checked_ram(ending_ram).astype(np.uint8)
        value = (
            -1
            if death
            else _action_value(
                self.batch_start_ram,
                ending,
                progress_reference,
                effect_frame=0,
                death_frame=None,
                window=1,
                min_progress_delta=self.min_progress_delta,
            )
        )
        X = np.stack(
            [
                extract_features(
                    current,
                    previous,
                    previous_action=previous_action,
                    action_value=value,
                )
                for previous, current, previous_action, _ in self.pending
            ]
        )
        y = np.asarray(
            [action for _, _, _, action in self.pending], dtype=np.int64
        )
        values = np.full(len(y), value, dtype=np.int8)
        self.context_X = np.concatenate((self.context_X, X))
        self.context_y = np.concatenate((self.context_y, y))
        self.context_values = np.concatenate((self.context_values, values))
        flushed_rows = len(self.pending)
        self.pending.clear()
        self.batch_start_ram = ending
        self.episode_flushed_rows += flushed_rows
        self.episode_flushed_batches += 1
        self.flushed_batches += 1
        # Durably append the resolved batch, but leave the current policy frozen.
        self._write()
        return {
            "rows": flushed_rows,
            "assigned_action_value": value,
            "accumulated_context_rows": len(self.context_y),
        }

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": ONLINE_SCHEMA,
            "feature_schema": SCHEMA,
            "feature_names": list(FEATURE_NAMES),
            "context": self._context_identity(),
            "batch_capacity": self.capacity,
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
        os.replace(temporary, self.path)

    def _context_identity(self):
        return {
            "included_level": self.base_metadata.get("included_level"),
            "excluded_level": self.base_metadata.get("excluded_level"),
            "outcome": self.base_metadata.get("outcome"),
            "label_offset": self.base_metadata.get("label_offset"),
            "min_progress_delta": self.base_metadata.get("min_progress_delta"),
            "stride": self.base_metadata.get("stride"),
            "seed": self.base_metadata.get("seed"),
        }

    def _refit(self):
        X = np.concatenate((self.base_X, self.context_X))
        y = np.concatenate((self.base_y, self.context_y))
        return self.policy.refit_context(X, y)

    def summary(self):
        return {
            "enabled": True,
            "cache": str(self.path.resolve()),
            "batch_capacity": self.capacity,
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
