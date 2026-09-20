"""Bounded, retrospectively labeled online context for TabPFN adaptation."""

from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from .actions import validate_action
from .dataset import _action_value, _is_death_state, _player_x, load_table
from .features import FEATURE_NAMES, SCHEMA, checked_ram, extract_features


ONLINE_SCHEMA = "tfm4mario-online-replay-v1"


class OnlineReplay:
    """Collect transitions, label them retrospectively, and refit a policy."""

    def __init__(
        self,
        policy,
        base_context: Path,
        path: Path,
        *,
        capacity=256,
        pre_death_frames=30,
    ):
        if capacity <= pre_death_frames:
            raise ValueError("online capacity must exceed pre_death_frames")
        self.policy = policy
        self.base_context = Path(base_context)
        self.path = Path(path)
        self.capacity = int(capacity)
        self.pre_death_frames = int(pre_death_frames)
        self.base_X, self.base_y, self.base_metadata = load_table(self.base_context)
        self.replay_X = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        self.replay_y = np.empty(0, dtype=np.int64)
        self.replay_values = np.empty(0, dtype=np.int8)
        self.pending = []
        self.staged_X = []
        self.staged_y = []
        self.staged_values = []
        self.progress_max = None
        self.updates = 0
        self.last_refit_seconds = None
        self.episode_death = False
        self._load()
        if len(self.replay_y):
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
            raise ValueError(f"Invalid online replay cache: {self.path}") from error
        if (
            metadata.get("schema") != ONLINE_SCHEMA
            or metadata.get("feature_schema") != SCHEMA
            or metadata.get("feature_names") != list(FEATURE_NAMES)
            or metadata.get("context") != self._context_identity()
            or X.shape != (len(y), len(FEATURE_NAMES))
            or values.shape != (len(y),)
        ):
            raise ValueError(f"Incompatible online replay cache: {self.path}")
        for action in np.unique(y):
            validate_action(action)
        if not set(map(int, np.unique(values))).issubset({-1, 0, 1}):
            raise ValueError(f"Invalid action values in online replay: {self.path}")
        self.replay_X = X[-self.capacity :]
        self.replay_y = y[-self.capacity :]
        self.replay_values = values[-self.capacity :]

    def begin_episode(self, initial_ram):
        if self.pending:
            raise RuntimeError("Previous online replay episode was not finalized")
        self.staged_X.clear()
        self.staged_y.clear()
        self.staged_values.clear()
        self.progress_max = _player_x(checked_ram(initial_ram))
        self.episode_death = False

    def observe(self, previous_ram, current_ram, action, after_ram, *, death=False):
        previous = checked_ram(previous_ram).astype(np.uint8)
        current = checked_ram(current_ram).astype(np.uint8)
        after = checked_ram(after_ram).astype(np.uint8)
        action = validate_action(action)
        if self.progress_max is None:
            self.progress_max = _player_x(current)
        positive = _action_value(
            current,
            after,
            self.progress_max,
            effect_frame=0,
            death_frame=None,
            window=self.pre_death_frames,
        )
        self.progress_max = max(self.progress_max, _player_x(after))
        self.episode_death = self.episode_death or bool(
            death or _is_death_state(after)
        )
        self.pending.append((previous, current, action, positive))
        # Resolve only transitions old enough that a future death cannot place
        # them inside the blame window. No model refit occurs during an episode.
        if len(self.pending) > self.pre_death_frames:
            self._stage(self.pending.pop(0))
        return None

    def end_episode(self, *, death=False, refit=True):
        if not self.pending and not self.staged_y:
            self.progress_max = None
            return None
        death = bool(death or self.episode_death)
        for previous, current, action, positive in self.pending:
            self._stage(
                (previous, current, action, -1 if death else positive)
            )
        self.pending.clear()
        X = np.stack(self.staged_X)
        y = np.asarray(self.staged_y, dtype=np.int64)
        values_array = np.asarray(self.staged_values, dtype=np.int8)
        self.replay_X = np.concatenate((self.replay_X, X))[-self.capacity :]
        self.replay_y = np.concatenate((self.replay_y, y))[-self.capacity :]
        self.replay_values = np.concatenate(
            (self.replay_values, values_array)
        )[-self.capacity :]
        self.progress_max = None
        self.episode_death = False
        self._write()
        self.last_refit_seconds = self._refit() if refit else None
        self.updates += 1
        committed_rows = len(y)
        self.staged_X.clear()
        self.staged_y.clear()
        self.staged_values.clear()
        return {
            "committed_rows": committed_rows,
            "cache_rows": len(self.replay_y),
            "pending_rows": len(self.pending),
            "action_value_counts": dict(
                sorted(Counter(map(int, self.replay_values)).items())
            ),
            "refit_seconds": self.last_refit_seconds,
            "refit_performed": bool(refit),
            "updates": self.updates,
        }

    def _stage(self, item):
        previous, current, action, value = item
        self.staged_X.append(
            extract_features(current, previous, action_value=value)
        )
        self.staged_y.append(action)
        self.staged_values.append(value)
        # Staging is bounded too. The replay is intentionally recent-experience
        # FIFO, so retain only the newest episode transitions.
        if len(self.staged_y) > self.capacity:
            self.staged_X.pop(0)
            self.staged_y.pop(0)
            self.staged_values.pop(0)

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": ONLINE_SCHEMA,
            "feature_schema": SCHEMA,
            "feature_names": list(FEATURE_NAMES),
            "context": self._context_identity(),
            "capacity": self.capacity,
            "pre_death_frames": self.pre_death_frames,
            "rows": len(self.replay_y),
            "action_value_counts": dict(
                sorted(Counter(map(int, self.replay_values)).items())
            ),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                X=self.replay_X,
                y=self.replay_y,
                action_values=self.replay_values,
                metadata=np.asarray(json.dumps(metadata)),
            )
        os.replace(temporary, self.path)

    def _context_identity(self):
        return {
            "included_level": self.base_metadata.get("included_level"),
            "excluded_level": self.base_metadata.get("excluded_level"),
            "outcome": self.base_metadata.get("outcome"),
            "label_offset": self.base_metadata.get("label_offset"),
        }

    def _refit(self):
        X = np.concatenate((self.base_X, self.replay_X))
        y = np.concatenate((self.base_y, self.replay_y))
        return self.policy.refit_context(X, y)

    def summary(self):
        return {
            "enabled": True,
            "cache": str(self.path.resolve()),
            "capacity": self.capacity,
            "cache_rows": len(self.replay_y),
            "pending_rows": len(self.pending),
            "updates": self.updates,
            "last_refit_seconds": self.last_refit_seconds,
            "action_value_counts": dict(
                sorted(Counter(map(int, self.replay_values)).items())
            ),
        }
