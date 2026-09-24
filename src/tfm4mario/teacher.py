"""Isolated rollout collection from the published Mario 1-3 DQN teacher."""

from collections import deque
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .actions import BUTTONS, to_nes_action
from .game import get_ram, make_env
from .metadata_cache import CACHE_SCHEMA, MANIFEST


TEACHER_ENV_ID = "SuperMarioBros-1-3-v0"
TEACHER_ACTIONS = (
    (),
    ("right",),
    ("right", "A"),
    ("right", "B"),
    ("right", "A", "B"),
    ("A",),
    ("left",),
)
DATASET_ACTIONS = tuple(
    sum(BUTTONS[button] for button in buttons) for buttons in TEACHER_ACTIONS
)
ACTION_REPEAT = 4
OBSERVATION_SIZE = 84


class TeacherDQN:
    """The checkpoint's original CNN with dependency-free preprocessing."""

    def __init__(self, checkpoint: Path, device="auto"):
        import torch
        from torch import nn

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, len(TEACHER_ACTIONS)),
        )

        # Sequential has no custom forward path between the named submodules.
        class Network(nn.Module):
            def __init__(self, features, fc):
                super().__init__()
                self.features = features
                self.fc = fc

            def forward(self, value):
                value = self.features(value).reshape(value.shape[0], -1)
                return self.fc(value)

        self.network = Network(features, fc).to(self.device)
        try:
            state = torch.load(
                Path(checkpoint), map_location=self.device, weights_only=True
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"Cannot safely load teacher checkpoint: {checkpoint}") from error
        self.network.load_state_dict(state, strict=True)
        self.network.eval()
        self.frames = deque(maxlen=4)

    def reset(self, observation):
        zero = np.zeros((OBSERVATION_SIZE, OBSERVATION_SIZE), dtype=np.float32)
        self.frames.clear()
        self.frames.extend([zero.copy(), zero.copy(), zero.copy()])
        self.frames.append(_preprocess(observation))

    def observe(self, observations):
        if not observations:
            return
        pooled = np.maximum.reduce(observations[-2:])
        self.frames.append(_preprocess(pooled))

    def action(self, rng, epsilon=0.0):
        import torch

        if len(self.frames) != 4:
            raise RuntimeError("Teacher frame stack has not been reset")
        if rng.random() < epsilon:
            return int(rng.integers(len(TEACHER_ACTIONS))), True
        value = torch.from_numpy(np.stack(self.frames)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            index = int(self.network(value).argmax(dim=1).item())
        return index, False


def _preprocess(observation):
    """Match RGB-to-gray plus 84x84 area downsampling without OpenCV."""
    rgb = np.asarray(observation)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an RGB observation, got {rgb.shape}")
    # OpenCV's RGB2GRAY produces uint8 before INTER_AREA resizing. Keeping that
    # quantization matters to a policy trained on the resulting pixel values.
    gray = np.rint(
        rgb[..., 0].astype(np.float32) * 0.299
        + rgb[..., 1].astype(np.float32) * 0.587
        + rgb[..., 2].astype(np.float32) * 0.114
    ).clip(0, 255)
    resized = _area_resize(gray, OBSERVATION_SIZE, OBSERVATION_SIZE)
    return np.rint(resized).clip(0, 255).astype(np.float32) / 255.0


def _area_weights(source, target):
    scale = source / target
    weights = np.zeros((target, source), dtype=np.float32)
    for destination in range(target):
        start, stop = destination * scale, (destination + 1) * scale
        first, last = int(np.floor(start)), int(np.ceil(stop))
        for position in range(first, last):
            overlap = min(stop, position + 1) - max(start, position)
            if overlap > 0:
                weights[destination, position] = overlap / scale
    return weights


def _area_resize(image, height, width):
    vertical = _area_weights(image.shape[0], height)
    horizontal = _area_weights(image.shape[1], width)
    return vertical @ image @ horizontal.T


def _reset(env, seed):
    result = env.reset(seed=seed)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _step(env, action):
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    elif len(result) == 4:
        observation, reward, done, info = result
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done and not truncated)
    else:
        raise ValueError("Unexpected environment step result")
    return observation, float(reward), bool(terminated), bool(truncated), info


def _rollout(env, teacher, *, seed, max_frames, epsilon):
    observation, info = _reset(env, seed)
    teacher.reset(observation)
    reset_ram = get_ram(env)
    rng = np.random.default_rng(seed)
    rams, actions, rewards = [], [], []
    frames = decisions = exploratory_decisions = 0
    reward_total = 0.0
    terminated = truncated = False
    flag_get = bool(info.get("flag_get", False))
    started = time.perf_counter()
    while frames < max_frames and not (terminated or truncated or flag_get):
        action_index, explored = teacher.action(rng, epsilon)
        dataset_action = DATASET_ACTIONS[action_index]
        observations = []
        decisions += 1
        exploratory_decisions += int(explored)
        for _ in range(min(ACTION_REPEAT, max_frames - frames)):
            observation, reward, terminated, truncated, info = _step(
                env, to_nes_action(dataset_action)
            )
            observations.append(observation)
            rams.append(get_ram(env))
            actions.append(dataset_action)
            rewards.append(reward)
            frames += 1
            reward_total += reward
            flag_get = flag_get or bool(info.get("flag_get", False))
            if terminated or truncated or flag_get:
                break
        teacher.observe(observations)
    return {
        "reset_ram": reset_ram,
        "ram": np.asarray(rams, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.uint8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "summary": {
            "seed": seed,
            "frames": frames,
            "decisions": decisions,
            "exploratory_decisions": exploratory_decisions,
            "reward": reward_total,
            "x_pos": info.get("x_pos"),
            "flag_get": flag_get,
            "terminated": terminated,
            "truncated": truncated,
            "frame_limit_reached": frames >= max_frames
            and not (terminated or truncated or flag_get),
            "wall_seconds": time.perf_counter() - started,
        },
    }


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_shard(output: Path, trajectory, index):
    episode = f"pretrained-dqn-1-3-{index:03d}"
    ram, actions = trajectory["ram"], trajectory["actions"]
    numbers = np.arange(1, len(actions) + 1, dtype=np.int32)
    paths = np.asarray(
        [f"teacher/{episode}/frame-{number:06d}.win.png" for number in numbers]
    )
    metadata = {
        "schema": CACHE_SCHEMA,
        "episode": episode,
        "outcome": "win",
        "world": "1",
        "level": "3",
        "ram_encoding": "raw",
        "frames": len(actions),
        "source": "roclark/super-mario-bros-dqn",
        "teacher_action_repeat": ACTION_REPEAT,
    }
    target = output / f"{episode}.win.npz"
    temporary = target.with_suffix(".tmp")
    arrays = dict(
        ram=ram,
        actions=actions,
        frames=numbers,
        paths=paths,
        metadata=np.asarray(json.dumps(metadata)),
    )
    if "reset_ram" in trajectory:
        arrays["reset_ram"] = trajectory["reset_ram"]
    if "rewards" in trajectory:
        arrays["rewards"] = trajectory["rewards"]
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, target)


def collect_teacher(
    checkpoint: Path,
    output: Path,
    *,
    trajectories=3,
    max_attempts=12,
    max_frames=18000,
    epsilon=0.01,
    seed=0,
    device="auto",
):
    """Collect successful, action-distinct 1-3 demonstrations into an NPZ cache."""
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new path: {output}")
    if trajectories < 1 or max_attempts < trajectories or max_frames < 1:
        raise ValueError(
            "trajectories/max_frames must be positive and max_attempts must cover trajectories"
        )
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    teacher = TeacherDQN(checkpoint, device)
    env = make_env(TEACHER_ENV_ID)
    output.mkdir(parents=True)
    manifest_path = output / MANIFEST
    manifest = {
        "schema": CACHE_SCHEMA,
        "outcome": "win",
        "ram_encoding": "raw",
        "complete": False,
        "source": "roclark/super-mario-bros-dqn",
        "environment": TEACHER_ENV_ID,
        "teacher_action_repeat": ACTION_REPEAT,
        "teacher_epsilon": epsilon,
        "requested_trajectories": trajectories,
        "attempts": [],
        "shards": 0,
        "frames": 0,
    }
    _write_json(manifest_path, manifest)
    accepted_actions = []
    try:
        for attempt in range(max_attempts):
            trajectory = _rollout(
                env,
                teacher,
                seed=seed + attempt,
                max_frames=max_frames,
                epsilon=epsilon,
            )
            summary = trajectory["summary"]
            duplicate = any(
                np.array_equal(trajectory["actions"], previous)
                for previous in accepted_actions
            )
            accepted = bool(summary["flag_get"] and not duplicate)
            summary.update({"duplicate": duplicate, "accepted": accepted})
            manifest["attempts"].append(summary)
            if accepted:
                _write_shard(output, trajectory, len(accepted_actions))
                accepted_actions.append(trajectory["actions"].copy())
                manifest["shards"] = len(accepted_actions)
                manifest["frames"] += summary["frames"]
            _write_json(manifest_path, manifest)
            print(json.dumps(summary), flush=True)
            if len(accepted_actions) == trajectories:
                break
    finally:
        env.close()
    if len(accepted_actions) != trajectories:
        raise RuntimeError(
            f"Collected {len(accepted_actions)}/{trajectories} successful unique "
            f"trajectories in {max_attempts} attempts; partial cache remains at {output}"
        )
    manifest["complete"] = True
    _write_json(manifest_path, manifest)
    return {
        "cache": str(output.resolve()),
        "teacher": manifest["source"],
        "environment": TEACHER_ENV_ID,
        "trajectories": manifest["shards"],
        "frames": manifest["frames"],
        "attempts": len(manifest["attempts"]),
        "epsilon": epsilon,
        "device": str(teacher.device),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect pretrained DQN Mario 1-3 demonstrations")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=18000)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    result = collect_teacher(
        args.checkpoint,
        args.output,
        trajectories=args.trajectories,
        max_attempts=args.max_attempts,
        max_frames=args.max_frames,
        epsilon=args.epsilon,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
