"""Synchronous nes-py rollout: emulator time pauses during policy inference."""

import json
from pathlib import Path
import time

import numpy as np

from .actions import to_nes_action
from .features import checked_ram


def make_env(env_id: str, render=False):
    try:
        import gymnasium as gym
        import gym_super_mario_bros  # noqa: F401 -- environment registration
    except ImportError as exc:
        raise RuntimeError("Install the game extra: uv sync --extra game") from exc
    env = gym.make(env_id, render_mode="human" if render else None)
    if getattr(env.action_space, "n", None) != 256:
        env.close()
        raise ValueError("Expected the raw 256-action environment, without JoypadSpace")
    return env


def get_ram(env):
    return checked_ram(env.unwrapped.ram).astype(np.uint8)


def reset_env(env, seed):
    result = env.reset(seed=seed)
    return result[1] if isinstance(result, tuple) and len(result) == 2 else {}


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        _, reward, terminated, truncated, info = result
        return float(reward), bool(terminated), bool(truncated), info
    if len(result) == 4:
        _, reward, done, info = result
        truncated = bool(info.get("TimeLimit.truncated", False))
        return float(reward), bool(done and not truncated), truncated, info
    raise ValueError("Unexpected environment step result")


def rollout(env, policy, *, max_frames=18000, action_repeat=1, seed=0, trace=None):
    if max_frames < 1 or action_repeat < 1:
        raise ValueError("max_frames/action_repeat must be positive")
    info = reset_env(env, seed)
    reward_total = 0.0
    frames = 0
    decisions = 0
    latencies = []
    terminated = truncated = False
    flag_get = bool(info.get("flag_get", False))
    started = time.perf_counter()
    while frames < max_frames and not (terminated or truncated or flag_get):
        decision = policy.predict_ram(get_ram(env))
        latencies.append(decision["predict_seconds"])
        decisions += 1
        # Never add a silent frameskip: the context's targets are per-frame.
        for _ in range(min(action_repeat, max_frames - frames)):
            reward, terminated, truncated, info = step_env(env, to_nes_action(decision["action"]))
            reward_total += reward
            frames += 1
            flag_get = flag_get or bool(info.get("flag_get", False))
            if terminated or truncated or flag_get:
                break
        if trace is not None:
            trace.write(json.dumps({"frame": frames, **decision, "reward_total": reward_total,
                                    "x_pos": info.get("x_pos"), "flag_get": flag_get}) + "\n")
            trace.flush()
    return {"frames": frames, "decisions": decisions, "reward": reward_total,
            "flag_get": flag_get, "terminated": terminated, "truncated": truncated,
            "frame_limit_reached": frames >= max_frames and not (terminated or truncated or flag_get),
            "wall_seconds": time.perf_counter() - started,
            "predict_p50_seconds": float(np.median(latencies)) if latencies else None,
            "predict_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
            "x_pos": info.get("x_pos"), "action_repeat": action_repeat}


def play(policy, env_id, output: Path, *, episodes=1, max_frames=18000,
         action_repeat=1, seed=0, render=False):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if output.exists():
        raise FileExistsError(f"Rollout directory exists: {output}")
    env = make_env(env_id, render)
    try:
        output.mkdir(parents=True)
        results = []
        for index in range(episodes):
            with (output / f"episode-{index:03d}.jsonl").open("w", encoding="utf-8") as trace:
                result = rollout(env, policy, max_frames=max_frames, action_repeat=action_repeat,
                                 seed=seed + index, trace=trace)
            results.append(result)
            (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
        return results
    finally:
        env.close()
