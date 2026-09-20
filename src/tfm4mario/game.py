"""Synchronous nes-py rollout: emulator time pauses during policy inference."""

import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .actions import to_nes_action
from .features import checked_ram


def make_env(env_id: str, render=False, record_video=False):
    try:
        import gymnasium as gym
        import gym_super_mario_bros  # noqa: F401 -- environment registration
    except ImportError as exc:
        raise RuntimeError("Install the game extra: uv sync --extra game") from exc
    if render and record_video:
        raise ValueError("render and record_video cannot both be enabled")
    render_mode = "rgb_array" if record_video else ("human" if render else None)
    env = gym.make(env_id, render_mode=render_mode)
    if getattr(env.action_space, "n", None) != 256:
        env.close()
        raise ValueError("Expected the raw 256-action environment, without JoypadSpace")
    return env


class VideoRecorder:
    """Stream RGB emulator frames to FFmpeg without retaining them in RAM."""

    def __init__(self, path: Path, fps=60):
        self.path = Path(path)
        self.fps = fps
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            raise RuntimeError(
                "FFmpeg is required for video recording and was not found on PATH"
            )
        self.process = None
        self.log = None
        self.shape = None

    def write(self, frame):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected an RGB frame with shape (height, width, 3), got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 video frames, got {frame.dtype}")
        if self.process is None:
            self._start(frame.shape)
        elif frame.shape != self.shape:
            raise ValueError(
                f"Video frame shape changed from {self.shape} to {frame.shape}"
            )
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(
                f"FFmpeg stopped while writing {self.path}; see {self.log.name}"
            ) from exc

    def _start(self, shape):
        height, width, _ = shape
        self.shape = shape
        log_path = self.path.with_suffix(".ffmpeg.log")
        self.log = log_path.open("wb")
        command = [
            self.ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log
        )

    def close(self):
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        returncode = self.process.wait()
        self.log.close()
        if returncode:
            raise RuntimeError(
                f"FFmpeg exited with code {returncode}; see {self.log.name}"
            )


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


def rollout(
    env,
    policy,
    *,
    max_frames=18000,
    action_repeat=1,
    seed=0,
    action_selection="sample",
    trace=None,
    video=None,
):
    if max_frames < 1 or action_repeat < 1:
        raise ValueError("max_frames/action_repeat must be positive")
    info = reset_env(env, seed)
    if hasattr(policy, "reset_history"):
        policy.reset_history()
    reward_total = 0.0
    frames = 0
    decisions = 0
    latencies = []
    terminated = truncated = False
    flag_get = bool(info.get("flag_get", False))
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    previous_ram = None
    if video is not None:
        video.write(env.render())
    while frames < max_frames and not (terminated or truncated or flag_get):
        current_ram = get_ram(env)
        decision = policy.predict_ram(
            current_ram,
            previous_ram=previous_ram,
            success=1,
            selection=action_selection,
            rng=rng,
        )
        latencies.append(decision["predict_seconds"])
        decisions += 1
        # Never add a silent frameskip: the context's targets are per-frame.
        for _ in range(min(action_repeat, max_frames - frames)):
            # Preserve the state immediately before the last emulated step.  At
            # the next decision it is exactly one frame behind current RAM even
            # when action_repeat is greater than one.
            previous_ram = get_ram(env)
            reward, terminated, truncated, info = step_env(
                env, to_nes_action(decision["action"])
            )
            reward_total += reward
            frames += 1
            if video is not None:
                video.write(env.render())
            flag_get = flag_get or bool(info.get("flag_get", False))
            if terminated or truncated or flag_get:
                break
        if trace is not None:
            trace.write(
                json.dumps(
                    {
                        "frame": frames,
                        **decision,
                        "reward_total": reward_total,
                        "x_pos": info.get("x_pos"),
                        "flag_get": flag_get,
                    }
                )
                + "\n"
            )
            trace.flush()
    return {
        "frames": frames,
        "decisions": decisions,
        "reward": reward_total,
        "flag_get": flag_get,
        "terminated": terminated,
        "truncated": truncated,
        "frame_limit_reached": frames >= max_frames
        and not (terminated or truncated or flag_get),
        "wall_seconds": time.perf_counter() - started,
        "predict_p50_seconds": float(np.median(latencies)) if latencies else None,
        "predict_p95_seconds": (
            float(np.percentile(latencies, 95)) if latencies else None
        ),
        "x_pos": info.get("x_pos"),
        "action_repeat": action_repeat,
        "action_selection": action_selection,
    }


def play(
    policy,
    env_id,
    output: Path,
    *,
    episodes=1,
    max_frames=18000,
    action_repeat=1,
    seed=0,
    render=False,
    action_selection="sample",
    record_video=False,
    video_fps=60,
):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if output.exists():
        raise FileExistsError(f"Rollout directory exists: {output}")
    env = make_env(env_id, render, record_video)
    try:
        output.mkdir(parents=True)
        results = []
        for index in range(episodes):
            video_path = output / f"episode-{index:03d}.mp4"
            recorder = VideoRecorder(video_path, video_fps) if record_video else None
            try:
                with (output / f"episode-{index:03d}.jsonl").open(
                    "w", encoding="utf-8"
                ) as trace:
                    result = rollout(
                        env,
                        policy,
                        max_frames=max_frames,
                        action_repeat=action_repeat,
                        seed=seed + index,
                        action_selection=action_selection,
                        trace=trace,
                        video=recorder,
                    )
            finally:
                if recorder is not None:
                    recorder.close()
            if recorder is not None:
                result["video"] = str(video_path)
            results.append(result)
            (output / "summary.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            print(json.dumps(result), flush=True)
        return results
    finally:
        env.close()
