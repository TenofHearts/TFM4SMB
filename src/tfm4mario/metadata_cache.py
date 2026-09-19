"""Persistent NPZ cache of smbdataset's PNG metadata."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os

import numpy as np

from .ram import Frame, OutcomeMismatchError, parse_frame, read_frame

CACHE_SCHEMA = "tfm4mario-smbdataset-metadata-v1"
MANIFEST = "manifest.json"


def _write_manifest(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid metadata-cache manifest: {path}") from error


def _valid_shard(path: Path, expected: dict) -> int | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            info = json.loads(str(data["metadata"]))
            count = len(data["frames"])
            if (
                info.get("schema") != CACHE_SCHEMA
                or any(info.get(key) != value for key, value in expected.items())
                or data["actions"].shape != (count,)
                or data["paths"].shape != (count,)
            ):
                return None
            return count
    except OSError, ValueError, KeyError, json.JSONDecodeError:
        return None


def build_cache(
    root: Path,
    output: Path,
    *,
    outcome: str = "win",
    encoding: str = "dataset-cr",
    workers: int | None = None,
) -> dict:
    """Extract PNG metadata into resumable, per-episode compressed NPZ files."""
    root, output = Path(root), Path(output)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")
    if outcome not in {"win", "fail", "all"}:
        raise ValueError("outcome must be win, fail, or all")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST
    settings = {"schema": CACHE_SCHEMA, "outcome": outcome, "ram_encoding": encoding}
    if manifest_path.exists():
        current = _read_manifest(manifest_path)
        mismatches = {
            key: (current.get(key), value)
            for key, value in settings.items()
            if current.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing cache has incompatible settings: {mismatches}")
    manifest = {**settings, "complete": False, "shards": 0, "frames": 0}
    _write_manifest(manifest_path, manifest)

    directories = [path for path in root.iterdir() if path.is_dir()]
    if not directories:
        directories = [root]
    episode_groups = []
    for directory in sorted(directories):
        groups = {}
        for path in directory.glob("*.png"):
            frame = parse_frame(path)
            if outcome == "all" or frame.outcome == outcome:
                groups.setdefault((frame.episode, frame.outcome), []).append(frame)
        for key, frames in groups.items():
            frames.sort(key=lambda frame: frame.number)
            if len({frame.number for frame in frames}) != len(frames):
                raise ValueError(f"Duplicate frame numbers in {key}")
            episode_groups.append(frames)
    episode_groups.sort(key=lambda frames: (frames[0].episode, frames[0].outcome))
    if not episode_groups:
        raise ValueError(f"No {outcome} trajectory PNGs under {root}")

    worker_count = workers or min(32, (os.cpu_count() or 1) + 4)
    total_frames = 0
    new_shards = 0
    cached_episodes = 0
    skipped_trajectories = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for index, frames in enumerate(episode_groups, 1):
            first = frames[0]
            shard = output / f"{first.episode}.{first.outcome}.npz"
            expected = {
                "episode": first.episode,
                "outcome": first.outcome,
                "world": first.world,
                "level": first.level,
                "ram_encoding": encoding,
            }
            cached_count = _valid_shard(shard, expected) if shard.exists() else None
            if cached_count is None:
                try:
                    ram = np.stack(
                        list(
                            executor.map(
                                lambda frame: read_frame(frame, encoding),
                                frames,
                                buffersize=max(worker_count * 2, 1),
                            )
                        )
                    )
                except OutcomeMismatchError as error:
                    skipped = {
                        "episode": first.episode,
                        "outcome": first.outcome,
                        "frames": len(frames),
                        "reason": str(error),
                    }
                    skipped_trajectories.append(skipped)
                    print(
                        f"Discarded trajectory {first.episode}.{first.outcome} "
                        f"({len(frames):,} frames): embedded OUTCOME mismatch",
                        flush=True,
                    )
                    continue
                paths = np.asarray(
                    [frame.path.relative_to(root).as_posix() for frame in frames]
                )
                actions = np.asarray([frame.action for frame in frames], dtype=np.uint8)
                numbers = np.asarray([frame.number for frame in frames], dtype=np.int32)
                info = {"schema": CACHE_SCHEMA, **expected, "frames": len(frames)}
                temporary = shard.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        ram=ram,
                        actions=actions,
                        frames=numbers,
                        paths=paths,
                        metadata=np.asarray(json.dumps(info)),
                    )
                os.replace(temporary, shard)
                cached_count = len(frames)
                new_shards += 1
            total_frames += cached_count
            cached_episodes += 1
            print(
                f"Cached {index:,}/{len(episode_groups):,} episodes "
                f"({total_frames:,} frames)",
                flush=True,
            )

    manifest.update(
        {
            "complete": True,
            "shards": cached_episodes,
            "frames": total_frames,
            "skipped_trajectories": skipped_trajectories,
        }
    )
    _write_manifest(manifest_path, manifest)
    return {
        "cache": str(output.resolve()),
        "outcome": outcome,
        "ram_encoding": encoding,
        "episodes": cached_episodes,
        "frames": total_frames,
        "new_shards": new_shards,
        "skipped_trajectories": skipped_trajectories,
    }


class MetadataCache:
    """Read-only access to a complete directory of NPZ episode shards."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.info = _read_manifest(self.path / MANIFEST)
        if (
            self.info.get("schema") != CACHE_SCHEMA
            or self.info.get("complete") is not True
        ):
            raise ValueError(
                f"Metadata cache is incomplete or has an unsupported schema: {self.path}"
            )
        self._locations = {}
        self._loaded_shard = None
        self._loaded_ram = None

    def close(self):
        self._loaded_shard = None
        self._loaded_ram = None

    def frames(self, outcome: str):
        result = []
        self._locations.clear()
        for shard in sorted(self.path.glob("*.npz")):
            with np.load(shard, allow_pickle=False) as data:
                info = json.loads(str(data["metadata"]))
                if info.get("schema") != CACHE_SCHEMA:
                    raise ValueError(f"Unsupported cache shard: {shard}")
                if outcome != "all" and info["outcome"] != outcome:
                    continue
                paths, numbers, actions = data["paths"], data["frames"], data["actions"]
                for index in range(len(numbers)):
                    frame = Frame(
                        Path(str(paths[index])),
                        info["episode"],
                        info["world"],
                        info["level"],
                        int(numbers[index]),
                        int(actions[index]),
                        info["outcome"],
                    )
                    result.append(frame)
                    self._locations[frame.path.as_posix()] = (shard, index)
        result.sort(key=lambda frame: (frame.episode, frame.number))
        return result

    def ram(self, frame: Frame):
        location = self._locations.get(frame.path.as_posix())
        if location is None:
            raise ValueError(f"Missing cached RAM: {frame.path}")
        shard, index = location
        if shard != self._loaded_shard:
            with np.load(shard, allow_pickle=False) as data:
                self._loaded_ram = data["ram"].copy()
            if self._loaded_ram.ndim != 2 or self._loaded_ram.shape[1] != 2048:
                raise ValueError(f"Invalid RAM array in cache shard: {shard}")
            self._loaded_shard = shard
        return self._loaded_ram[index].copy()
