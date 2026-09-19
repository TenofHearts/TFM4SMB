"""CLI entry point; preprocessing does not import Torch or download weights."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parser():
    p = argparse.ArgumentParser(description="TabPFN 3.5 Mario RAM imitation pipeline")
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser(
        "prepare", help="Extract a sampled table from selected trajectory PNGs"
    )
    prep.add_argument("--data", type=Path)
    prep.add_argument("--output", type=Path)
    prep.add_argument("--outcome", choices=["win", "fail", "all"], default="win")
    prep.add_argument("--stride", type=positive, default=4)
    prep.add_argument("--max-rows", type=positive, default=8192)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument(
        "--label-offset",
        type=int,
        choices=[0, 1],
        default=1,
        help="1: RAM[f] -> next action (default); 0: reconstruct applied action",
    )
    prep.add_argument(
        "--exclude-level", help="Exclude one world-level, for example 8-4"
    )
    prep.add_argument(
        "--head-rows-per-trajectory",
        type=int,
        default=16,
        help="Keep this many opening candidates per trajectory before reservoir sampling",
    )
    prep.add_argument(
        "--ram-encoding", choices=["dataset-cr", "raw"], default="dataset-cr"
    )
    fit = sub.add_parser(
        "train", help="Fit pretrained v3.5 on the context (no gradient updates)"
    )
    fit.add_argument("--context", type=Path)
    fit.add_argument("--output", type=Path)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--n-estimators", type=positive, default=1)
    fit.add_argument(
        "--fit-mode",
        choices=["fit_with_cache", "fit_preprocessors", "low_memory"],
        default="fit_with_cache",
    )
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument(
        "--model-path", type=Path, help="Optional local v3.5 safetensors checkpoint"
    )
    predict = sub.add_parser(
        "predict", help="Predict from a dataset PNG or raw 2048-byte RAM dump"
    )
    predict.add_argument("--model", type=Path)
    inputs = predict.add_mutually_exclusive_group()
    inputs.add_argument("--png", type=Path)
    inputs.add_argument("--ram", type=Path)
    predict.add_argument(
        "--ram-encoding", choices=["dataset-cr", "raw"], default="dataset-cr"
    )
    predict.add_argument("--device", default="auto")
    ev = sub.add_parser(
        "evaluate", help="Evaluate on a separately prepared table you supply"
    )
    ev.add_argument("--model", type=Path)
    ev.add_argument("--data", type=Path)
    ev.add_argument("--device", default="auto")
    ev.add_argument("--batch-size", type=positive, default=128)
    play = sub.add_parser("play", help="Run a synchronous Mario rollout")
    play.add_argument("--model", type=Path)
    play.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    play.add_argument("--output", type=Path)
    play.add_argument("--device", default="auto")
    play.add_argument("--episodes", type=positive, default=1)
    play.add_argument("--max-frames", type=positive, default=18000)
    play.add_argument("--action-repeat", type=positive, default=1)
    play.add_argument(
        "--action-selection", choices=["sample", "argmax"], default="sample"
    )
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    play.add_argument(
        "--record-video", action=argparse.BooleanOptionalAction, default=False
    )
    play.add_argument("--video-fps", type=positive, default=60)
    doctor = sub.add_parser(
        "doctor", help="Inspect compute and optionally exercise an emulator"
    )
    doctor.add_argument(
        "--env-id", help="Reset and step this environment; no model needed"
    )
    commands = {
        "prepare": prep,
        "train": fit,
        "predict": predict,
        "evaluate": ev,
        "play": play,
        "doctor": doctor,
    }
    for command in commands.values():
        command.add_argument(
            "--config",
            type=Path,
            default=argparse.SUPPRESS,
            help="TOML configuration (default: config.toml)",
        )
    return p, commands


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p, commands = parser()
    if "--help" in argv or "-h" in argv:
        return p.parse_args(argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path("config.toml"))
    config_path = config_parser.parse_known_args(argv)[0].config
    from .config import apply_config

    apply_config(p, commands, config_path)
    args = p.parse_args(argv)
    required = {
        "prepare": ["data", "output"],
        "train": ["context", "output"],
        "predict": ["model"],
        "evaluate": ["model", "data"],
        "play": ["model", "output"],
        "doctor": [],
    }
    for name in required[args.command]:
        if getattr(args, name, None) is None:
            p.error(f"Set {args.command}.{name} in config or pass --{name}")
    if args.command == "predict":
        # Explicit CLI input replaces the configured alternative input.
        if any(arg == "--png" or arg.startswith("--png=") for arg in argv):
            args.ram = None
        if any(arg == "--ram" or arg.startswith("--ram=") for arg in argv):
            args.png = None
        if bool(args.png) == bool(args.ram):
            p.error("Set exactly one of predict.ram or predict.png")
    return args


def concise(value):
    if isinstance(value, dict):
        return {
            key: (len(item) if key == "feature_names" else concise(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [concise(item) for item in value]
    return value


def main():
    args = parse_args()
    if args.command == "prepare":
        from .dataset import prepare

        result = prepare(
            args.data,
            args.output,
            outcome=args.outcome,
            stride=args.stride,
            max_rows=args.max_rows,
            seed=args.seed,
            label_offset=args.label_offset,
            encoding=args.ram_encoding,
            exclude_level=args.exclude_level,
            head_rows_per_trajectory=args.head_rows_per_trajectory,
        )
    elif args.command == "train":
        from .policy import train

        result = train(
            args.context,
            args.output,
            device=args.device,
            n_estimators=args.n_estimators,
            fit_mode=args.fit_mode,
            seed=args.seed,
            model_path=args.model_path,
        )
    elif args.command in {"predict", "evaluate", "play"}:
        from .policy import Policy, evaluate

        policy = Policy(args.model, args.device)
        if args.command == "predict":
            from .ram import parse_frame, read_frame

            ram = (
                read_frame(parse_frame(args.png), args.ram_encoding)
                if args.png
                else np.frombuffer(args.ram.read_bytes(), dtype=np.uint8)
            )
            result = policy.predict_ram(ram)
        elif args.command == "evaluate":
            result = evaluate(policy, args.data, args.batch_size)
        else:
            from .game import play

            result = play(
                policy,
                args.env_id,
                args.output,
                episodes=args.episodes,
                max_frames=args.max_frames,
                action_repeat=args.action_repeat,
                seed=args.seed,
                render=args.render,
                action_selection=args.action_selection,
                record_video=args.record_video,
                video_fps=args.video_fps,
            )
    else:
        import torch

        result = {
            "torch": torch.__version__,
            "tabpfn": importlib.metadata.version("tabpfn"),
            "cuda_available": torch.cuda.is_available(),
            "gpus": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "vram_gib": torch.cuda.get_device_properties(i).total_memory
                    / 1024**3,
                }
                for i in range(torch.cuda.device_count())
            ],
        }
        if args.env_id:
            from .game import make_env, get_ram, reset_env, step_env
            from .features import extract_features

            env = make_env(args.env_id)
            try:
                reset_env(env, 0)
                step_env(env, 0)
                ram = get_ram(env)
                result["environment"] = {
                    "id": args.env_id,
                    "ram_bytes": len(ram),
                    "features": len(extract_features(ram)),
                }
            finally:
                env.close()
    print(json.dumps(concise(result), indent=2))
