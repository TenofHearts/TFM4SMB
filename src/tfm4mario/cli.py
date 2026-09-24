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


def probability(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def gameplay_action(value):
    from .actions import validate_action

    try:
        return validate_action(int(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parser():
    p = argparse.ArgumentParser(description="TabPFN 3.5 Mario RAM imitation pipeline")
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser(
        "prepare", help="Extract a sampled table from selected trajectory PNGs"
    )
    prep.add_argument("--data", type=Path)
    prep.add_argument("--output", type=Path)
    prep.add_argument("--outcome", choices=["win", "fail", "all"], default="all")
    prep.add_argument("--stride", type=positive, default=1)
    prep.add_argument("--max-rows", type=positive, default=8192)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument(
        "--label-offset",
        type=int,
        choices=[0, 1],
        default=1,
        help="1: RAM[f] -> next action (default); 0: reconstruct applied action",
    )
    levels = prep.add_mutually_exclusive_group()
    levels.add_argument(
        "--include-level", help="Use only one world-level, for example 1-1"
    )
    levels.add_argument(
        "--exclude-level", help="Exclude one world-level, for example 8-4"
    )
    prep.add_argument(
        "--head-rows-per-trajectory",
        type=int,
        default=16,
        help="Prefer this many opening candidates per trajectory during selection",
    )
    prep.add_argument(
        "--pre-death-frames",
        type=positive,
        default=30,
        help="Label this many frames immediately before detected death as -1",
    )
    prep.add_argument(
        "--min-progress-delta",
        type=positive,
        default=3,
        help=(
            "Require this many pixels of forward movement in one frame, plus a "
            "new trajectory maximum, before labeling progress as successful"
        ),
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
    predict.add_argument(
        "--previous-action",
        type=gameplay_action,
        help=(
            "Action applied immediately before this RAM state; inferred from a "
            "dataset PNG and otherwise defaults to no-op"
        ),
    )
    predict.add_argument("--device", default="auto")
    ev = sub.add_parser(
        "evaluate", help="Evaluate on a separately prepared table you supply"
    )
    ev.add_argument("--model", type=Path)
    ev.add_argument("--data", type=Path)
    ev.add_argument("--device", default="auto")
    ev.add_argument("--batch-size", type=positive, default=128)
    def add_game_arguments(command, *, default_episodes):
        command.add_argument("--model", type=Path)
        command.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
        command.add_argument("--output", type=Path)
        command.add_argument("--device", default="auto")
        command.add_argument("--episodes", type=positive, default=default_episodes)
        command.add_argument("--max-frames", type=positive, default=18000)
        command.add_argument("--action-repeat", type=positive, default=1)
        command.add_argument(
            "--action-selection",
            choices=["sample", "epsilon_sample", "argmax"],
            default="sample",
        )
        command.add_argument("--epsilon", type=probability, default=0.3)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument(
            "--render", action=argparse.BooleanOptionalAction, default=False
        )
        command.add_argument(
            "--record-video", action=argparse.BooleanOptionalAction, default=False
        )
        command.add_argument("--video-fps", type=positive, default=60)

    play = sub.add_parser("play", help="Run a frozen-policy Mario rollout")
    add_game_arguments(play, default_episodes=1)
    adapt = sub.add_parser(
        "adapt", help="Run rolling delayed-credit collection and between-episode refits"
    )
    add_game_arguments(adapt, default_episodes=5)
    adapt.add_argument("--context", type=Path)
    adapt.add_argument("--online-cache", type=Path)
    adapt.add_argument(
        "--online-capacity",
        type=positive,
        default=16,
        help="Raw-frame future horizon used independently for every action",
    )
    adapt.add_argument("--online-death-lookback-actions", type=positive, default=4)
    adapt.add_argument(
        "--online-min-progress-delta",
        type=positive,
        default=3,
        help=(
            "Require this many pixels of advancement beyond the batch-start "
            "milestone before labeling an online batch successful"
        ),
    )
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
        "adapt": adapt,
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
        "adapt": ["model", "context", "online_cache", "output"],
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
            include_level=args.include_level,
            exclude_level=args.exclude_level,
            head_rows_per_trajectory=args.head_rows_per_trajectory,
            pre_death_frames=args.pre_death_frames,
            min_progress_delta=args.min_progress_delta,
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
    elif args.command in {"predict", "evaluate", "play", "adapt"}:
        from .policy import Policy, evaluate

        policy = Policy(args.model, args.device)
        if args.command == "predict":
            from .ram import parse_frame, read_frame

            if args.png:
                frame = parse_frame(args.png)
                ram = read_frame(frame, args.ram_encoding)
                previous_action = (
                    frame.action
                    if args.previous_action is None
                    else args.previous_action
                )
            else:
                ram = np.frombuffer(args.ram.read_bytes(), dtype=np.uint8)
                previous_action = args.previous_action or 0
            result = policy.predict_ram(ram, previous_action=previous_action)
        elif args.command == "evaluate":
            result = evaluate(policy, args.data, args.batch_size)
        else:
            from .game import play

            online = None
            if args.command == "adapt":
                from .online import OnlineReplay

                online = OnlineReplay(
                    policy,
                    args.context,
                    args.online_cache,
                    capacity=args.online_capacity,
                    min_progress_delta=args.online_min_progress_delta,
                    death_lookback_actions=args.online_death_lookback_actions,
                )

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
                epsilon=args.epsilon,
                record_video=args.record_video,
                video_fps=args.video_fps,
                online=online,
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
