"""Validated TOML defaults; command-line flags have final precedence."""

import argparse
from pathlib import Path
import tomllib

PATH_TARGETS = {
    "selected_data": [("prepare", "data")],
    "context": [("prepare", "output"), ("train", "context"), ("adapt", "context")],
    "model": [
        ("train", "output"),
        ("predict", "model"),
        ("evaluate", "model"),
        ("play", "model"),
        ("adapt", "model"),
    ],
    "evaluation": [("evaluate", "data")],
    "rollout": [("play", "output")],
    "adaptive_rollout": [("adapt", "output")],
    "online_cache": [("adapt", "online_cache")],
}
RUNTIME_TARGETS = {
    "device": ["train", "predict", "evaluate", "play", "adapt"],
    "seed": ["prepare", "train", "play", "adapt"],
}


def apply_config(parser, commands, path: Path):
    if not path.is_file():
        parser.error(f"Config file does not exist: {path}")
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        parser.error(f"Cannot read {path}: {exc}")
    unknown = set(config) - set(commands) - {"paths", "runtime"}
    if unknown:
        parser.error(f"Unknown config sections: {sorted(unknown)}")
    defaults = {command: {} for command in commands}
    for section, settings in config.items():
        if not isinstance(settings, dict):
            parser.error(f"Config [{section}] must be a table")
        for name, value in settings.items():
            if section == "paths":
                if name not in PATH_TARGETS:
                    parser.error(f"Unknown config key: paths.{name}")
                for command, argument in PATH_TARGETS[name]:
                    defaults[command][argument] = value
            elif section == "runtime":
                if name not in RUNTIME_TARGETS:
                    parser.error(f"Unknown config key: runtime.{name}")
                for command in RUNTIME_TARGETS[name]:
                    defaults[command][name] = value
    # Per-command settings override shared paths/runtime regardless of TOML order.
    for command in commands:
        defaults[command].update(config.get(command, {}))
        actions = {
            action.dest: action
            for action in commands[command]._actions
            if action.dest not in {"help", "config"}
        }
        for name, value in defaults[command].items():
            if name not in actions:
                parser.error(f"Unknown config key: {command}.{name}")
            action = actions[name]
            try:
                if isinstance(action, argparse.BooleanOptionalAction):
                    if not isinstance(value, bool):
                        raise ValueError("must be a TOML boolean")
                elif action.type is Path:
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("must be a nonempty path string")
                    value = Path(value).expanduser()
                    if not value.is_absolute():
                        value = path.resolve().parent / value
                elif action.type is not None:
                    if action.type is float or action.type.__name__ in {
                        "probability",
                    }:
                        if not isinstance(value, (int, float)) or isinstance(value, bool):
                            raise ValueError("must be a TOML number")
                    elif not isinstance(value, int) or isinstance(value, bool):
                        raise ValueError("must be a TOML integer")
                    value = action.type(value)
                elif not isinstance(value, str):
                    raise ValueError("must be a string")
                if action.choices is not None and value not in action.choices:
                    raise ValueError(f"must be one of {action.choices}")
            except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
                parser.error(f"Invalid config {command}.{name}: {exc}")
            defaults[command][name] = value
        commands[command].set_defaults(**defaults[command])
    return config
