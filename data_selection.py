"""Extract the smbdataset PNG metadata once for fast repeated preparation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tfm4mario.metadata_cache import build_cache

DEFAULT_DATA = Path(r"D:\Downloads\Compressed\data")
DEFAULT_OUTPUT = Path(__file__).parent / "processed_data" / "metadata_cache"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache the RAM, controller, outcome, and identity metadata from smbdataset PNGs"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Extracted smbdataset directory (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"NPZ cache directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--outcome",
        choices=["win", "fail", "all"],
        default="win",
        help="Episodes to cache (default: win)",
    )
    parser.add_argument(
        "--ram-encoding", choices=["dataset-cr", "raw"], default="dataset-cr"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Concurrent PNG readers (default: based on CPU count)",
    )
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    result = build_cache(
        args.data,
        args.output,
        outcome=args.outcome,
        encoding=args.ram_encoding,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))
