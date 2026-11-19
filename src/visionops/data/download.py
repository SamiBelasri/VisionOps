"""Explicit dataset preparation commands."""

import argparse
from pathlib import Path

from .datasets import build_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a VisionOps dataset.")
    parser.add_argument("--dataset", choices=["eurosat"], required=True)
    parser.add_argument("--root", required=True, help="Destination directory.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow downloading the dataset from its upstream source.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    dataset = build_dataset(
        args.dataset,
        root,
        train=True,
        download=args.download,
    )
    print(f"Prepared dataset: {args.dataset}")
    print(f"Root: {root}")
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {len(dataset.classes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
