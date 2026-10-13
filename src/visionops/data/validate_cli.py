"""Command-line entry point for dataset validation."""

import argparse
import json
import sys

from .validate import validate_imagefolder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an ImageFolder dataset.")
    parser.add_argument("--root", required=True, help="Dataset directory to validate.")
    parser.add_argument(
        "--output",
        help="Optional path for a JSON report. The report is always printed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_imagefolder(args.root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(rendered + "\n")

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
