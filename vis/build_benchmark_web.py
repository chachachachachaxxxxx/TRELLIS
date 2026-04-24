#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis_edit.benchmark_pages import DEFAULT_BENCHMARK_ROOT, build_focus_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild benchmark web entrypoints.")
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=str(DEFAULT_BENCHMARK_ROOT),
        help="Benchmark root containing daily/ and focus/.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    focus_index = build_focus_index(benchmark_root)
    if focus_index is None:
        print(f"No focus manifest found under {benchmark_root / 'focus'}")
        return 1

    print(f"Focus index rebuilt: {focus_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
