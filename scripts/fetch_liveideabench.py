from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.benchmarks import download_liveideabench


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the liveideabench benchmark.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "data" / "benchmarks" / "liveideabench",
        help="Local cache directory for the benchmark.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = download_liveideabench(args.root, force=args.force)

    print("== Downloaded ==")
    print(f"Root: {paths.root}")
    print(f"CSV: {paths.csv_path}")
    print(f"README: {paths.readme_path}")
    print(f"Manifest: {paths.manifest_path}")


if __name__ == "__main__":
    main()
