from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.benchmarks import download_ai_idea_bench_2025


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download AI Idea Bench 2025 assets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "data" / "benchmarks" / "ai_idea_bench_2025",
        help="Local cache directory for the benchmark.",
    )
    parser.add_argument(
        "--include-paper-assets",
        action="store_true",
        help="Also download the official paper archive zip. This is about 31 GB.",
    )
    parser.add_argument(
        "--extract-paper-assets",
        action="store_true",
        help="Extract the downloaded paper archive into papers_data/. Requires --include-paper-assets.",
    )
    parser.add_argument(
        "--allow-large-download",
        action="store_true",
        help="Required to confirm that you really want the large paper-asset download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.extract_paper_assets and not args.include_paper_assets:
        raise SystemExit("--extract-paper-assets requires --include-paper-assets.")
    if args.include_paper_assets and not args.allow_large_download:
        raise SystemExit(
            "--include-paper-assets is blocked by default because the archive is about 31 GB. "
            "Re-run with --allow-large-download if you really want it."
        )

    paths = download_ai_idea_bench_2025(
        args.root,
        include_papers=args.include_paper_assets,
        extract_papers=args.extract_paper_assets,
        force=args.force,
    )

    print("== Downloaded ==")
    print(f"Root: {paths.root}")
    print(f"Metadata: {paths.metadata_path}")
    print(f"README: {paths.readme_path}")
    print(f"Manifest: {paths.manifest_path}")
    if args.include_paper_assets:
        print(f"Papers archive: {paths.papers_archive_path}")
    if args.extract_paper_assets:
        print(f"Papers dir: {paths.papers_dir}")


if __name__ == "__main__":
    main()
