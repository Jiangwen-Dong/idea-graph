from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.controller_eval_runtime import (
    load_run_manifest_rows,
    summarize_packet_runs,
    write_packet_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a saved controller-evaluation packet."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Directory containing run_manifest.jsonl.",
    )
    parser.add_argument(
        "--write-root",
        required=True,
        type=Path,
        help="Directory where summary artifacts are written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.input_root / "run_manifest.jsonl"
    rows = load_run_manifest_rows(manifest_path)
    summary = summarize_packet_runs(rows)
    write_packet_summary(args.write_root, summary)
    print(f"Wrote controller-evaluation summary for {len(rows)} runs to {args.write_root}")


if __name__ == "__main__":
    main()
