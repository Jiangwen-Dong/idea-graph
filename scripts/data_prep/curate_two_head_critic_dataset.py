from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_curation import (
    DEFAULT_CURATED_DATASET_QUOTAS,
    curate_two_head_critic_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curate a gold two-head critic dataset from an existing parallel two-head package."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "graph_critic_datasets",
    )
    parser.add_argument("--dataset-name", type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = curate_two_head_critic_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        quotas=DEFAULT_CURATED_DATASET_QUOTAS,
    )
    print(f"Dataset directory: {result.dataset_dir}")
    print(f"Edit label count: {result.edit_label_count}")
    print(f"Commit label count: {result.commit_label_count}")


if __name__ == "__main__":
    main()
