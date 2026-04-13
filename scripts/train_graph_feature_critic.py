from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.graph_feature_critic import (
    build_graph_feature_examples,
    train_graph_feature_critic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the lightweight offline graph-feature critic."
    )
    parser.add_argument(
        "--candidate-dataset-dir",
        type=Path,
        required=True,
        help="Directory containing candidate_dataset.jsonl.",
    )
    parser.add_argument(
        "--g1-dataset-dir",
        type=Path,
        required=True,
        help="Directory containing trajectory_examples.jsonl and terminal_state_manifest.jsonl.",
    )
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        required=True,
        help="Path to partition_manifest.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where model.pkl, metrics.json, and metadata.json will be written.",
    )
    parser.add_argument(
        "--commit-positive-weight",
        type=float,
        default=2.0,
        help="Sample-weight multiplier for positive terminal commit examples.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = build_graph_feature_examples(
        candidate_dataset_dir=args.candidate_dataset_dir,
        g1_dataset_dir=args.g1_dataset_dir,
        partition_manifest_path=args.partition_manifest,
    )
    result = train_graph_feature_critic(
        dataset=dataset,
        output_dir=args.output_dir,
        commit_positive_weight=args.commit_positive_weight,
    )
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
