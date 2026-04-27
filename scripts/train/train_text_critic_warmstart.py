from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import write_text_file
from idea_graph.online_text_critic import (
    NAMESPACE_NAMES,
    load_candidate_rows,
    load_partition_manifest_rows,
    train_warmstart_text_critic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the offline warm-start text critic.")
    parser.add_argument(
        "--candidate-dataset-dir",
        type=Path,
        required=True,
        help="Directory containing candidate_dataset.jsonl.",
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
        help="Directory where warm-start artifacts will be written.",
    )
    parser.add_argument(
        "--required-namespace",
        action="append",
        default=[],
        help="Required supervision namespace. Repeatable. Defaults to teacher_logged and terminal_commit.",
    )
    parser.add_argument(
        "--commit-positive-weight",
        type=float,
        default=2.0,
        help="Sample-weight multiplier for positive terminal_commit examples.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    required_namespaces = args.required_namespace or list(NAMESPACE_NAMES)
    candidate_rows = load_candidate_rows(args.candidate_dataset_dir)
    partition_rows = load_partition_manifest_rows(args.partition_manifest)
    model, bundle, metrics = train_warmstart_text_critic(
        candidate_rows,
        partition_rows,
        required_namespaces=required_namespaces,
        commit_positive_weight=args.commit_positive_weight,
    )

    namespace_support = bundle.namespace_support
    metadata = {
        "candidate_dataset_path": str(Path(args.candidate_dataset_dir).resolve()),
        "partition_manifest_path": str(Path(args.partition_manifest).resolve()),
        "required_namespaces": list(required_namespaces),
        "train_group_count": bundle.split_audit["train_group_count"],
        "validation_group_count": bundle.split_audit["validation_group_count"],
        "group_overlap_count": bundle.split_audit["group_overlap_count"],
        "train_commit_candidate_count": namespace_support["critic_train"]["terminal_commit"]["row_count"],
        "train_commit_positive_count": namespace_support["critic_train"]["terminal_commit"]["positive_count"],
        "validation_commit_candidate_count": namespace_support["critic_dev"]["terminal_commit"]["row_count"],
        "validation_commit_positive_count": namespace_support["critic_dev"]["terminal_commit"]["positive_count"],
        "namespace_support": namespace_support,
        "feature_policy": "Warm-start uses frozen TF-IDF text features and a weighted logistic-regression head.",
        "warmstart_framing": (
            "This offline warm-start remains a lightweight scorer-based controller pilot. "
            "It uses critic_train/critic_dev only and ignores paper_eval entirely."
        ),
    }
    training_config = {
        "required_namespaces": list(required_namespaces),
        "commit_positive_weight": float(args.commit_positive_weight),
        "train_partition_role": "critic_train",
        "validation_partition_role": "critic_dev",
        "ignored_partition_roles": ["paper_eval"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (Path(args.output_dir) / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    write_text_file(Path(args.output_dir) / "metrics.json", json.dumps(metrics, indent=2))
    write_text_file(Path(args.output_dir) / "metadata.json", json.dumps(metadata, indent=2))
    write_text_file(
        Path(args.output_dir) / "training_config.json",
        json.dumps(training_config, indent=2),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
