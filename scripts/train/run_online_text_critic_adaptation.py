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
    evaluate_warmstart_text_critic,
    load_candidate_rows,
    load_partition_manifest_rows,
    partition_rows_for_role,
    train_online_text_critic_adaptation,
    build_partition_role_lookup,
    _load_jsonl,
    _candidate_example_from_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny online adaptation pass for the text critic.")
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
        "--online-buffer",
        type=Path,
        required=True,
        help="JSONL file containing critic_train replay rows in candidate-slate format.",
    )
    parser.add_argument(
        "--warmstart-model",
        type=Path,
        required=True,
        help="Path to the offline warm-start model.pkl artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where adaptation artifacts will be written.",
    )
    parser.add_argument(
        "--offline-fraction",
        type=float,
        default=0.7,
        help="Target offline fraction inside the mixed training buffer.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Maximum number of mixed training examples to use.",
    )
    parser.add_argument(
        "--commit-positive-weight",
        type=float,
        default=2.0,
        help="Sample-weight multiplier for positive commit rows.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for mixed-buffer sampling.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidate_rows = load_candidate_rows(args.candidate_dataset_dir)
    partition_rows = load_partition_manifest_rows(args.partition_manifest)
    online_rows = _load_jsonl(args.online_buffer)

    with Path(args.warmstart_model).open("rb") as handle:
        warmstart_model = pickle.load(handle)

    partition_lookup = build_partition_role_lookup(partition_rows)
    dev_rows = partition_rows_for_role(
        candidate_rows,
        partition_lookup,
        partition_role="critic_dev",
    )
    dev_examples = [
        _candidate_example_from_row(row, partition_role="critic_dev")
        for row in dev_rows
    ]
    baseline_metrics = evaluate_warmstart_text_critic(warmstart_model, dev_examples)

    adapted_model, result = train_online_text_critic_adaptation(
        candidate_rows,
        partition_rows,
        online_rows,
        offline_fraction=args.offline_fraction,
        max_train_examples=args.max_train_examples,
        random_seed=args.random_seed,
        commit_positive_weight=args.commit_positive_weight,
    )

    metadata = dict(result.metadata)
    metadata["baseline_metrics"] = baseline_metrics
    metadata["warmstart_model_path"] = str(Path(args.warmstart_model).resolve())
    metadata["candidate_dataset_path"] = str(Path(args.candidate_dataset_dir).resolve())
    metadata["partition_manifest_path"] = str(Path(args.partition_manifest).resolve())
    metadata["online_buffer_path"] = str(Path(args.online_buffer).resolve())
    metadata["runner_framing"] = (
        "This artifact is a tiny mixed-buffer online-adaptation checkpoint for the lightweight "
        "text critic. It remains critic_train-only for updates and critic_dev-only for evaluation."
    )

    adaptation_config = {
        "offline_fraction": float(args.offline_fraction),
        "max_train_examples": args.max_train_examples,
        "commit_positive_weight": float(args.commit_positive_weight),
        "random_seed": int(args.random_seed),
    }
    online_buffer_audit = {
        "row_count": len(online_rows),
        "group_count": len({str(row.get("group_id", "")).strip() for row in online_rows}),
        "partition_roles": sorted(
            {str(row.get("partition_role", "")).strip() for row in online_rows if str(row.get("partition_role", "")).strip()}
        ),
        "source_counts": {
            source: sum(1 for row in online_rows if str(row.get("source", "")).strip() == source)
            for source in sorted(
                {str(row.get("source", "")).strip() for row in online_rows if str(row.get("source", "")).strip()}
            )
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (Path(args.output_dir) / "model.pkl").open("wb") as handle:
        pickle.dump(adapted_model, handle)
    write_text_file(Path(args.output_dir) / "metrics.json", json.dumps(result.metrics, indent=2))
    write_text_file(Path(args.output_dir) / "metadata.json", json.dumps(metadata, indent=2))
    write_text_file(
        Path(args.output_dir) / "adaptation_config.json",
        json.dumps(adaptation_config, indent=2),
    )
    write_text_file(
        Path(args.output_dir) / "online_buffer_audit.json",
        json.dumps(online_buffer_audit, indent=2),
    )
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
