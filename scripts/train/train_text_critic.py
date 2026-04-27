from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import read_text_file, write_text_file
from idea_graph.text_critic import (
    build_split_audit,
    build_training_examples,
    evaluate_state_rankings,
    train_text_critic,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, raw_line in enumerate(read_text_file(path).splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_index} must contain a JSON object.")
        rows.append(dict(payload))
    return rows


def _commit_audit(rows: list[dict[str, Any]], split: str) -> dict[str, int]:
    split_rows = [row for row in rows if str(row.get("split", "train")).strip() == split]
    commit_rows = [row for row in split_rows if bool(row.get("is_commit", False))]
    return {
        "candidate_count": len(commit_rows),
        "positive_count": sum(1 for row in commit_rows if bool(row.get("is_logged_selected", False))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a text-only next-action critic.")
    parser.add_argument(
        "--candidate-dataset-dir",
        type=Path,
        required=True,
        help="Directory containing candidate_dataset.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where model and metrics will be written.",
    )
    args = parser.parse_args()

    candidate_path = Path(args.candidate_dataset_dir) / "candidate_dataset.jsonl"
    rows = _load_jsonl(candidate_path)
    examples = build_training_examples(rows)
    train_examples = [example for example in examples if example.split == "train"]
    validation_examples = [example for example in examples if example.split == "validation"]
    if not validation_examples:
        raise ValueError("Validation split is empty; refusing to train/evaluate pilot critic.")

    split_audit = build_split_audit(train_examples, validation_examples)
    if split_audit["group_overlap_count"] != 0:
        raise ValueError(
            f"Train/validation group overlap detected: {split_audit['group_overlap_count']} overlapping groups."
        )

    model = train_text_critic(train_examples)
    metrics = evaluate_state_rankings(model, validation_examples)
    metrics["train_example_count"] = len(train_examples)
    metrics["validation_example_count"] = len(validation_examples)

    train_commit = _commit_audit(rows, "train")
    validation_commit = _commit_audit(rows, "validation")
    metadata = {
        "candidate_dataset_path": str(Path(args.candidate_dataset_dir).resolve()),
        "train_example_count": len(train_examples),
        "validation_example_count": len(validation_examples),
        "train_group_count": split_audit["train_group_count"],
        "validation_group_count": split_audit["validation_group_count"],
        "group_overlap_count": split_audit["group_overlap_count"],
        "train_commit_candidate_count": train_commit["candidate_count"],
        "train_commit_positive_count": train_commit["positive_count"],
        "validation_commit_candidate_count": validation_commit["candidate_count"],
        "validation_commit_positive_count": validation_commit["positive_count"],
        "feature_policy": "Model-facing text strips candidate segments with source= and rationale=.",
        "pilot_framing": (
            "This pilot uses logged-edit imitation supervision over candidate slates; "
            "it is not a full commit-capable controller."
        ),
        "commit_label_note": (
            "Commit candidates have zero positive labels in both train and validation."
            if train_commit["positive_count"] == 0 and validation_commit["positive_count"] == 0
            else "Commit candidates include positive labels in at least one split."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.output_dir) / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    write_text_file(Path(args.output_dir) / "metrics.json", json.dumps(metrics, indent=2))
    write_text_file(Path(args.output_dir) / "metadata.json", json.dumps(metadata, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
