from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.online_text_critic import (
    build_namespace_support,
    build_warmstart_training_bundle,
    build_partition_role_lookup,
    train_warmstart_text_critic,
    validate_required_namespace_support,
)


class OnlineTextCriticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.partition_rows = [
            {
                "group_id": "g-train",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "train-case",
                "source_split": "train",
                "partition_role": "critic_train",
            },
            {
                "group_id": "g-dev",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "dev-case",
                "source_split": "validation",
                "partition_role": "critic_dev",
            },
            {
                "group_id": "g-paper",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "paper-case",
                "source_split": "validation",
                "partition_role": "paper_eval",
            },
        ]
        self.candidate_rows = [
            {
                "state_id": "train-edit",
                "candidate_id": "train-edit::0",
                "group_id": "g-train",
                "state_text": "train state best edit",
                "candidate_text": "best action",
                "is_logged_selected": True,
                "is_commit": False,
                "is_commit_positive_state": False,
                "targets": {"weak_value_01": 0.7, "native_value_01": 0.8},
            },
            {
                "state_id": "train-edit",
                "candidate_id": "train-edit::1",
                "group_id": "g-train",
                "state_text": "train state best edit",
                "candidate_text": "weak action",
                "is_logged_selected": False,
                "is_commit": True,
                "is_commit_positive_state": False,
                "targets": {"weak_value_01": 0.7, "native_value_01": 0.8},
            },
            {
                "state_id": "train-commit",
                "candidate_id": "train-commit::0",
                "group_id": "g-train",
                "state_text": "train terminal state",
                "candidate_text": "commit winner",
                "is_logged_selected": True,
                "is_commit": True,
                "is_commit_positive_state": True,
                "targets": {"weak_value_01": 0.75, "native_value_01": 0.82},
            },
            {
                "state_id": "train-commit",
                "candidate_id": "train-commit::1",
                "group_id": "g-train",
                "state_text": "train terminal state",
                "candidate_text": "other action",
                "is_logged_selected": False,
                "is_commit": False,
                "is_commit_positive_state": False,
                "targets": {"weak_value_01": 0.75, "native_value_01": 0.82},
            },
            {
                "state_id": "dev-edit",
                "candidate_id": "dev-edit::0",
                "group_id": "g-dev",
                "state_text": "dev state best edit",
                "candidate_text": "best action",
                "is_logged_selected": True,
                "is_commit": False,
                "is_commit_positive_state": False,
                "targets": {"weak_value_01": 0.65, "native_value_01": 0.77},
            },
            {
                "state_id": "dev-edit",
                "candidate_id": "dev-edit::1",
                "group_id": "g-dev",
                "state_text": "dev state best edit",
                "candidate_text": "weak action",
                "is_logged_selected": False,
                "is_commit": True,
                "is_commit_positive_state": False,
                "targets": {"weak_value_01": 0.65, "native_value_01": 0.77},
            },
            {
                "state_id": "dev-commit",
                "candidate_id": "dev-commit::0",
                "group_id": "g-dev",
                "state_text": "dev terminal state",
                "candidate_text": "commit winner",
                "is_logged_selected": True,
                "is_commit": True,
                "is_commit_positive_state": True,
                "targets": {"weak_value_01": 0.67, "native_value_01": 0.79},
            },
            {
                "state_id": "dev-commit",
                "candidate_id": "dev-commit::1",
                "group_id": "g-dev",
                "state_text": "dev terminal state",
                "candidate_text": "other action",
                "is_logged_selected": False,
                "is_commit": False,
                "is_commit_positive_state": False,
                "targets": {"weak_value_01": 0.67, "native_value_01": 0.79},
            },
            {
                "state_id": "paper-eval",
                "candidate_id": "paper-eval::0",
                "group_id": "g-paper",
                "state_text": "paper eval state",
                "candidate_text": "best action",
                "is_logged_selected": True,
                "is_commit": False,
                "is_commit_positive_state": False,
                "targets": {"weak_value_01": 0.9, "native_value_01": 0.9},
            },
            {
                "state_id": "paper-eval",
                "candidate_id": "paper-eval::1",
                "group_id": "g-paper",
                "state_text": "paper eval state",
                "candidate_text": "commit action",
                "is_logged_selected": False,
                "is_commit": True,
                "is_commit_positive_state": False,
                "targets": {"weak_value_01": 0.9, "native_value_01": 0.9},
            },
        ]

    def test_build_partition_role_lookup_rejects_duplicates(self) -> None:
        duplicate_rows = [self.partition_rows[0], dict(self.partition_rows[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate partition assignment"):
            build_partition_role_lookup(duplicate_rows)

    def test_build_warmstart_training_bundle_uses_partition_roles_and_ignores_paper_eval(self) -> None:
        bundle = build_warmstart_training_bundle(self.candidate_rows, self.partition_rows)
        self.assertEqual(len(bundle.train_examples), 4)
        self.assertEqual(len(bundle.dev_examples), 4)
        self.assertEqual(bundle.split_audit["train_group_count"], 1)
        self.assertEqual(bundle.split_audit["validation_group_count"], 1)
        self.assertEqual(bundle.namespace_support["critic_train"]["terminal_commit"]["positive_count"], 1)
        self.assertEqual(bundle.namespace_support["critic_dev"]["terminal_commit"]["positive_count"], 1)

    def test_validate_required_namespace_support_raises_when_dev_commit_is_missing(self) -> None:
        rows = [dict(row) for row in self.candidate_rows]
        for row in rows:
            if row["group_id"] == "g-dev" and row["is_commit"]:
                row["is_commit_positive_state"] = False
                row["is_logged_selected"] = False
        partition_lookup = build_partition_role_lookup(self.partition_rows)
        namespace_support = build_namespace_support(rows, partition_lookup)
        with self.assertRaisesRegex(ValueError, "terminal_commit"):
            validate_required_namespace_support(
                namespace_support,
                partition_role="critic_dev",
                required_namespaces=("teacher_logged", "terminal_commit"),
            )

    def test_train_warmstart_text_critic_reports_metrics_and_support(self) -> None:
        _, bundle, metrics = train_warmstart_text_critic(self.candidate_rows, self.partition_rows)
        self.assertEqual(metrics["train_example_count"], 4)
        self.assertEqual(metrics["validation_example_count"], 4)
        self.assertIn("top1_accuracy", metrics)
        self.assertIn("mean_reciprocal_rank", metrics)
        self.assertEqual(bundle.namespace_support["critic_train"]["teacher_logged"]["positive_count"], 2)
        self.assertEqual(bundle.namespace_support["critic_dev"]["teacher_logged"]["positive_count"], 2)


if __name__ == "__main__":
    unittest.main()
