from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.online_text_critic import (
    build_online_adaptation_examples,
    build_namespace_support,
    build_partition_examples,
    build_warmstart_training_bundle,
    build_partition_role_lookup,
    partition_rows_for_role,
    train_online_text_critic_adaptation,
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
        self.online_rows = [
            {
                "state_id": "online-edit",
                "candidate_id": "online-edit::0",
                "group_id": "g-train",
                "partition_role": "critic_train",
                "source": "online",
                "state_text": "online state best edit",
                "candidate_text": "best action",
                "is_logged_selected": True,
                "is_commit": False,
                "is_commit_positive_state": False,
                "targets": {"weak_value_01": 0.72, "native_value_01": 0.83},
            },
            {
                "state_id": "online-edit",
                "candidate_id": "online-edit::1",
                "group_id": "g-train",
                "partition_role": "critic_train",
                "source": "online",
                "state_text": "online state best edit",
                "candidate_text": "weak action",
                "is_logged_selected": False,
                "is_commit": True,
                "is_commit_positive_state": False,
                "targets": {"weak_value_01": 0.72, "native_value_01": 0.83},
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

    def test_partition_rows_for_role_filters_candidate_rows_by_partition(self) -> None:
        partition_lookup = build_partition_role_lookup(self.partition_rows)
        train_rows = partition_rows_for_role(
            self.candidate_rows,
            partition_lookup,
            partition_role="critic_train",
        )
        self.assertEqual(len(train_rows), 4)
        self.assertEqual({row["group_id"] for row in train_rows}, {"g-train"})
        self.assertEqual({row["partition_role"] for row in train_rows}, {"critic_train"})

    def test_build_online_adaptation_examples_rejects_non_train_rows(self) -> None:
        rows = [dict(self.online_rows[0]), dict(self.online_rows[1])]
        rows[0]["partition_role"] = "critic_dev"
        with self.assertRaisesRegex(ValueError, "critic_train"):
            build_online_adaptation_examples(rows)

    def test_train_online_text_critic_adaptation_reports_offline_and_online_counts(self) -> None:
        _, result = train_online_text_critic_adaptation(
            self.candidate_rows,
            self.partition_rows,
            self.online_rows,
            offline_fraction=0.5,
            max_train_examples=4,
            random_seed=0,
        )
        self.assertEqual(result.metrics["validation_example_count"], 4)
        self.assertGreater(result.metadata["offline_example_count"], 0)
        self.assertGreater(result.metadata["online_example_count"], 0)
        self.assertEqual(result.metadata["dev_example_count"], 4)

    def test_run_online_text_critic_adaptation_cli_writes_artifacts(self) -> None:
        tmp_dir = Path(mkdtemp())
        try:
            candidate_dir = tmp_dir / "candidate"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = candidate_dir / "candidate_dataset.jsonl"
            candidate_text = "".join(json.dumps(row) + "\n" for row in self.candidate_rows)
            candidate_path.write_text(candidate_text, encoding="utf-8")

            partition_path = tmp_dir / "partition_manifest.jsonl"
            partition_text = "".join(json.dumps(row) + "\n" for row in self.partition_rows)
            partition_path.write_text(partition_text, encoding="utf-8")

            online_path = tmp_dir / "online_buffer.jsonl"
            online_text = "".join(json.dumps(row) + "\n" for row in self.online_rows)
            online_path.write_text(online_text, encoding="utf-8")

            warmstart_dir = tmp_dir / "warmstart"
            warmstart_dir.mkdir(parents=True, exist_ok=True)
            model, _, _ = train_warmstart_text_critic(self.candidate_rows, self.partition_rows)
            with (warmstart_dir / "model.pkl").open("wb") as handle:
                pickle.dump(model, handle)

            output_dir = tmp_dir / "adapted"
            script_path = ROOT / "scripts" / "run_online_text_critic_adaptation.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--candidate-dataset-dir",
                    str(candidate_dir),
                    "--partition-manifest",
                    str(partition_path),
                    "--online-buffer",
                    str(online_path),
                    "--warmstart-model",
                    str(warmstart_dir / "model.pkl"),
                    "--output-dir",
                    str(output_dir),
                    "--offline-fraction",
                    "0.5",
                    "--max-train-examples",
                    "4",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "metadata.json").exists())
            self.assertTrue((output_dir / "adaptation_config.json").exists())
            self.assertTrue((output_dir / "model.pkl").exists())
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn("baseline_metrics", metadata)
            self.assertGreater(metadata["online_example_count"], 0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
