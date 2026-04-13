from __future__ import annotations

import json
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

from idea_graph.fs_utils import _windows_safe_path, write_text_file
from idea_graph.graph_feature_critic import (
    build_graph_feature_examples,
    train_graph_feature_critic,
)


class GraphFeatureCriticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.candidate_dir = self.tmp_dir / "g25"
        self.g1_dir = self.tmp_dir / "g1"
        self.partition_path = self.tmp_dir / "partition_manifest.jsonl"
        self.output_dir = self.tmp_dir / "graph_feature_model"

        train_run = "C:/tmp/run-train"
        dev_run = "C:/tmp/run-dev"

        candidate_rows = [
            {
                "state_id": f"{train_run}::step:0000::Round1::MechanismProposer",
                "candidate_id": f"{train_run}::step:0000::Round1::MechanismProposer::candidate:0000",
                "group_id": "AI_Idea_Bench_2025::train-case",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "train-case",
                "run_dir": train_run,
                "step_index": 0,
                "round_name": "Round1",
                "role": "MechanismProposer",
                "state_kind": "pre_action",
                "candidate_count": 2,
                "candidate_kind": "add_support_edge",
                "candidate_target_ids": ["N001", "N002"],
                "candidate_payload": {"branch_id": "B001"},
                "candidate_source": "enumerated",
                "candidate_text": "kind=add_support_edge|targets=N001,N002",
                "is_commit": False,
                "is_logged_selected": True,
                "is_commit_positive_state": False,
            },
            {
                "state_id": f"{train_run}::step:0000::Round1::MechanismProposer",
                "candidate_id": f"{train_run}::step:0000::Round1::MechanismProposer::candidate:0001",
                "group_id": "AI_Idea_Bench_2025::train-case",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "train-case",
                "run_dir": train_run,
                "step_index": 0,
                "round_name": "Round1",
                "role": "MechanismProposer",
                "state_kind": "pre_action",
                "candidate_count": 2,
                "candidate_kind": "add_dependency_edge",
                "candidate_target_ids": ["N001", "N003"],
                "candidate_payload": {"branch_id": "B001"},
                "candidate_source": "enumerated",
                "candidate_text": "kind=add_dependency_edge|targets=N001,N003",
                "is_commit": False,
                "is_logged_selected": False,
                "is_commit_positive_state": False,
            },
            {
                "state_id": f"{train_run}::step:0001::Terminal::CommitController",
                "candidate_id": f"{train_run}::step:0001::Terminal::CommitController::candidate:0000",
                "group_id": "AI_Idea_Bench_2025::train-case",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "train-case",
                "run_dir": train_run,
                "step_index": 1,
                "round_name": "Terminal",
                "role": "CommitController",
                "state_kind": "terminal_commit",
                "candidate_count": 2,
                "candidate_kind": "commit",
                "candidate_target_ids": [],
                "candidate_payload": {"branch_id": "B001"},
                "candidate_source": "enumerated",
                "candidate_text": "kind=commit|targets=<none>",
                "is_commit": True,
                "is_logged_selected": True,
                "is_commit_positive_state": True,
            },
            {
                "state_id": f"{train_run}::step:0001::Terminal::CommitController",
                "candidate_id": f"{train_run}::step:0001::Terminal::CommitController::candidate:0001",
                "group_id": "AI_Idea_Bench_2025::train-case",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "train-case",
                "run_dir": train_run,
                "step_index": 1,
                "round_name": "Terminal",
                "role": "CommitController",
                "state_kind": "terminal_commit",
                "candidate_count": 2,
                "candidate_kind": "attach_evidence",
                "candidate_target_ids": ["N001"],
                "candidate_payload": {"branch_id": "B001"},
                "candidate_source": "enumerated",
                "candidate_text": "kind=attach_evidence|targets=N001",
                "is_commit": False,
                "is_logged_selected": False,
                "is_commit_positive_state": False,
            },
            {
                "state_id": f"{dev_run}::step:0000::Round1::MechanismProposer",
                "candidate_id": f"{dev_run}::step:0000::Round1::MechanismProposer::candidate:0000",
                "group_id": "AI_Idea_Bench_2025::dev-case",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "dev-case",
                "run_dir": dev_run,
                "step_index": 0,
                "round_name": "Round1",
                "role": "MechanismProposer",
                "state_kind": "pre_action",
                "candidate_count": 2,
                "candidate_kind": "add_support_edge",
                "candidate_target_ids": ["N001", "N002"],
                "candidate_payload": {"branch_id": "B001"},
                "candidate_source": "enumerated",
                "candidate_text": "kind=add_support_edge|targets=N001,N002",
                "is_commit": False,
                "is_logged_selected": True,
                "is_commit_positive_state": False,
            },
            {
                "state_id": f"{dev_run}::step:0000::Round1::MechanismProposer",
                "candidate_id": f"{dev_run}::step:0000::Round1::MechanismProposer::candidate:0001",
                "group_id": "AI_Idea_Bench_2025::dev-case",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "dev-case",
                "run_dir": dev_run,
                "step_index": 0,
                "round_name": "Round1",
                "role": "MechanismProposer",
                "state_kind": "pre_action",
                "candidate_count": 2,
                "candidate_kind": "add_dependency_edge",
                "candidate_target_ids": ["N001", "N003"],
                "candidate_payload": {"branch_id": "B001"},
                "candidate_source": "enumerated",
                "candidate_text": "kind=add_dependency_edge|targets=N001,N003",
                "is_commit": False,
                "is_logged_selected": False,
                "is_commit_positive_state": False,
            },
            {
                "state_id": f"{dev_run}::step:0001::Terminal::CommitController",
                "candidate_id": f"{dev_run}::step:0001::Terminal::CommitController::candidate:0000",
                "group_id": "AI_Idea_Bench_2025::dev-case",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "dev-case",
                "run_dir": dev_run,
                "step_index": 1,
                "round_name": "Terminal",
                "role": "CommitController",
                "state_kind": "terminal_commit",
                "candidate_count": 2,
                "candidate_kind": "commit",
                "candidate_target_ids": [],
                "candidate_payload": {"branch_id": "B001"},
                "candidate_source": "enumerated",
                "candidate_text": "kind=commit|targets=<none>",
                "is_commit": True,
                "is_logged_selected": True,
                "is_commit_positive_state": True,
            },
            {
                "state_id": f"{dev_run}::step:0001::Terminal::CommitController",
                "candidate_id": f"{dev_run}::step:0001::Terminal::CommitController::candidate:0001",
                "group_id": "AI_Idea_Bench_2025::dev-case",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "dev-case",
                "run_dir": dev_run,
                "step_index": 1,
                "round_name": "Terminal",
                "role": "CommitController",
                "state_kind": "terminal_commit",
                "candidate_count": 2,
                "candidate_kind": "attach_evidence",
                "candidate_target_ids": ["N001"],
                "candidate_payload": {"branch_id": "B001"},
                "candidate_source": "enumerated",
                "candidate_text": "kind=attach_evidence|targets=N001",
                "is_commit": False,
                "is_logged_selected": False,
                "is_commit_positive_state": False,
            },
        ]

        trajectory_rows = [
            {
                "run_dir": train_run,
                "step_index": 0,
                "round_name": "Round1",
                "role": "MechanismProposer",
                "before_state_snapshot": "state_snapshots/train-step-000.json",
            },
            {
                "run_dir": dev_run,
                "step_index": 0,
                "round_name": "Round1",
                "role": "MechanismProposer",
                "before_state_snapshot": "state_snapshots/dev-step-000.json",
            },
        ]
        terminal_rows = [
            {
                "run_dir": train_run,
                "step_index": 1,
                "round_name": "Terminal",
                "role": "CommitController",
                "before_state_snapshot": "terminal_state_snapshots/train-terminal.json",
            },
            {
                "run_dir": dev_run,
                "step_index": 1,
                "round_name": "Terminal",
                "role": "CommitController",
                "before_state_snapshot": "terminal_state_snapshots/dev-terminal.json",
            },
        ]
        partition_rows = [
            {
                "group_id": "AI_Idea_Bench_2025::train-case",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "train-case",
                "source_split": "train",
                "partition_role": "critic_train",
            },
            {
                "group_id": "AI_Idea_Bench_2025::dev-case",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "dev-case",
                "source_split": "validation",
                "partition_role": "critic_dev",
            },
        ]

        train_snapshot = {
            "node_count": 3,
            "edge_count": 2,
            "contradiction_count": 0,
            "support_edge_count": 1,
            "nodes": {
                "N001": {"type": "Hypothesis", "role": "MechanismProposer", "confidence": 0.8, "evidence": ["e1"]},
                "N002": {"type": "Method", "role": "MechanismProposer", "confidence": 0.7, "evidence": []},
                "N003": {"type": "EvalPlan", "role": "EvaluationDesigner", "confidence": 0.6, "evidence": []},
            },
            "edges": [
                {"relation": "supports", "resolved": False},
                {"relation": "depends_on", "resolved": False},
            ],
        }
        train_terminal_snapshot = {
            "node_count": 4,
            "edge_count": 3,
            "contradiction_count": 0,
            "support_edge_count": 2,
            "nodes": {
                "N001": {"type": "Hypothesis", "role": "MechanismProposer", "confidence": 0.9, "evidence": ["e1", "e2"]},
                "N002": {"type": "Method", "role": "MechanismProposer", "confidence": 0.8, "evidence": []},
                "N003": {"type": "EvalPlan", "role": "EvaluationDesigner", "confidence": 0.7, "evidence": []},
                "N004": {"type": "Result", "role": "ImpactReframer", "confidence": 0.6, "evidence": []},
            },
            "edges": [
                {"relation": "supports", "resolved": False},
                {"relation": "supports", "resolved": False},
                {"relation": "depends_on", "resolved": False},
            ],
        }
        dev_snapshot = {
            "node_count": 3,
            "edge_count": 2,
            "contradiction_count": 1,
            "support_edge_count": 0,
            "nodes": {
                "N001": {"type": "Hypothesis", "role": "MechanismProposer", "confidence": 0.75, "evidence": ["e1"]},
                "N002": {"type": "Method", "role": "MechanismProposer", "confidence": 0.65, "evidence": []},
                "N003": {"type": "Risk", "role": "FeasibilityCritic", "confidence": 0.55, "evidence": []},
            },
            "edges": [
                {"relation": "contradicts", "resolved": False},
                {"relation": "depends_on", "resolved": False},
            ],
        }
        dev_terminal_snapshot = {
            "node_count": 4,
            "edge_count": 3,
            "contradiction_count": 0,
            "support_edge_count": 1,
            "nodes": {
                "N001": {"type": "Hypothesis", "role": "MechanismProposer", "confidence": 0.85, "evidence": ["e1", "e2"]},
                "N002": {"type": "Method", "role": "MechanismProposer", "confidence": 0.8, "evidence": []},
                "N003": {"type": "Risk", "role": "FeasibilityCritic", "confidence": 0.5, "evidence": []},
                "N004": {"type": "Result", "role": "ImpactReframer", "confidence": 0.7, "evidence": []},
            },
            "edges": [
                {"relation": "supports", "resolved": False},
                {"relation": "depends_on", "resolved": False},
                {"relation": "contradicts", "resolved": True},
            ],
        }

        self._write_jsonl(self.candidate_dir / "candidate_dataset.jsonl", candidate_rows)
        self._write_jsonl(self.g1_dir / "trajectory_examples.jsonl", trajectory_rows)
        self._write_jsonl(self.g1_dir / "terminal_state_manifest.jsonl", terminal_rows)
        self._write_jsonl(self.partition_path, partition_rows)
        write_text_file(self.g1_dir / "state_snapshots/train-step-000.json", json.dumps(train_snapshot))
        write_text_file(self.g1_dir / "state_snapshots/dev-step-000.json", json.dumps(dev_snapshot))
        write_text_file(
            self.g1_dir / "terminal_state_snapshots/train-terminal.json",
            json.dumps(train_terminal_snapshot),
        )
        write_text_file(
            self.g1_dir / "terminal_state_snapshots/dev-terminal.json",
            json.dumps(dev_terminal_snapshot),
        )

    def tearDown(self) -> None:
        shutil.rmtree(_windows_safe_path(self.tmp_dir), ignore_errors=True)

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        write_text_file(path, text)

    def test_build_graph_feature_examples_preserves_partition_safety_and_features(self) -> None:
        dataset = build_graph_feature_examples(
            candidate_dataset_dir=self.candidate_dir,
            g1_dataset_dir=self.g1_dir,
            partition_manifest_path=self.partition_path,
        )
        self.assertEqual(dataset.split_counts["critic_train"], 4)
        self.assertEqual(dataset.split_counts["critic_dev"], 4)
        self.assertEqual(dataset.group_overlap_count, 0)
        example = dataset.examples[0]
        self.assertIn("state_node_count", example.feature_dict)
        self.assertIn("candidate_kind=add_support_edge", example.feature_dict)
        self.assertIn("target_node_type__Hypothesis", example.feature_dict)

    def test_train_graph_feature_critic_reports_dev_metrics(self) -> None:
        dataset = build_graph_feature_examples(
            candidate_dataset_dir=self.candidate_dir,
            g1_dataset_dir=self.g1_dir,
            partition_manifest_path=self.partition_path,
        )
        result = train_graph_feature_critic(
            dataset=dataset,
            output_dir=self.output_dir,
            commit_positive_weight=2.0,
        )
        self.assertGreater(result.metrics["state_count"], 0)
        self.assertIn("top1_accuracy", result.metrics)
        self.assertIn("mean_reciprocal_rank", result.metrics)
        self.assertTrue(result.model_path.exists())
        self.assertTrue(result.metadata_path.exists())

    def test_cli_train_graph_feature_critic_writes_metrics(self) -> None:
        script_path = ROOT / "scripts" / "train_graph_feature_critic.py"
        command = [
            sys.executable,
            str(script_path),
            "--candidate-dataset-dir",
            str(self.candidate_dir),
            "--g1-dataset-dir",
            str(self.g1_dir),
            "--partition-manifest",
            str(self.partition_path),
            "--output-dir",
            str(self.output_dir),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertTrue((self.output_dir / "metrics.json").exists())
        self.assertTrue((self.output_dir / "metadata.json").exists())
        self.assertTrue((self.output_dir / "model.pkl").exists())


if __name__ == "__main__":
    unittest.main()
