from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import _windows_safe_path, write_text_file
from idea_graph.relation_graph_critic_data import HashTextEmbeddingBackend
from idea_graph.relation_graph_two_head_data import build_relation_graph_two_head_dataset


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    write_text_file(path, "\n".join(json.dumps(row) for row in rows) + "\n")


class RelationGraphTwoHeadDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.dataset_dir = self.tmp_dir / "parallel_two_head"
        self.g1_dir = self.tmp_dir / "g1"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.g1_dir.mkdir(parents=True, exist_ok=True)
        (self.g1_dir / "parallel_state_snapshots").mkdir(parents=True, exist_ok=True)
        (self.g1_dir / "post_round_commit_state_snapshots").mkdir(parents=True, exist_ok=True)

        snapshot = {
            "node_count": 3,
            "edge_count": 2,
            "contradiction_count": 0,
            "support_edge_count": 1,
            "nodes": {
                "N001": {
                    "id": "N001",
                    "type": "Hypothesis",
                    "text": "shared evidence anchor",
                    "role": "MechanismProposer",
                    "branch_id": "B001",
                    "confidence": 0.8,
                    "evidence": ["shared evidence anchor"],
                    "status": "active",
                },
                "N002": {
                    "id": "N002",
                    "type": "Method",
                    "text": "shared evidence anchor",
                    "role": "MechanismProposer",
                    "branch_id": "B001",
                    "confidence": 0.7,
                    "evidence": [],
                    "status": "active",
                },
                "N003": {
                    "id": "N003",
                    "type": "EvalPlan",
                    "text": "shared evidence anchor",
                    "role": "EvaluationDesigner",
                    "branch_id": "B002",
                    "confidence": 0.6,
                    "evidence": [],
                    "status": "active",
                },
            },
            "edges": [
                {
                    "id": "E001",
                    "source_id": "N001",
                    "target_id": "N002",
                    "relation": "depends_on",
                    "resolved": True,
                },
                {
                    "id": "E002",
                    "source_id": "N002",
                    "target_id": "N003",
                    "relation": "supports",
                    "resolved": False,
                },
            ],
        }
        write_text_file(
            self.g1_dir / "parallel_state_snapshots" / "train-parallel.json",
            json.dumps(snapshot),
        )
        write_text_file(
            self.g1_dir / "parallel_state_snapshots" / "dev-parallel.json",
            json.dumps(snapshot),
        )
        write_text_file(
            self.g1_dir / "post_round_commit_state_snapshots" / "train-post-round.json",
            json.dumps({**snapshot, "action_count": 1}),
        )
        write_text_file(
            self.g1_dir / "post_round_commit_state_snapshots" / "dev-post-round.json",
            json.dumps({**snapshot, "action_count": 1}),
        )

        edit_rows = [
            {
                "state_id": "train::parallel::Round1::MechanismProposer",
                "candidate_id": "train::parallel::Round1::MechanismProposer::candidate:0000",
                "group_id": "AI_Idea_Bench_2025::train-case",
                "split": "train",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "train-case",
                "run_dir": "train-run",
                "step_index": 0,
                "round_name": "Round1",
                "role": "MechanismProposer",
                "state_kind": "parallel_pre_action",
                "state_text": "shared evidence anchor",
                "candidate_index": 0,
                "candidate_count": 2,
                "candidate_kind": "add_support_edge",
                "candidate_target_ids": ["N001", "N002"],
                "candidate_payload": {"branch_id": "B001"},
                "candidate_source": "parallel_selected",
                "candidate_text": "kind=add_support_edge",
                "is_logged_selected": True,
                "runtime_protocol": "parallel_graph_v2",
                "label_source": "parallel_protocol_teacher_v1",
                "before_state_snapshot": "parallel_state_snapshots/train-parallel.json",
            },
            {
                "state_id": "dev::parallel::Round1::MechanismProposer",
                "candidate_id": "dev::parallel::Round1::MechanismProposer::candidate:0000",
                "group_id": "AI_Idea_Bench_2025::dev-case",
                "split": "validation",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "dev-case",
                "run_dir": "dev-run",
                "step_index": 0,
                "round_name": "Round1",
                "role": "MechanismProposer",
                "state_kind": "parallel_pre_action",
                "state_text": "shared evidence anchor",
                "candidate_index": 0,
                "candidate_count": 2,
                "candidate_kind": "add_support_edge",
                "candidate_target_ids": ["N001", "N002"],
                "candidate_payload": {"branch_id": "B001"},
                "candidate_source": "parallel_selected",
                "candidate_text": "kind=add_support_edge",
                "is_logged_selected": True,
                "runtime_protocol": "parallel_graph_v2",
                "label_source": "parallel_protocol_teacher_v1",
                "before_state_snapshot": "parallel_state_snapshots/dev-parallel.json",
            },
        ]
        commit_rows = [
            {
                "state_id": "train::parallel::Round1::post_round_commit",
                "group_id": "AI_Idea_Bench_2025::train-case",
                "split": "train",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "train-case",
                "run_dir": "train-run",
                "step_index": 0,
                "round_name": "Round1",
                "role": "CommitController",
                "state_kind": "parallel_post_round",
                "state_text": "shared evidence anchor",
                "runtime_protocol": "parallel_graph_v2",
                "label_source": "maturity_snapshot",
                "before_state_snapshot": "post_round_commit_state_snapshots/train-post-round.json",
                "commit_supervision": {"available": True, "label": 0, "source": "maturity_snapshot"},
            },
            {
                "state_id": "dev::parallel::Round1::post_round_commit",
                "group_id": "AI_Idea_Bench_2025::dev-case",
                "split": "validation",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "dev-case",
                "run_dir": "dev-run",
                "step_index": 0,
                "round_name": "Round1",
                "role": "CommitController",
                "state_kind": "parallel_post_round",
                "state_text": "shared evidence anchor",
                "runtime_protocol": "parallel_graph_v2",
                "label_source": "maturity_snapshot",
                "before_state_snapshot": "post_round_commit_state_snapshots/dev-post-round.json",
                "commit_supervision": {"available": True, "label": 1, "source": "maturity_snapshot"},
            },
        ]
        _write_jsonl(_windows_safe_path(self.dataset_dir / "edit_head_rows.jsonl"), edit_rows)
        _write_jsonl(_windows_safe_path(self.dataset_dir / "commit_head_rows.jsonl"), commit_rows)

    def tearDown(self) -> None:
        shutil.rmtree(_windows_safe_path(self.tmp_dir), ignore_errors=True)

    def test_build_relation_graph_two_head_dataset_returns_train_and_dev_examples(self) -> None:
        dataset = build_relation_graph_two_head_dataset(
            dataset_dir=self.dataset_dir,
            g1_dataset_dir=self.g1_dir,
            text_backend=HashTextEmbeddingBackend(dim=8),
        )

        self.assertEqual(len(dataset.edit_train_examples), 1)
        self.assertEqual(len(dataset.edit_dev_examples), 1)
        self.assertEqual(len(dataset.commit_train_examples), 1)
        self.assertEqual(len(dataset.commit_dev_examples), 1)
        self.assertEqual(dataset.edit_train_examples[0].target_node_indices, [0, 1])
        self.assertEqual(dataset.commit_dev_examples[0].label, 1)
        self.assertEqual(dataset.edit_train_examples[0].node_text_embeddings.shape, (3, 8))
        self.assertEqual(dataset.metadata["candidate_kind_count"], 2)


if __name__ == "__main__":
    unittest.main()
