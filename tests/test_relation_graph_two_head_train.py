from __future__ import annotations

import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.relation_graph_critic_data import RelationGraphCandidateExample
from idea_graph.relation_graph_two_head_data import (
    RelationGraphCommitExample,
    RelationGraphTwoHeadDataset,
)
from idea_graph.relation_graph_two_head_train import train_relation_graph_two_head_critic


def _edit_example(*, state_id: str, candidate_id: str, label: int, split: str) -> RelationGraphCandidateExample:
    return RelationGraphCandidateExample(
        state_id=state_id,
        candidate_id=candidate_id,
        group_id=f"group::{split}",
        split=split,
        label=label,
        is_commit=False,
        candidate_kind_id=0,
        candidate_text_embedding=np.ones((8,), dtype=np.float32),
        state_text_embedding=np.ones((8,), dtype=np.float32),
        node_text_embeddings=np.ones((3, 8), dtype=np.float32),
        node_type_ids=[0, 1, 2],
        node_role_ids=[0, 0, 1],
        node_confidence=[0.8, 0.7, 0.6],
        node_evidence_count=[1.0, 0.0, 0.0],
        edge_index=[(0, 1), (1, 2)],
        edge_type_ids=[0, 1],
        edge_resolved=[1.0, 0.0],
        target_node_indices=[0, 1],
    )


def _commit_example(*, state_id: str, label: int, split: str) -> RelationGraphCommitExample:
    return RelationGraphCommitExample(
        state_id=state_id,
        group_id=f"group::{split}",
        split=split,
        label=label,
        state_text_embedding=np.ones((8,), dtype=np.float32),
        node_text_embeddings=np.ones((3, 8), dtype=np.float32),
        node_type_ids=[0, 1, 2],
        node_role_ids=[0, 0, 1],
        node_confidence=[0.8, 0.7, 0.6],
        node_evidence_count=[1.0, 0.0, 0.0],
        edge_index=[(0, 1), (1, 2)],
        edge_type_ids=[0, 1],
        edge_resolved=[1.0, 0.0],
        support_coverage=1.0 if label else 0.5,
        unresolved_contradiction_ratio=0.0 if label else 1.0,
        utility=7.5 if label else 4.5,
    )


class RelationGraphTwoHeadTrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_train_relation_graph_two_head_critic_writes_metrics(self) -> None:
        dataset = RelationGraphTwoHeadDataset(
            edit_train_examples=[
                _edit_example(state_id="train-state", candidate_id="c0", label=1, split="train"),
                _edit_example(state_id="train-state", candidate_id="c1", label=0, split="train"),
            ],
            edit_dev_examples=[
                _edit_example(state_id="dev-state", candidate_id="c0", label=1, split="validation"),
                _edit_example(state_id="dev-state", candidate_id="c1", label=0, split="validation"),
            ],
            commit_train_examples=[_commit_example(state_id="commit-train", label=0, split="train")],
            commit_dev_examples=[_commit_example(state_id="commit-dev", label=1, split="validation")],
            metadata={
                "node_type_count": 4,
                "role_count": 4,
                "edge_type_count": 4,
                "candidate_kind_count": 4,
            },
        )

        artifacts = train_relation_graph_two_head_critic(
            dataset=dataset,
            output_dir=self.tmp_dir / "model_output",
            hidden_dim=16,
            batch_size=1,
            epochs=1,
            learning_rate=1e-3,
            text_backend_name="hash",
        )

        self.assertIsNotNone(artifacts.model_path)
        self.assertTrue((self.tmp_dir / "model_output" / "model.pt").exists())
        self.assertTrue((self.tmp_dir / "model_output" / "edit_metrics.json").exists())
        self.assertTrue((self.tmp_dir / "model_output" / "commit_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
