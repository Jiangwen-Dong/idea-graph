from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.relation_graph_critic_data import RelationGraphCandidateExample, collate_relation_graph_examples
from idea_graph.relation_graph_two_head_data import (
    RelationGraphCommitExample,
    collate_relation_graph_commit_examples,
)
from idea_graph.relation_graph_two_head_model import RelationGraphTwoHeadCritic


class RelationGraphTwoHeadModelTests(unittest.TestCase):
    def test_two_head_model_returns_edit_scores_and_commit_logits(self) -> None:
        edit_example = RelationGraphCandidateExample(
            state_id="edit-state",
            candidate_id="edit-state::candidate:0000",
            group_id="group-a",
            split="train",
            label=1,
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
        commit_example = RelationGraphCommitExample(
            state_id="commit-state",
            group_id="group-a",
            split="train",
            label=1,
            state_text_embedding=np.ones((8,), dtype=np.float32),
            node_text_embeddings=np.ones((3, 8), dtype=np.float32),
            node_type_ids=[0, 1, 2],
            node_role_ids=[0, 0, 1],
            node_confidence=[0.8, 0.7, 0.6],
            node_evidence_count=[1.0, 0.0, 0.0],
            edge_index=[(0, 1), (1, 2)],
            edge_type_ids=[0, 1],
            edge_resolved=[1.0, 0.0],
            support_coverage=1.0,
            unresolved_contradiction_ratio=0.0,
            utility=7.5,
        )
        edit_batch = collate_relation_graph_examples([edit_example])
        commit_batch = collate_relation_graph_commit_examples([commit_example])

        model = RelationGraphTwoHeadCritic(
            text_dim=8,
            hidden_dim=16,
            node_type_count=4,
            role_count=4,
            edge_type_count=4,
            candidate_kind_count=4,
        )
        edit_scores = model.score_edit_batch(edit_batch)
        commit_logits = model.score_commit_batch(commit_batch)

        self.assertEqual(tuple(edit_scores.shape), (1,))
        self.assertEqual(tuple(commit_logits.shape), (1,))


if __name__ == "__main__":
    unittest.main()
