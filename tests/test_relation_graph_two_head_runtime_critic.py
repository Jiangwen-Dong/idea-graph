from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.engine import build_seed_graphs, maturity_snapshot, merge_seed_graphs
from idea_graph.models import IdeaGraph
from idea_graph.relation_graph_two_head_model import RelationGraphTwoHeadCritic
from idea_graph.relation_graph_two_head_runtime_critic import load_relation_graph_two_head_runtime_bundle


class RelationGraphTwoHeadRuntimeCriticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_loaded_two_head_runtime_bundle_scores_edit_candidates_and_commit_graph(self) -> None:
        model_dir = self.tmp_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        vocabularies = {
            "node_type_to_id": {"unknown": 0},
            "role_to_id": {"unknown": 0},
            "edge_type_to_id": {"unknown": 0},
            "candidate_kind_to_id": {"skip": 0, "freeze_branch": 1, "unknown": 2},
        }
        (model_dir / "vocabularies.json").write_text(json.dumps(vocabularies), encoding="utf-8")
        (model_dir / "training_config.json").write_text(
            json.dumps(
                {
                    "text_backend": "hash",
                    "embedding_dim": 8,
                    "hidden_dim": 16,
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "hidden_dim": 16,
                    "text_backend": "hash",
                }
            ),
            encoding="utf-8",
        )
        model = RelationGraphTwoHeadCritic(
            text_dim=8,
            hidden_dim=16,
            node_type_count=1,
            role_count=1,
            edge_type_count=1,
            candidate_kind_count=3,
        )
        torch.save(model.state_dict(), model_dir / "model.pt")

        graph = IdeaGraph(topic="topic", literature=["paper a"])
        build_seed_graphs(graph)
        merge_seed_graphs(graph)
        branch_id = next(branch.id for branch in graph.branches.values())

        bundle = load_relation_graph_two_head_runtime_bundle(model_dir)
        runtime_batch = bundle.build_runtime_batch(
            graph=graph,
            candidate_specs=[
                {
                    "candidate_id": "skip",
                    "kind": "skip",
                    "target_ids": [],
                    "payload": {"branch_id": branch_id},
                },
                {
                    "candidate_id": "freeze",
                    "kind": "freeze_branch",
                    "target_ids": [],
                    "payload": {"branch_id": branch_id},
                },
            ],
            use_commit=False,
        )

        edit_scores = bundle.score_runtime_batch(runtime_batch.batch)
        commit_probability = bundle.score_commit_graph(graph, snapshot=maturity_snapshot(graph))

        self.assertEqual(len(edit_scores), 2)
        self.assertTrue(all(isinstance(score, float) for score in edit_scores))
        self.assertGreaterEqual(commit_probability, 0.0)
        self.assertLessEqual(commit_probability, 1.0)


if __name__ == "__main__":
    unittest.main()
