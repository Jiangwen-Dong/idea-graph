from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for candidate in (str(SRC), str(TESTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from idea_graph.relation_graph_critic_data import (  # type: ignore[attr-defined]
    HashTextEmbeddingBackend,
    build_relation_graph_dataset,
    collate_relation_graph_examples,
)
from idea_graph.relation_graph_critic_model import RelationGraphCritic  # type: ignore[import-not-found]
from test_relation_graph_critic_data import write_relation_graph_fixture


class RelationGraphCriticModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(self._testMethodName)
        self.fixture = write_relation_graph_fixture(self.tmp_dir)
        self.dataset = build_relation_graph_dataset(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
            text_backend=HashTextEmbeddingBackend(dim=8),
        )

    def tearDown(self) -> None:
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if self.tmp_dir.exists():
            self.tmp_dir.rmdir()

    def test_relation_graph_critic_forward_returns_one_score_per_candidate(self) -> None:
        batch = collate_relation_graph_examples(self.dataset.train_examples)
        model = RelationGraphCritic(
            text_dim=8,
            hidden_dim=16,
            node_type_count=batch.node_type_vocab_size,
            role_count=batch.role_vocab_size,
            edge_type_count=batch.edge_type_vocab_size,
            candidate_kind_count=batch.candidate_kind_vocab_size,
        )

        scores = model(batch)

        self.assertEqual(tuple(scores.shape), (2,))

    def test_relation_graph_critic_changes_score_when_target_indices_change(self) -> None:
        batch = collate_relation_graph_examples(self.dataset.train_examples)
        model = RelationGraphCritic(
            text_dim=8,
            hidden_dim=16,
            node_type_count=batch.node_type_vocab_size,
            role_count=batch.role_vocab_size,
            edge_type_count=batch.edge_type_vocab_size,
            candidate_kind_count=batch.candidate_kind_vocab_size,
        )

        left = model(batch).detach().clone()
        shifted_target_mask = batch.target_mask.roll(shifts=1, dims=1)
        shifted_neighbor_mask = batch.neighbor_mask.roll(shifts=1, dims=1)
        updated_batch = batch.with_updates(
            target_mask=shifted_target_mask,
            neighbor_mask=shifted_neighbor_mask,
        )
        right = model(updated_batch).detach().clone()

        self.assertFalse(torch.allclose(left, right))

    def test_relation_graph_batch_to_moves_all_tensor_fields(self) -> None:
        batch = collate_relation_graph_examples(self.dataset.train_examples)

        moved = batch.to(torch.device("cpu"))

        self.assertEqual(moved.node_text_embeddings.device.type, "cpu")
        self.assertEqual(moved.edge_index.device.type, "cpu")
        self.assertEqual(moved.target_mask.device.type, "cpu")
        self.assertEqual(moved.graph_mask.device.type, "cpu")
