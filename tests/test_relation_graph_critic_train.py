from __future__ import annotations

from dataclasses import dataclass
import random
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.relation_graph_critic_train import (  # type: ignore[import-not-found]
    _iterate_state_grouped_batches,
    compute_state_ranking_loss,
    evaluate_relation_graph_rankings,
)


class RelationGraphCriticTrainTests(unittest.TestCase):
    def test_iterate_state_grouped_batches_keeps_each_state_together(self) -> None:
        @dataclass(frozen=True)
        class Example:
            state_id: str

        examples = [
            Example("s1"),
            Example("s1"),
            Example("s2"),
            Example("s3"),
            Example("s3"),
        ]

        batches = _iterate_state_grouped_batches(examples, batch_size=2, shuffle=False)
        batch_state_ids = [[example.state_id for example in batch] for batch in batches]

        self.assertEqual(batch_state_ids, [["s1", "s1", "s2"], ["s3", "s3"]])

    def test_iterate_state_grouped_batches_can_shuffle_across_epochs(self) -> None:
        @dataclass(frozen=True)
        class Example:
            state_id: str

        examples = [Example(f"s{index}") for index in range(8)]
        rng = random.Random(0)

        first = _iterate_state_grouped_batches(examples, batch_size=2, shuffle=True, rng=rng)
        second = _iterate_state_grouped_batches(examples, batch_size=2, shuffle=True, rng=rng)

        self.assertNotEqual(
            [[example.state_id for example in batch] for batch in first],
            [[example.state_id for example in batch] for batch in second],
        )

    def test_compute_state_ranking_loss_rewards_positive_candidate(self) -> None:
        scores = torch.tensor([3.0, 1.0, -2.0, -3.0], dtype=torch.float32)
        labels = torch.tensor([0.9, 0, 1, 0], dtype=torch.float32)
        state_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        loss = compute_state_ranking_loss(scores, labels, state_index)

        self.assertLess(float(loss), 0.4)

    def test_evaluate_relation_graph_rankings_reports_all_and_edit_only_metrics(self) -> None:
        metrics = evaluate_relation_graph_rankings(
            state_rows=[
                {
                    "state_id": "s1",
                    "candidate_id": "c1",
                    "label": 0.9,
                    "score": 0.9,
                    "is_commit": False,
                },
                {
                    "state_id": "s1",
                    "candidate_id": "c2",
                    "label": 0,
                    "score": 0.2,
                    "is_commit": True,
                },
                {
                    "state_id": "s2",
                    "candidate_id": "c3",
                    "label": 0,
                    "score": 0.8,
                    "is_commit": False,
                },
                {
                    "state_id": "s2",
                    "candidate_id": "c4",
                    "label": 1,
                    "score": 0.7,
                    "is_commit": False,
                },
                {
                    "state_id": "s3",
                    "candidate_id": "c5",
                    "label": 0,
                    "score": 0.3,
                    "is_commit": False,
                },
                {
                    "state_id": "s3",
                    "candidate_id": "c6",
                    "label": 0,
                    "score": 0.1,
                    "is_commit": True,
                },
            ]
        )

        self.assertEqual(metrics["all"]["state_count"], 2)
        self.assertEqual(metrics["all"]["top1_accuracy"], 0.5)
        self.assertEqual(metrics["all"]["mean_reciprocal_rank"], 0.75)
        self.assertEqual(metrics["edit_only"]["state_count"], 2)
        self.assertEqual(metrics["edit_only"]["top1_accuracy"], 0.5)
        self.assertEqual(metrics["edit_only"]["mean_reciprocal_rank"], 0.75)
