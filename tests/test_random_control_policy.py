from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.random_control_policy import RandomControlPolicy


class RandomControlPolicyTests(unittest.TestCase):
    def test_seed_stability_for_same_candidate_slate(self) -> None:
        candidates = [
            {"candidate_id": "c0", "kind": "add_support_edge", "candidate_source": "utility_add_support"},
            {"candidate_id": "c1", "kind": "attach_evidence", "candidate_source": "utility_attach_evidence"},
            {"candidate_id": "c2", "kind": "skip", "candidate_source": "parallel_skip"},
        ]
        policy_a = RandomControlPolicy(seed=7)
        policy_b = RandomControlPolicy(seed=7)

        picks_a = [
            policy_a.choose(round_name="Round2", role="MechanismProposer", candidate_specs=candidates)[
                "candidate_id"
            ]
            for _ in range(5)
        ]
        picks_b = [
            policy_b.choose(round_name="Round2", role="MechanismProposer", candidate_specs=candidates)[
                "candidate_id"
            ]
            for _ in range(5)
        ]

        self.assertEqual(picks_a, picks_b)

    def test_samples_only_from_utility_candidates(self) -> None:
        candidates = [
            {"candidate_id": "c0", "kind": "add_support_edge", "candidate_source": "parallel_selected"},
            {"candidate_id": "c1", "kind": "attach_evidence", "candidate_source": "utility_attach_evidence"},
            {"candidate_id": "c2", "kind": "add_dependency_edge", "candidate_source": "utility_add_dependency"},
            {"candidate_id": "c3", "kind": "skip", "candidate_source": "parallel_skip"},
        ]
        policy = RandomControlPolicy(seed=3)

        picked_ids = {
            policy.choose(round_name="Round2", role="MechanismProposer", candidate_specs=candidates)[
                "candidate_id"
            ]
            for _ in range(16)
        }

        self.assertTrue(picked_ids)
        self.assertTrue(picked_ids.issubset({"c1", "c2"}))

    def test_falls_back_to_skip_when_no_utility_candidate_exists(self) -> None:
        candidates = [
            {"candidate_id": "c0", "kind": "freeze_branch", "candidate_source": "parallel_selected"},
            {"candidate_id": "c1", "kind": "skip", "candidate_source": "parallel_skip"},
        ]
        policy = RandomControlPolicy(seed=1)

        selected = policy.choose(round_name="Round3", role="MechanismProposer", candidate_specs=candidates)

        self.assertEqual(selected["candidate_id"], "c1")
        self.assertEqual(selected["kind"], "skip")

    def test_returns_copy_of_selected_candidate(self) -> None:
        candidates = [
            {"candidate_id": "c0", "kind": "add_support_edge", "weight": 1.0, "candidate_source": "utility_add_support"},
            {"candidate_id": "c1", "kind": "skip", "weight": 2.0, "candidate_source": "parallel_skip"},
        ]
        policy = RandomControlPolicy(seed=1)

        selected = policy.choose(round_name="Round1", role="MechanismProposer", candidate_specs=candidates)

        self.assertIsInstance(selected, dict)
        self.assertIn("candidate_id", selected)
        self.assertIn("kind", selected)
        self.assertIsNot(selected, candidates[0])
        self.assertIsNot(selected, candidates[1])


if __name__ == "__main__":
    unittest.main()
