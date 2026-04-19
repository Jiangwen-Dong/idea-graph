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
            {"candidate_id": "c0", "kind": "add_support_edge"},
            {"candidate_id": "c1", "kind": "attach_evidence"},
            {"candidate_id": "c2", "kind": "skip"},
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

    def test_returns_copy_of_selected_candidate(self) -> None:
        candidates = [
            {"candidate_id": "c0", "kind": "add_support_edge", "weight": 1.0},
            {"candidate_id": "c1", "kind": "skip", "weight": 2.0},
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
