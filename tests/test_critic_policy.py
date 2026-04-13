from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_policy import (
    CriticPolicyDecision,
    SafeCriticPolicyConfig,
    ScoredCandidate,
    choose_critic_action,
)


class CriticPolicyTests(unittest.TestCase):
    def _state(self, *, round_index: int) -> dict[str, object]:
        return {"state_id": f"state-{round_index}", "round_index": round_index}

    def _candidate(
        self,
        candidate_id: str,
        *,
        score: float,
        is_commit: bool = False,
        confidence: float | None = None,
    ) -> ScoredCandidate:
        return ScoredCandidate(
            candidate_id=candidate_id,
            score=score,
            is_commit=is_commit,
            confidence=confidence,
        )

    def test_commit_blocked_before_min_round(self) -> None:
        decision = choose_critic_action(
            state=self._state(round_index=0),
            critic_candidates=[
                self._candidate("commit", score=0.95, is_commit=True, confidence=0.95),
                self._candidate("critic-edit", score=0.90),
            ],
            heuristic_candidate=self._candidate("heuristic-edit", score=0.70),
            config=SafeCriticPolicyConfig(min_commit_round=2),
        )
        self.assertIsInstance(decision, CriticPolicyDecision)
        self.assertEqual(decision.selected_candidate_id, "critic-edit")
        self.assertFalse(decision.commit_allowed)

    def test_heuristic_fallback_when_override_margin_is_small(self) -> None:
        decision = choose_critic_action(
            state=self._state(round_index=3),
            critic_candidates=[self._candidate("critic-best", score=0.61)],
            heuristic_candidate=self._candidate("heuristic-best", score=0.60),
            config=SafeCriticPolicyConfig(tau_override=0.05),
        )
        self.assertEqual(decision.selected_candidate_id, "heuristic-best")
        self.assertTrue(decision.used_heuristic_fallback)

    def test_critic_override_when_margin_is_large_enough(self) -> None:
        decision = choose_critic_action(
            state=self._state(round_index=3),
            critic_candidates=[self._candidate("critic-best", score=0.72)],
            heuristic_candidate=self._candidate("heuristic-best", score=0.60),
            config=SafeCriticPolicyConfig(tau_override=0.05),
        )
        self.assertEqual(decision.selected_candidate_id, "critic-best")
        self.assertFalse(decision.used_heuristic_fallback)

    def test_commit_selected_when_margin_and_confidence_are_high_enough(self) -> None:
        decision = choose_critic_action(
            state=self._state(round_index=3),
            critic_candidates=[
                self._candidate("commit", score=0.92, is_commit=True, confidence=0.90),
                self._candidate("critic-edit", score=0.75),
            ],
            heuristic_candidate=self._candidate("heuristic-best", score=0.70),
            config=SafeCriticPolicyConfig(min_commit_round=2, tau_commit=0.08, gamma_commit=0.80),
        )
        self.assertEqual(decision.selected_candidate_id, "commit")
        self.assertTrue(decision.commit_allowed)


if __name__ == "__main__":
    unittest.main()
