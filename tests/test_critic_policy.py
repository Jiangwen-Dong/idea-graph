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
        predicted_gain: float = 0.0,
        support_gain: float = 0.0,
        contradiction_gain: float = 0.0,
        maturity_gain: float = 0.0,
        after_is_mature: bool = False,
    ) -> ScoredCandidate:
        return ScoredCandidate(
            candidate_id=candidate_id,
            score=score,
            is_commit=is_commit,
            confidence=confidence,
            predicted_gain=predicted_gain,
            support_gain=support_gain,
            contradiction_gain=contradiction_gain,
            maturity_gain=maturity_gain,
            after_is_mature=after_is_mature,
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

    def test_override_uses_round_specific_threshold(self) -> None:
        round2_decision = choose_critic_action(
            state=self._state(round_index=2),
            critic_candidates=[self._candidate("critic-best", score=0.72)],
            heuristic_candidate=self._candidate("heuristic-best", score=0.60),
            config=SafeCriticPolicyConfig(
                tau_override=0.05,
                tau_override_by_round={2: 0.15},
            ),
        )
        round3_decision = choose_critic_action(
            state=self._state(round_index=3),
            critic_candidates=[self._candidate("critic-best", score=0.72)],
            heuristic_candidate=self._candidate("heuristic-best", score=0.60),
            config=SafeCriticPolicyConfig(
                tau_override=0.05,
                tau_override_by_round={2: 0.15},
            ),
        )

        self.assertEqual(round2_decision.selected_candidate_id, "heuristic-best")
        self.assertTrue(round2_decision.used_heuristic_fallback)
        self.assertEqual(round2_decision.fallback_reason, "override_margin_below_threshold")
        self.assertEqual(round3_decision.selected_candidate_id, "critic-best")
        self.assertFalse(round3_decision.used_heuristic_fallback)

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

    def test_commit_uses_round_specific_confidence_threshold(self) -> None:
        round3_decision = choose_critic_action(
            state=self._state(round_index=3),
            critic_candidates=[
                self._candidate("commit", score=0.94, is_commit=True, confidence=0.90),
                self._candidate("critic-edit", score=0.76),
            ],
            heuristic_candidate=self._candidate("heuristic-edit", score=0.70),
            config=SafeCriticPolicyConfig(
                min_commit_round=2,
                gamma_commit=0.60,
                gamma_commit_by_round={3: 0.95, 4: 0.80},
            ),
        )
        round4_decision = choose_critic_action(
            state=self._state(round_index=4),
            critic_candidates=[
                self._candidate("commit", score=0.94, is_commit=True, confidence=0.90),
                self._candidate("critic-edit", score=0.76),
            ],
            heuristic_candidate=self._candidate("heuristic-edit", score=0.70),
            config=SafeCriticPolicyConfig(
                min_commit_round=2,
                gamma_commit=0.60,
                gamma_commit_by_round={3: 0.95, 4: 0.80},
            ),
        )

        self.assertEqual(round3_decision.selected_candidate_id, "critic-edit")
        self.assertEqual(round3_decision.fallback_reason, "commit_confidence_below_round_threshold")
        self.assertEqual(round4_decision.selected_candidate_id, "commit")

    def test_blocks_fragile_maturity_jump_without_support_gain(self) -> None:
        decision = choose_critic_action(
            state={
                "round_index": 3,
                "support_coverage": 0.66,
                "unresolved_contradiction_ratio": 0.0,
            },
            critic_candidates=[
                self._candidate(
                    "critic-fragile",
                    score=0.82,
                    predicted_gain=1.10,
                    support_gain=0.00,
                    contradiction_gain=0.00,
                    maturity_gain=1.0,
                    after_is_mature=True,
                ),
            ],
            heuristic_candidate=self._candidate(
                "heuristic-safe",
                score=0.72,
                predicted_gain=0.74,
                support_gain=0.25,
                contradiction_gain=0.00,
                maturity_gain=0.0,
                after_is_mature=False,
            ),
            config=SafeCriticPolicyConfig(
                tau_override=0.05,
                guard_support_threshold=0.66,
                guard_support_gain_floor=0.10,
            ),
        )

        self.assertEqual(decision.selected_candidate_id, "heuristic-safe")
        self.assertEqual(decision.selected_source, "heuristic")
        self.assertTrue(decision.used_heuristic_fallback)

    def test_allows_maturity_override_when_support_gain_is_real(self) -> None:
        decision = choose_critic_action(
            state={
                "round_index": 3,
                "support_coverage": 0.70,
                "unresolved_contradiction_ratio": 0.0,
            },
            critic_candidates=[
                self._candidate(
                    "critic-grounded",
                    score=0.84,
                    predicted_gain=0.90,
                    support_gain=0.20,
                    contradiction_gain=0.00,
                    maturity_gain=1.0,
                    after_is_mature=True,
                ),
            ],
            heuristic_candidate=self._candidate(
                "heuristic",
                score=0.73,
                predicted_gain=0.60,
                support_gain=0.00,
                contradiction_gain=0.00,
                maturity_gain=0.0,
                after_is_mature=False,
            ),
            config=SafeCriticPolicyConfig(
                tau_override=0.05,
                guard_support_threshold=0.66,
                guard_support_gain_floor=0.10,
            ),
        )

        self.assertEqual(decision.selected_candidate_id, "critic-grounded")
        self.assertEqual(decision.selected_source, "critic")
        self.assertFalse(decision.used_heuristic_fallback)

    def test_blocks_override_when_critic_candidate_undercuts_meaningful_heuristic_gain(self) -> None:
        decision = choose_critic_action(
            state={
                "round_index": 3,
                "support_coverage": 0.60,
                "unresolved_contradiction_ratio": 0.0,
            },
            critic_candidates=[
                self._candidate(
                    "critic-low-gain",
                    score=0.92,
                    predicted_gain=0.0,
                    support_gain=0.0,
                    contradiction_gain=0.0,
                    maturity_gain=0.0,
                    after_is_mature=False,
                ),
            ],
            heuristic_candidate=self._candidate(
                "heuristic-high-gain",
                score=0.40,
                predicted_gain=0.80,
                support_gain=0.20,
                contradiction_gain=0.0,
                maturity_gain=0.0,
                after_is_mature=False,
            ),
            config=SafeCriticPolicyConfig(tau_override=0.05),
        )

        self.assertEqual(decision.selected_candidate_id, "heuristic-high-gain")
        self.assertEqual(decision.selected_source, "heuristic")
        self.assertTrue(decision.used_heuristic_fallback)

    def test_allows_override_when_critic_candidate_is_close_to_heuristic_gain(self) -> None:
        decision = choose_critic_action(
            state={
                "round_index": 3,
                "support_coverage": 0.60,
                "unresolved_contradiction_ratio": 0.0,
            },
            critic_candidates=[
                self._candidate(
                    "critic-close-gain",
                    score=0.92,
                    predicted_gain=0.38,
                    support_gain=0.0,
                    contradiction_gain=0.0,
                    maturity_gain=0.0,
                    after_is_mature=False,
                ),
            ],
            heuristic_candidate=self._candidate(
                "heuristic-reference",
                score=0.40,
                predicted_gain=0.50,
                support_gain=0.20,
                contradiction_gain=0.0,
                maturity_gain=0.0,
                after_is_mature=False,
            ),
            config=SafeCriticPolicyConfig(tau_override=0.05),
        )

        self.assertEqual(decision.selected_candidate_id, "critic-close-gain")
        self.assertEqual(decision.selected_source, "critic")
        self.assertFalse(decision.used_heuristic_fallback)

    def test_blocks_override_when_heuristic_gain_is_small_but_still_meaningful(self) -> None:
        decision = choose_critic_action(
            state={
                "round_index": 3,
                "support_coverage": 0.60,
                "unresolved_contradiction_ratio": 0.0,
            },
            critic_candidates=[
                self._candidate(
                    "critic-negative-gain",
                    score=0.90,
                    predicted_gain=-0.40,
                    after_is_mature=False,
                ),
            ],
            heuristic_candidate=self._candidate(
                "heuristic-small-positive",
                score=0.20,
                predicted_gain=0.09,
                after_is_mature=False,
            ),
            config=SafeCriticPolicyConfig(tau_override=0.05),
        )

        self.assertEqual(decision.selected_candidate_id, "heuristic-small-positive")
        self.assertEqual(decision.selected_source, "heuristic")
        self.assertTrue(decision.used_heuristic_fallback)


if __name__ == "__main__":
    unittest.main()
