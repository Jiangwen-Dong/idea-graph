from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.runtime_critic import (
    TextCriticRuntimeConfig,
    select_text_critic_candidate,
)
from idea_graph.models import Branch, IdeaGraph, Node


class _KeywordScoreModel:
    def __init__(self, mapping: dict[str, float]) -> None:
        self.mapping = dict(mapping)

    def score(self, texts):
        scores = []
        for text in texts:
            matched = 0.0
            for key, value in self.mapping.items():
                if key in text:
                    matched = value
            scores.append(float(matched))
        return scores


class RuntimeCriticTests(unittest.TestCase):
    def _build_graph(self) -> IdeaGraph:
        graph = IdeaGraph(topic="test topic", literature=["paper a"], metadata={})
        graph.branches["B001"] = Branch(id="B001", role="MechanismProposer")
        graph.nodes["N001"] = Node(
            id="N001",
            type="Problem",
            text="A concrete benchmark problem.",
            role="ImpactReframer",
            branch_id="B001",
            confidence=0.8,
        )
        graph.nodes["N002"] = Node(
            id="N002",
            type="Method",
            text="A candidate method idea.",
            role="MechanismProposer",
            branch_id="B001",
            confidence=0.8,
        )
        return graph

    def test_runtime_critic_overrides_heuristic_when_margin_is_large(self) -> None:
        graph = self._build_graph()
        candidates = [
            {
                "candidate_id": "heuristic",
                "kind": "attach_evidence",
                "target_ids": ["N002"],
                "payload": {"branch_id": "B001", "evidence": "weak-signal"},
                "rationale": "heuristic-option",
            },
            {
                "candidate_id": "critic-best",
                "kind": "attach_evidence",
                "target_ids": ["N001"],
                "payload": {"branch_id": "B001", "evidence": "strong-signal"},
                "rationale": "critic-option",
            },
        ]

        decision = select_text_critic_candidate(
            graph,
            round_name="Round2",
            role="MechanismProposer",
            candidate_specs=candidates,
            heuristic_candidate_id="heuristic",
            model=_KeywordScoreModel({"weak-signal": 0.40, "strong-signal": 0.82}),
            config=TextCriticRuntimeConfig(tau_override=0.05, use_commit=False),
        )

        self.assertEqual(decision.policy_decision.selected_candidate_id, "critic-best")
        self.assertEqual(decision.policy_decision.selected_source, "critic")
        self.assertFalse(decision.policy_decision.used_heuristic_fallback)

    def test_runtime_critic_falls_back_when_margin_is_small(self) -> None:
        graph = self._build_graph()
        candidates = [
            {
                "candidate_id": "heuristic",
                "kind": "attach_evidence",
                "target_ids": ["N002"],
                "payload": {"branch_id": "B001", "evidence": "weak-signal"},
                "rationale": "heuristic-option",
            },
            {
                "candidate_id": "critic-close",
                "kind": "attach_evidence",
                "target_ids": ["N001"],
                "payload": {"branch_id": "B001", "evidence": "close-signal"},
                "rationale": "critic-option",
            },
        ]

        decision = select_text_critic_candidate(
            graph,
            round_name="Round2",
            role="MechanismProposer",
            candidate_specs=candidates,
            heuristic_candidate_id="heuristic",
            model=_KeywordScoreModel({"weak-signal": 0.40, "close-signal": 0.43}),
            config=TextCriticRuntimeConfig(tau_override=0.05, use_commit=False),
        )

        self.assertEqual(decision.policy_decision.selected_candidate_id, "heuristic")
        self.assertEqual(decision.policy_decision.selected_source, "heuristic")
        self.assertTrue(decision.policy_decision.used_heuristic_fallback)


if __name__ == "__main__":
    unittest.main()
