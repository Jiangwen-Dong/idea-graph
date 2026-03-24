from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import _action_system_prompt, _seed_system_prompt, _synthesis_system_prompt
from idea_graph.agent_backend import _prompt_safe_metadata, _resolve_symbolic_reference_text
from idea_graph.collaboration_protocol import resolve_round_phase


class AgentBackendPromptTests(unittest.TestCase):
    def test_round_phase_resolves_for_extended_runs(self) -> None:
        self.assertEqual(resolve_round_phase("Round1").key, "structure")
        self.assertEqual(resolve_round_phase("Round2").key, "stress_test")
        self.assertEqual(resolve_round_phase("Round3").key, "repair")
        self.assertEqual(resolve_round_phase("Round4").key, "stress_test")
        self.assertEqual(resolve_round_phase("Round5").key, "repair")

    def test_action_prompt_discourages_defaulting_to_one_action_kind(self) -> None:
        prompt = _action_system_prompt("MechanismProposer", "Round5")
        self.assertIn("Do not default to a specific action kind", prompt)
        self.assertIn("propose_repair", prompt)
        self.assertIn("freeze_branch", prompt)
        self.assertIn("reference_paper_snippets[0].abstract", prompt)
        self.assertNotIn('{"kind":"add_support_edge","target_ids":["N001","N002"]', prompt)

    def test_seed_prompt_does_not_force_fixed_anchor_type(self) -> None:
        prompt = _seed_system_prompt("FeasibilityCritic")
        self.assertIn("Preferred anchor types for your role", prompt)
        self.assertIn("Choose an anchor type that best fits your role", prompt)
        self.assertNotIn('{"anchor":{"type":"Hypothesis"', prompt)

    def test_synthesis_prompt_requests_structured_research_idea_fields(self) -> None:
        prompt = _synthesis_system_prompt()
        self.assertIn("structured research idea", prompt)
        self.assertIn('"title"', prompt)
        self.assertIn('"existing_methods"', prompt)
        self.assertIn('"motivation"', prompt)
        self.assertIn("Do not output an abstract field", prompt)

    def test_prompt_safe_metadata_hides_target_paper_oracle_fields(self) -> None:
        safe = _prompt_safe_metadata(
            {
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_index": 15,
                "target_paper": "PanoPose",
                "method_summary": "gold summary",
                "motivation": "gold motivation",
                "raw_record": {"secret": True},
                "reference_titles": ["Paper A"],
                "paper_grounding": {
                    "target_paper_snippet": {"abstract": "gold abstract"},
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "Paper A",
                            "abstract": "A concrete abstract.",
                            "method": "A concrete method.",
                        }
                    ],
                },
            }
        )

        self.assertEqual(safe["benchmark"], "AI_Idea_Bench_2025")
        self.assertNotIn("target_paper", safe)
        self.assertNotIn("method_summary", safe)
        self.assertNotIn("motivation", safe)
        self.assertNotIn("raw_record", safe)
        self.assertNotIn("target_paper_snippet", safe["paper_grounding"])

    def test_symbolic_reference_text_is_resolved_into_concrete_evidence(self) -> None:
        resolved = _resolve_symbolic_reference_text(
            "reference_paper_snippets[0].abstract, reference_paper_snippets[1].method",
            {
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "Paper A",
                            "abstract": "Paper A studies self-supervised depth estimation for panoramic scenes.",
                        },
                        {
                            "resolved_title": "Paper B",
                            "method": "Paper B uses a pose network with explicit geometric constraints.",
                        },
                    ]
                }
            },
        )

        self.assertIn("Paper A", resolved)
        self.assertIn("self-supervised depth estimation", resolved)
        self.assertIn("Paper B", resolved)
        self.assertNotIn("reference_paper_snippets[0].abstract", resolved)


if __name__ == "__main__":
    unittest.main()
