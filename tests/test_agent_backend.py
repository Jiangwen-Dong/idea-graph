from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import (
    _action_system_prompt,
    _action_user_prompt,
    _dynamic_allowed_actions,
    _postprocess_final_proposal,
    _prompt_safe_metadata,
    _resolve_symbolic_reference_text,
    _salvage_action_decision,
    _seed_system_prompt,
    _seed_user_prompt,
    _synthesis_system_prompt,
)
from idea_graph.collaboration_protocol import resolve_round_phase
from idea_graph.engine import build_seed_graphs, create_edge, merge_seed_graphs
from idea_graph.models import FinalProposal, IdeaGraph


class AgentBackendPromptTests(unittest.TestCase):
    def _build_graph(self) -> IdeaGraph:
        graph = IdeaGraph(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
        )
        build_seed_graphs(graph)
        merge_seed_graphs(graph)
        return graph

    def test_round_phase_resolves_for_extended_runs(self) -> None:
        self.assertEqual(resolve_round_phase("Round1").key, "structure")
        self.assertEqual(resolve_round_phase("Round2").key, "stress_test")
        self.assertEqual(resolve_round_phase("Round3").key, "repair")
        self.assertEqual(resolve_round_phase("Round4").key, "stress_test")
        self.assertEqual(resolve_round_phase("Round5").key, "repair")

    def test_action_prompt_discourages_defaulting_to_one_action_kind(self) -> None:
        graph = self._build_graph()
        allowed_actions = _dynamic_allowed_actions(graph, "Round5")
        prompt = _action_system_prompt(graph, "MechanismProposer", "Round5", allowed_actions)
        self.assertIn("Do not default to a specific action kind", prompt)
        self.assertIn("propose_repair", prompt)
        self.assertIn("Freeze a branch only when preserving it as an alternative", prompt)
        self.assertIn("Never return field references", prompt)
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
        self.assertIn("Avoid generic method phrases", prompt)

    def test_action_user_prompt_includes_evidence_candidates_and_benchmark_focus(self) -> None:
        graph = IdeaGraph(
            topic="The topic of this paper is 3D language field modeling.",
            literature=["3D Gaussian Splatting", "LERF"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_input_packet": {
                    "topic": "The topic of this paper is 3D language field modeling.",
                    "reference_packet": [
                        {"title": "LERF", "snippet": "CLIP embeddings support long-tail open-vocabulary queries."}
                    ],
                },
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "LERF",
                            "method": "LERF supports long-tail open-vocabulary queries.",
                            "evaluation": "Evaluate on the LERF dataset and report localization accuracy.",
                        }
                    ]
                },
            },
        )
        build_seed_graphs(graph)
        merge_seed_graphs(graph)

        prompt = _action_user_prompt(graph, "Round2", "MechanismProposer")

        self.assertIn('"evidence_candidates"', prompt)
        self.assertIn('"benchmark_focus"', prompt)
        self.assertIn('LERF', prompt)
        self.assertIn('localization accuracy', prompt)

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

    def test_duplicate_support_edge_is_salvaged_without_repeating_the_same_pair(self) -> None:
        graph = self._build_graph()
        source = next(node for node in graph.active_nodes() if node.type == "Method")
        target = next(node for node in graph.active_nodes() if node.type == "Problem")
        create_edge(
            graph,
            source_id=source.id,
            relation="supports",
            target_id=target.id,
            role=source.role,
            branch_id=source.branch_id,
            note="Existing support edge for salvage test.",
        )

        kind, target_ids, payload, _ = _salvage_action_decision(
            graph,
            role=source.role,
            kind="add_support_edge",
            target_ids=[source.id, target.id],
            payload={"branch_id": source.branch_id},
            rationale="Test duplicate support salvage.",
        )

        self.assertIn(kind, {"add_support_edge", "attach_evidence", "freeze_branch"})
        if kind == "add_support_edge":
            self.assertNotEqual(target_ids, [source.id, target.id])
        if kind == "attach_evidence":
            self.assertIn("evidence", payload)

    def test_duplicate_evidence_request_can_fall_forward_to_direct_grounding(self) -> None:
        graph = self._build_graph()
        target = next(node for node in graph.active_nodes() if node.type == "Hypothesis")
        create_edge(
            graph,
            source_id=target.id,
            relation="requires_evidence",
            target_id=target.id,
            role=target.role,
            branch_id=target.branch_id,
            note="Existing evidence request for salvage test.",
        )

        kind, target_ids, payload, _ = _salvage_action_decision(
            graph,
            role=target.role,
            kind="request_evidence",
            target_ids=[target.id],
            payload={"branch_id": target.branch_id},
            rationale="Test duplicate evidence-request salvage.",
        )

        self.assertIn(kind, {"request_evidence", "attach_evidence", "freeze_branch"})
        if kind == "request_evidence":
            self.assertNotEqual(target_ids, [target.id])
        if kind == "attach_evidence":
            self.assertEqual(target_ids, [target.id])
            self.assertTrue(payload.get("evidence"))

    def test_postprocess_final_proposal_adds_safe_grounding_specificity(self) -> None:
        graph = IdeaGraph(
            topic="The topic of this paper is 3D language field modeling.",
            literature=["3D Gaussian Splatting", "LERF"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "LERF",
                            "method": "LERF uses CLIP features for open-vocabulary 3D querying.",
                            "evaluation": "Evaluate on the LERF dataset and report localization accuracy and IoU.",
                        },
                        {
                            "resolved_title": "3D Gaussian Splatting",
                            "method": "3D Gaussian Splatting enables efficient radiance-field rendering.",
                            "evaluation": "Report accuracy and IoU on held-out 3D scene understanding tasks.",
                        },
                    ]
                },
            },
        )
        proposal = FinalProposal(
            title="Language-Grounded 3D Fields",
            problem="Existing methods remain limited.",
            existing_methods="Current methods remain limited.",
            motivation="A more grounded 3D language model is needed.",
            hypothesis="A better graph could help.",
            method="Use a hybrid model to improve performance.",
            evaluation="Compare against strong baselines using task-relevant datasets and metrics.",
            significance="This could help.",
            caveats="It may fail.",
        )

        processed = _postprocess_final_proposal(graph, proposal)

        self.assertIn("LERF", processed.existing_methods)
        self.assertIn("LERF dataset", processed.evaluation)
        self.assertTrue("accuracy" in processed.evaluation or "IoU" in processed.evaluation)
        self.assertIn("ablation", processed.evaluation.casefold())

    def test_seed_prompt_exposes_weak_context_scaffold_for_keyword_only_benchmarks(self) -> None:
        graph = IdeaGraph(
            topic="Ideation topic keyword: meteorology",
            literature=[
                "Benchmark keyword: meteorology",
                "This benchmark row provides a keyword prompt rather than retrieved literature.",
            ],
            metadata={
                "benchmark": "liveideabench",
                "keyword": "meteorology",
                "benchmark_input_packet": {
                    "benchmark": "liveideabench",
                    "topic": "Ideation topic keyword: meteorology",
                    "keyword": "meteorology",
                    "reference_packet": [],
                },
            },
        )

        prompt = _seed_user_prompt(graph, "MechanismProposer")

        self.assertIn('"weak_context_mode": true', prompt.lower())
        self.assertIn('"weak_context_scaffold"', prompt)
        self.assertIn('meteorology', prompt)
        self.assertIn('divergence_axes', prompt)

    def test_postprocess_final_proposal_uses_weak_context_scaffold_to_reduce_generic_keyword_only_evaluation(self) -> None:
        graph = IdeaGraph(
            topic="Ideation topic keyword: meteorology",
            literature=[
                "Benchmark keyword: meteorology",
                "This benchmark row provides a keyword prompt rather than retrieved literature.",
            ],
            metadata={
                "benchmark": "liveideabench",
                "keyword": "meteorology",
                "benchmark_input_packet": {
                    "benchmark": "liveideabench",
                    "topic": "Ideation topic keyword: meteorology",
                    "keyword": "meteorology",
                    "reference_packet": [],
                },
            },
        )
        proposal = FinalProposal(
            title="Hybrid Models for Weather Prediction",
            problem="Current approaches remain limited.",
            existing_methods="For meteorology, plausible existing directions include common directions.",
            motivation="Better weather prediction matters.",
            hypothesis="A better model can help.",
            method="Use a hybrid model to improve performance.",
            evaluation="Evaluate on realistic benchmark tasks for meteorology, compare against strong data-driven and hybrid baselines, report task-specific quantitative metrics, and include ablations over the main components.",
            significance="This could help meteorology.",
            caveats="It may fail.",
        )

        processed = _postprocess_final_proposal(graph, proposal)

        self.assertIn("meteorology", processed.existing_methods.lower())
        self.assertNotIn("realistic benchmark tasks", processed.evaluation.lower())
        self.assertNotIn("task-specific quantitative metrics", processed.evaluation.lower())
        self.assertTrue(any(metric.lower() in processed.evaluation.lower() for metric in ("rmse", "mae", "crps")))


if __name__ == "__main__":
    unittest.main()
