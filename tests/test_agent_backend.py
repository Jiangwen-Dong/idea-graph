from __future__ import annotations

import json
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
    _synthesis_user_prompt,
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

    def test_seed_prompt_uses_functional_role_contract_language(self) -> None:
        prompt = _seed_system_prompt("ImpactReframer")

        self.assertIn("TaskFramer", prompt)
        self.assertIn("exact benchmark task", prompt)
        self.assertIn("failure mode", prompt)

    def test_action_prompt_uses_literature_grounder_contract_language(self) -> None:
        graph = self._build_graph()
        allowed_actions = _dynamic_allowed_actions(graph, "Round2")

        prompt = _action_system_prompt(graph, "NoveltyExaminer", "Round2", allowed_actions)

        self.assertIn("LiteratureGrounder", prompt)
        self.assertIn("visible reference-based gap", prompt)
        self.assertIn("novelty boundaries", prompt)

    def test_synthesis_prompt_requests_structured_research_idea_fields(self) -> None:
        prompt = _synthesis_system_prompt()
        self.assertIn("structured research idea", prompt)
        self.assertIn('"title"', prompt)
        self.assertIn('"existing_methods"', prompt)
        self.assertIn('"motivation"', prompt)
        self.assertIn("Do not output an abstract field", prompt)
        self.assertIn("Avoid generic method phrases", prompt)

    def test_synthesis_user_prompt_exposes_selected_claim_chain(self) -> None:
        graph = IdeaGraph(
            topic="The topic of this paper is human pose and shape estimation using LiDAR in uncontrolled environments.",
            literature=["LiDAR-Aid Inertial Poser", "Learning from Synthetic Humans", "SLOPER4D"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_input_packet": {
                    "benchmark": "AI_Idea_Bench_2025",
                    "topic": "Human pose and shape estimation using LiDAR in uncontrolled environments.",
                },
            },
        )
        build_seed_graphs(graph)
        merge_seed_graphs(graph)
        subgraph = {
            "node_ids": [node.id for node in graph.active_nodes()],
            "edge_ids": [edge.id for edge in graph.edges],
        }

        payload = json.loads(_synthesis_user_prompt(graph, subgraph))

        self.assertIn("selected_claim_chain", payload)
        self.assertIn("coverage", payload["selected_claim_chain"])

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

    def test_postprocess_final_proposal_avoids_raw_weak_context_scaffold_leakage(self) -> None:
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
            title="Physics-Guided Multi-Source Severe Weather Forecasting",
            problem="Short-term severe weather forecasts in regions with sparse ground sensor coverage suffer from poor accuracy due to limited observational data.",
            existing_methods="Current methods remain limited.",
            motivation="Integrating satellite and radar data with physics-guided models can enhance forecast accuracy.",
            hypothesis="A hybrid model that combines neural networks with physical equations will improve severe weather prediction accuracy.",
            method=(
                "The proposed method integrates a physics-based constraint module into a neural network architecture "
                "to enforce dynamical consistency during forecast generation. It also fuses satellite and radar data "
                "with reanalysis signals to compensate for missing ground observations. A calibration layer is added "
                "to model uncertainty in rare events and regional shifts."
            ),
            evaluation="Evaluate on realistic benchmark tasks for meteorology, compare against strong data-driven and hybrid baselines, report task-specific quantitative metrics, and include ablations over the main components.",
            significance="This could improve severe-weather forecasting in sparse-observation regions.",
            caveats="Regional distribution shifts may still reduce robustness.",
        )

        processed = _postprocess_final_proposal(graph, proposal)

        self.assertNotIn("Physics-Guided Forecasting:", processed.method)
        self.assertNotIn("Multi-Source Data Fusion:", processed.method)
        self.assertIn("dynamical consistency", processed.method.lower())

    def test_postprocess_final_proposal_removes_noisy_evaluation_fragments(self) -> None:
        graph = IdeaGraph(
            topic="The topic of this paper is Human Pose and Shape estimation with LiDAR.",
            literature=[],
        )
        proposal = FinalProposal(
            title="LiDAR HPS",
            problem="Occlusion remains challenging.",
            existing_methods="Existing methods remain limited.",
            motivation="We need stronger robustness.",
            hypothesis="A better fusion strategy can help.",
            method="Fuse LiDAR and inertial sensing.",
            evaluation=(
                "Evaluate on experiments were conducted on paper introduces FreeMotion, a novel dataset "
                "captured in diverse real scenarios with multi-modal and multi-view data. "
                "Report Chamfer distance (SUCD). Compare against strong baselines."
            ),
            significance="This could help.",
            caveats="It may fail.",
        )

        processed = _postprocess_final_proposal(graph, proposal)

        self.assertNotIn("experiments were conducted on paper introduces", processed.evaluation.lower())
        self.assertNotIn("captured in diverse real scenarios", processed.evaluation.lower())

    def test_postprocess_final_proposal_expands_sparse_weak_context_evaluation(self) -> None:
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
            title="Physics-Guided Multi-Source Severe Weather Forecasting",
            problem="Short-term severe weather forecasts in regions with sparse ground sensor coverage suffer from limited data availability and poor spatiotemporal resolution.",
            existing_methods="For meteorology, safe starting directions include spatiotemporal forecasting models, physics-aware simulation or data-assimilation pipelines, and multi-source environmental data fusion.",
            motivation="Integrating satellite, radar, and reanalysis data with physics-based constraints can enhance forecast accuracy.",
            hypothesis="A hybrid model that combines neural networks with physical equations governing atmospheric dynamics will improve severe weather prediction accuracy.",
            method="Implement a physics-guided forecasting model that integrates satellite and radar data with reanalysis signals to compensate for sparse ground observations.",
            evaluation="Evaluate the model on regional severe-weather case studies using RMSE, CRPS, and event F1 scores. Compare against baseline forecasting systems. Conduct ablation studies to assess the impact of physics constraints and data fusion strategies on performance.",
            significance="This approach could improve robust severe-weather forecasting.",
            caveats="Regional distribution shift and overfitting remain concerns.",
        )

        processed = _postprocess_final_proposal(graph, proposal)

        self.assertIn("stress test", processed.evaluation.lower())
        self.assertTrue(
            any(
                item in processed.evaluation.lower()
                for item in (
                    "reanalysis-based forecasting tasks",
                    "satellite and radar nowcasting benchmarks",
                )
            )
        )

    def test_postprocess_final_proposal_completes_weak_context_evaluation_coverage(self) -> None:
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
            title="Physics-Guided Spatiotemporal Forecasting for Regional Severe Weather",
            problem="How can we improve short-term severe weather forecasts in regions with sparse ground sensor coverage?",
            existing_methods="For meteorology, safe starting directions include spatiotemporal forecasting models, physics-aware simulation or data-assimilation pipelines, and multi-source environmental data fusion.",
            motivation="Accurate severe weather forecasting is critical for disaster preparedness.",
            hypothesis="Physics-guided forecasting can improve spatiotemporal prediction accuracy while preserving predictive flexibility.",
            method="The proposed method combines a physics-based constraint module with neural networks and integrates satellite and radar data with reanalysis signals.",
            evaluation="Evaluate on satellite and radar nowcasting benchmarks using RMSE and CRPS. Conduct ablation studies to assess the impact of physics constraints and data fusion. Stress test regional distribution shift and rare-event imbalance using regional severe-weather case studies. Also report MAE, event F1.",
            significance="This approach could improve severe weather forecasting in data-scarce regions.",
            caveats="Overfitting and regional distribution shift may still reduce robustness.",
        )

        processed = _postprocess_final_proposal(graph, proposal)

        self.assertIn("reanalysis-based forecasting tasks", processed.evaluation.lower())
        self.assertIn("calibration error", processed.evaluation.lower())
        self.assertIn("compare against", processed.evaluation.lower())

    def test_postprocess_final_proposal_adds_weak_context_method_limitations(self) -> None:
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
            title="Meteorology Forecasting",
            problem="Current severe weather prediction models remain limited.",
            existing_methods="For meteorology, safe starting directions include spatiotemporal forecasting models, physics-aware simulation or data-assimilation pipelines, and multi-source environmental data fusion.",
            motivation="Better forecasts matter.",
            hypothesis="A stronger forecasting method can help.",
            method="Use a stronger forecasting method.",
            evaluation="Evaluate on realistic benchmark tasks for meteorology, compare against strong baselines, and report task-specific quantitative metrics.",
            significance="This could help.",
            caveats="It may fail.",
        )

        processed = _postprocess_final_proposal(graph, proposal)

        self.assertIn("however", processed.existing_methods.lower())
        self.assertTrue(
            any(
                item in processed.existing_methods.lower()
                for item in ("regional distribution shift", "rare-event imbalance")
            )
        )

    def test_postprocess_final_proposal_uses_structured_benchmark_grounding_for_aiib(self) -> None:
        graph = IdeaGraph(
            topic="The topic of this paper is 3D language field modeling.",
            literature=["3D Gaussian Splatting", "LERF"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "literature_grounding": {
                    "source": "paper_snippets",
                    "reference_titles": [
                        "3D Gaussian Splatting",
                        "LERF: Language Embedded Radiance Fields",
                    ],
                    "design_highlights": [
                        "Scene-specific language autoencoder for compact CLIP features.",
                    ],
                    "dataset_items": [
                        "LERF dataset",
                        "App Polycam",
                        "3D-OVS dataset",
                    ],
                    "metric_items": [
                        "localization accuracy",
                        "IoU",
                        "mIoU",
                    ],
                    "existing_methods_summary": (
                        "The provided literature context includes 3D Gaussian Splatting and LERF."
                    ),
                    "experiment_plan_summary": (
                        "Evaluate on LERF dataset, App Polycam, and 3D-OVS dataset. "
                        "Report localization accuracy, IoU, and mIoU."
                    ),
                    "weak_context_scaffold": {},
                },
            },
        )
        proposal = FinalProposal(
            title="Language-Driven 3D Radiance Field Modeling",
            problem="Current methods struggle to connect language and 3D scene representations.",
            existing_methods="Current methods remain limited.",
            motivation="Better language-aware 3D modeling would improve open-vocabulary scene understanding.",
            hypothesis="A stronger representation can help.",
            method="Use a stronger representation for language-aware radiance fields.",
            evaluation="Compare against strong baselines and include ablations.",
            significance="This could help 3D scene understanding.",
            caveats="It may introduce extra compute cost.",
        )

        processed = _postprocess_final_proposal(graph, proposal)

        self.assertIn("LERF dataset", processed.evaluation)
        self.assertIn("App Polycam", processed.evaluation)
        self.assertTrue(
            any(metric in processed.evaluation for metric in ("localization accuracy", "IoU", "mIoU"))
        )
        self.assertIn("LERF", processed.existing_methods)

    def test_postprocess_final_proposal_removes_exact_scaffold_sentence_from_method(self) -> None:
        graph = IdeaGraph(
            topic="The topic of this paper is 3D language field modeling.",
            literature=["3D Gaussian Splatting", "LERF"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "literature_grounding": {
                    "source": "paper_snippets",
                    "reference_titles": [
                        "3D Gaussian Splatting",
                        "LERF: Language Embedded Radiance Fields",
                    ],
                    "design_highlights": [
                        "3D Gaussian Splatting with Language Features: LangSplat represents a 3D scene as a collection of 3D Gaussians, each enhanced with language embeddings derived from CLIP features.",
                    ],
                    "dataset_items": [],
                    "metric_items": [],
                    "existing_methods_summary": "The provided literature context includes 3D Gaussian Splatting and LERF.",
                    "experiment_plan_summary": "",
                    "weak_context_scaffold": {},
                },
            },
        )
        proposal = FinalProposal(
            title="Language-Guided 3D Radiance Fields",
            problem="Current methods remain limited.",
            existing_methods="Current methods remain limited.",
            motivation="Better language-aware 3D fields matter.",
            hypothesis="A stronger representation can help.",
            method=(
                "We encode CLIP-aligned language features directly into 3D Gaussian splats for open-vocabulary querying. "
                "A hierarchical query stage refines scene regions without explicit masks. "
                "Use 3d gaussian splatting with language features to LangSplat represents a 3D scene as a collection of 3D Gaussians, each enhanced with language embeddings derived from CLIP features."
            ),
            evaluation="Compare against strong baselines and include ablations.",
            significance="This could help.",
            caveats="It may add compute cost.",
        )

        processed = _postprocess_final_proposal(graph, proposal)

        self.assertNotIn("Use 3d gaussian splatting with language features", processed.method)
        self.assertIn("CLIP-aligned language features", processed.method)

    def test_postprocess_final_proposal_removes_exact_weak_context_scaffold_sentence_from_method(self) -> None:
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
            title="Physics-Guided Multi-Source Severe Weather Forecasting",
            problem="Forecasting severe weather remains difficult under sparse observations.",
            existing_methods="Current methods remain limited.",
            motivation="Better forecasts matter.",
            hypothesis="Physics-guided data fusion can help.",
            method=(
                "The proposed method integrates a physics-based constraint module into a neural network architecture "
                "to enforce dynamical consistency during forecast generation. "
                "It also fuses satellite and radar data with reanalysis signals to compensate for missing ground observations. "
                "Use multi-source data fusion to integrate satellite, radar, station, or reanalysis signals to improve coverage."
            ),
            evaluation="Evaluate on realistic benchmark tasks for meteorology, compare against strong baselines, and report task-specific quantitative metrics.",
            significance="This could help.",
            caveats="It may fail.",
        )

        processed = _postprocess_final_proposal(graph, proposal)

        self.assertNotIn("Use multi-source data fusion", processed.method)
        self.assertIn("reanalysis signals", processed.method)


if __name__ == "__main__":
    unittest.main()
