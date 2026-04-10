from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import ActionDecision
from idea_graph.claim_chain import select_claim_chain
from idea_graph.engine import (
    build_seed_graphs,
    choose_round_action,
    create_branch,
    create_edge,
    create_node,
    maturity_snapshot,
    merge_seed_graphs,
    run_experiment,
    select_final_subgraph,
    synthesize_proposal,
    unresolved_contradiction_edges,
    utility_breakdown,
)
from idea_graph.models import FinalProposal, IdeaGraph, MaturitySnapshot
from idea_graph.models import UtilityBreakdown


class InvalidActionBackend:
    name = "openai-compatible"

    def generate_seed(self, graph: IdeaGraph, role: str):
        raise RuntimeError("seed generation disabled for test")

    def choose_action(self, graph: IdeaGraph, round_name: str, role: str) -> ActionDecision:
        return ActionDecision(
            kind="add_support_edge",
            target_ids=[],
            payload={"branch_id": "B001"},
            rationale="invalid test action",
        )

    def synthesize_final_proposal(self, graph: IdeaGraph, subgraph: dict[str, object]) -> FinalProposal:
        return FinalProposal(
            problem="p",
            hypothesis="h",
            method="m",
            evaluation="e",
            significance="s",
            caveats="c",
        )


class MisalignedRepairBackend:
    name = "openai-compatible"

    def generate_seed(self, graph: IdeaGraph, role: str):
        raise RuntimeError("seed generation disabled for test")

    def choose_action(self, graph: IdeaGraph, round_name: str, role: str) -> ActionDecision:
        branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
        if round_name == "Round3":
            unresolved = unresolved_contradiction_edges(graph)
            target_id = unresolved[0].source_id if unresolved else graph.active_nodes()[0].id
            return ActionDecision(
                kind="propose_repair",
                target_ids=[target_id],
                payload={"branch_id": branch_id, "repair_text": "Apply a generic repair."},
                rationale="Deliberately misaligned repair for fallback testing.",
            )

        action = choose_round_action(graph, round_name, role)
        return ActionDecision(
            kind=action.kind,
            target_ids=list(action.target_ids),
            payload=dict(action.payload),
            rationale=action.rationale,
        )

    def synthesize_final_proposal(self, graph: IdeaGraph, subgraph: dict[str, object]) -> FinalProposal:
        return FinalProposal(
            problem="p",
            hypothesis="h",
            method="m",
            evaluation="e",
            significance="s",
            caveats="c",
        )


class EngineTests(unittest.TestCase):
    def _build_seed_graph(self) -> IdeaGraph:
        graph = IdeaGraph(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
        )
        build_seed_graphs(graph)
        merge_seed_graphs(graph)
        return graph

    def test_invalid_llm_actions_fall_back_without_crashing(self) -> None:
        messages: list[str] = []
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            collaboration_backend=InvalidActionBackend(),
            progress_callback=messages.append,
        )

        self.assertIsNotNone(graph.final_proposal)
        self.assertGreater(len(graph.actions), 0)
        self.assertIn("seed_generation_error", graph.metadata)
        self.assertEqual(len(graph.metadata.get("action_errors", [])), len(graph.actions))
        self.assertTrue(any("using deterministic fallback" in message for message in messages))

    def test_progress_callback_receives_round_updates(self) -> None:
        messages: list[str] = []
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            progress_callback=messages.append,
        )

        self.assertIsNotNone(graph.final_proposal)
        self.assertTrue(any("Round1 started" in message for message in messages))
        self.assertTrue(any("Run complete" in message for message in messages))
        self.assertIn("progress_log", graph.metadata)

    def test_progress_callback_uses_functional_role_display_names(self) -> None:
        messages: list[str] = []
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            progress_callback=messages.append,
        )

        self.assertIsNotNone(graph.final_proposal)
        self.assertTrue(any("TaskFramer" in message for message in messages))
        self.assertTrue(any("LiteratureGrounder" in message for message in messages))

    def test_deterministic_synthesis_produces_richer_structured_fields(self) -> None:
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            max_rounds=3,
            stop_when_mature=False,
        )

        self.assertIsNotNone(graph.final_proposal)
        assert graph.final_proposal is not None
        self.assertTrue(graph.final_proposal.title)
        self.assertEqual(graph.final_proposal.abstract, "")
        self.assertTrue(graph.final_proposal.problem)
        self.assertTrue(graph.final_proposal.existing_methods)
        self.assertTrue(graph.final_proposal.motivation)
        self.assertTrue(graph.final_proposal.method)
        self.assertTrue(graph.final_proposal.evaluation)

    def test_deterministic_synthesis_uses_grounded_datasets_and_metrics_when_available(self) -> None:
        metadata = {
            "target_paper": "PanoPose",
            "method_summary": (
                "PanoPose comprises a depth-net and a pose-net, utilizing self-supervision through "
                "image reconstruction based on estimated depth and relative pose."
            ),
            "raw_record": {
                "summary": {
                    "method": {
                        "datasets": (
                            "Experiments were conducted on several datasets, including "
                            "PanoSUNCG (synthetic indoor), Mapillary Metropolis (real-world panoramic images), "
                            "360VO Dataset (synthetic urban scenes), and custom datasets Building and Campus "
                            "(collected with an Insta 360 ONE X2 camera)."
                        ),
                        "metrics": (
                            "Evaluation metrics include Relative Rotation Error (RRE), Relative Translation "
                            "Angle Error (RTAE), and Relative Scale Error (RSE). For global pose estimation, "
                            "Absolute Rotation Error (ARE) and Absolute Translation Error (ATE) are used."
                        ),
                    }
                }
            },
        }
        graph = IdeaGraph(
            topic="The topic of this paper is estimating scaled relative poses in panoramic images.",
            literature=[
                "Unsupervised learning of depth and ego-motion from video",
                "An efficient solution to the five-point relative pose problem",
            ],
            metadata=metadata,
        )
        build_seed_graphs(graph)
        merge_seed_graphs(graph)
        subgraph = {
            "node_ids": [
                node.id
                for node in graph.active_nodes()
                if node.type in {"Problem", "Hypothesis", "Method", "EvalPlan"}
            ][:4],
            "edge_ids": [],
        }

        proposal = synthesize_proposal(graph, subgraph)

        self.assertEqual(proposal.abstract, "")
        self.assertTrue(
            any(
                dataset_name in proposal.evaluation
                for dataset_name in ("PanoSUNCG (synthetic indoor)", "Mapillary Metropolis", "360VO Dataset")
            )
        )
        self.assertIn("Relative Rotation Error (RRE)", proposal.evaluation)
        self.assertNotIn("...", proposal.evaluation)

    def test_run_experiment_respects_custom_max_rounds(self) -> None:
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            max_rounds=5,
            stop_when_mature=False,
        )

        self.assertEqual(len(graph.round_summaries), 5)
        self.assertEqual(len(graph.actions), 25)
        self.assertEqual(graph.metadata.get("executed_round_count"), 5)
        self.assertEqual(graph.metadata.get("stop_reason"), "max_rounds_reached")

    def test_run_experiment_stops_early_when_mature(self) -> None:
        mature_snapshot = MaturitySnapshot(
            support_coverage=0.8,
            unresolved_contradiction_ratio=0.0,
            utility=8.0,
            utility_stable=True,
            completeness=True,
            is_mature=True,
        )
        immature_snapshot = MaturitySnapshot(
            support_coverage=0.3,
            unresolved_contradiction_ratio=1.0,
            utility=4.0,
            utility_stable=False,
            completeness=False,
            is_mature=False,
        )

        with patch("idea_graph.engine.maturity_snapshot", side_effect=[immature_snapshot, mature_snapshot]):
            graph = run_experiment(
                topic="graph-based scientific ideation",
                literature=["paper a", "paper b", "paper c", "paper d"],
                max_rounds=6,
                stop_when_mature=True,
            )

        self.assertEqual(len(graph.round_summaries), 2)
        self.assertEqual(graph.matured_at_round, "Round2")
        self.assertTrue(graph.metadata.get("stopped_early"))
        self.assertEqual(graph.metadata.get("stop_reason"), "mature_at_Round2")

    def test_run_experiment_does_not_mature_in_round1_without_enough_history(self) -> None:
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            max_rounds=2,
            stop_when_mature=True,
        )

        self.assertGreaterEqual(len(graph.round_summaries), 2)

    def test_select_final_subgraph_prefers_claim_chain_when_available(self) -> None:
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
        problem_branch = create_branch(graph, "ImpactReframer")
        gap_branch = create_branch(graph, "NoveltyExaminer")
        method_branch = create_branch(graph, "MechanismProposer")
        eval_branch = create_branch(graph, "EvaluationDesigner")
        risk_branch = create_branch(graph, "FeasibilityCritic")

        problem = create_node(
            graph,
            node_type="Problem",
            text="LiDAR HPS estimation degrades under occlusion in uncontrolled environments.",
            role="ImpactReframer",
            branch_id=problem_branch.id,
            confidence=0.85,
        )
        gap = create_node(
            graph,
            node_type="NoveltyClaim",
            text="Current LiDAR-inertial methods do not model occlusion-driven distribution shift.",
            role="NoveltyExaminer",
            branch_id=gap_branch.id,
            confidence=0.82,
            evidence=["LiDAR-inertial baselines are brittle under occlusion."],
        )
        method = create_node(
            graph,
            node_type="Method",
            text="Use geometry-aware synthetic occlusion generation with mesh-aligned distillation.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.87,
            evidence=["Synthetic augmentation can close real-world gaps."],
        )
        evaluation = create_node(
            graph,
            node_type="EvalPlan",
            text="Evaluate on SLOPER4D with MPJPE and occlusion-stress ablations.",
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            confidence=0.84,
        )
        risk = create_node(
            graph,
            node_type="Risk",
            text="Synthetic occlusion may still mismatch free-environment sensor noise.",
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
            confidence=0.76,
        )

        create_edge(
            graph,
            source_id=gap.id,
            relation="supports",
            target_id=problem.id,
            role="NoveltyExaminer",
            branch_id=gap_branch.id,
        )
        create_edge(
            graph,
            source_id=method.id,
            relation="supports",
            target_id=gap.id,
            role="MechanismProposer",
            branch_id=method_branch.id,
        )
        create_edge(
            graph,
            source_id=evaluation.id,
            relation="depends_on",
            target_id=method.id,
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
        )
        create_edge(
            graph,
            source_id=risk.id,
            relation="contradicts",
            target_id=method.id,
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
        )

        chain = select_claim_chain(graph)
        final_subgraph = select_final_subgraph(graph)

        assert chain is not None
        self.assertTrue(chain["coverage"]["is_synthesis_ready"])
        self.assertEqual(final_subgraph.get("selection_mode"), "claim_chain")
        self.assertEqual(set(final_subgraph["node_ids"]), set(chain["subgraph"]["node_ids"]))

    def test_maturity_requires_complete_claim_chain(self) -> None:
        graph = IdeaGraph(
            topic="The topic of this paper is human pose and shape estimation using LiDAR in uncontrolled environments.",
            literature=["LiDAR-Aid Inertial Poser", "SLOPER4D"],
            metadata={"benchmark": "AI_Idea_Bench_2025"},
        )
        problem_branch = create_branch(graph, "ImpactReframer")
        gap_branch = create_branch(graph, "NoveltyExaminer")
        method_branch = create_branch(graph, "MechanismProposer")
        eval_branch = create_branch(graph, "EvaluationDesigner")

        problem = create_node(
            graph,
            node_type="Problem",
            text="LiDAR HPS estimation degrades under occlusion in uncontrolled environments.",
            role="ImpactReframer",
            branch_id=problem_branch.id,
            confidence=0.85,
        )
        gap = create_node(
            graph,
            node_type="NoveltyClaim",
            text="Current LiDAR-inertial methods do not model occlusion-driven distribution shift.",
            role="NoveltyExaminer",
            branch_id=gap_branch.id,
            confidence=0.82,
            evidence=["LiDAR-inertial baselines are brittle under occlusion."],
        )
        method = create_node(
            graph,
            node_type="Method",
            text="Use geometry-aware synthetic occlusion generation with mesh-aligned distillation.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.87,
            evidence=["Synthetic augmentation can close real-world gaps."],
        )
        hypothesis = create_node(
            graph,
            node_type="Hypothesis",
            text="Mesh-aligned occlusion synthesis will improve robustness.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.83,
        )
        evaluation = create_node(
            graph,
            node_type="EvalPlan",
            text="Evaluate on SLOPER4D with MPJPE and occlusion-stress ablations.",
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            confidence=0.84,
            evidence=["Use MPJPE and occlusion-stratified evaluation."],
        )

        create_edge(
            graph,
            source_id=gap.id,
            relation="supports",
            target_id=problem.id,
            role="NoveltyExaminer",
            branch_id=gap_branch.id,
        )
        create_edge(
            graph,
            source_id=method.id,
            relation="supports",
            target_id=gap.id,
            role="MechanismProposer",
            branch_id=method_branch.id,
        )
        create_edge(
            graph,
            source_id=hypothesis.id,
            relation="refines",
            target_id=method.id,
            role="MechanismProposer",
            branch_id=method_branch.id,
        )
        create_edge(
            graph,
            source_id=evaluation.id,
            relation="depends_on",
            target_id=method.id,
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
        )

        graph.utility_history = [7.9, 8.0, 8.05]
        with patch(
            "idea_graph.engine.utility_breakdown",
            return_value=UtilityBreakdown(
                promise=0.8,
                support=1.0,
                coherence=0.82,
                evidence=0.42,
                novelty=0.7,
                contradiction_penalty=0.0,
                open_risk_penalty=0.0,
                size_penalty=0.0,
                total=8.2,
            ),
        ):
            snapshot = maturity_snapshot(graph)

        self.assertFalse(snapshot.is_mature)

    def test_choose_round_action_handles_missing_impact_hypothesis_in_late_rounds(self) -> None:
        graph = self._build_seed_graph()
        for node in graph.active_nodes():
            if node.role == "ImpactReframer" and node.type == "Hypothesis":
                node.status = "archived-for-test"

        action = choose_round_action(graph, "Round4", "FeasibilityCritic")

        self.assertEqual(action.kind, "attach_evidence")
        self.assertEqual(len(action.target_ids), 1)

    def test_choose_round_action_handles_missing_impact_hypothesis_in_structure_phase(self) -> None:
        graph = self._build_seed_graph()
        for node in graph.active_nodes():
            if node.role == "ImpactReframer" and node.type == "Hypothesis":
                node.status = "archived-for-test"

        action = choose_round_action(graph, "Round1", "ImpactReframer")

        self.assertEqual(action.kind, "add_support_edge")
        self.assertEqual(len(action.target_ids), 2)

    def test_choose_round_action_targets_unresolved_contradictions_in_repair_phase(self) -> None:
        graph = self._build_seed_graph()

        action = choose_round_action(graph, "Round3", "MechanismProposer")
        contradiction_target_ids = {
            edge.target_id for edge in unresolved_contradiction_edges(graph) if not edge.resolved
        }

        self.assertEqual(action.kind, "propose_repair")
        self.assertIn(action.target_ids[0], contradiction_target_ids)

    def test_misaligned_llm_repair_actions_fall_back_to_contradiction_targeted_repairs(self) -> None:
        messages: list[str] = []
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            collaboration_backend=MisalignedRepairBackend(),
            progress_callback=messages.append,
            max_rounds=3,
            stop_when_mature=False,
        )

        self.assertIsNotNone(graph.final_proposal)
        self.assertTrue(
            any(
                "does not target any unresolved contradiction target" in error["error"]
                for error in graph.metadata.get("action_errors", [])
            )
        )
        self.assertTrue(any("using deterministic fallback" in message for message in messages))
        self.assertLess(graph.round_summaries[-1][1].unresolved_contradiction_ratio, 1.0)

    def test_utility_breakdown_exposes_component_scores(self) -> None:
        graph = self._build_seed_graph()

        breakdown = utility_breakdown(graph)

        self.assertGreaterEqual(breakdown.total, 0.0)
        self.assertLessEqual(breakdown.total, 10.0)
        self.assertGreaterEqual(breakdown.promise, 0.0)
        self.assertLessEqual(breakdown.coherence, 1.0)

    def test_final_subgraph_includes_selection_metadata(self) -> None:
        graph = self._build_seed_graph()

        subgraph = select_final_subgraph(graph)

        self.assertIn("selection_mode", subgraph)
        self.assertIn("utility_breakdown", subgraph)
        self.assertAlmostEqual(
            float(subgraph["utility"]),
            float(subgraph["utility_breakdown"]["total"]),
            places=2,
        )
        self.assertTrue(subgraph["node_ids"])

    def test_keyword_only_weak_context_maturity_requires_keyword_specific_structure(self) -> None:
        graph = IdeaGraph(
            topic="Ideation topic keyword: meteorology",
            literature=["Benchmark keyword: meteorology"],
            metadata={
                "benchmark": "liveideabench",
                "keyword": "meteorology",
                "idea_graph_min_rounds_before_maturity": 2,
            },
        )
        graph.utility_history = [7.6]
        branch = create_branch(graph, "MechanismProposer")
        problem = create_node(
            graph,
            node_type="Problem",
            text="Current approaches remain limited in accuracy and reliability.",
            role="ImpactReframer",
            branch_id=branch.id,
            confidence=0.82,
        )
        hypothesis = create_node(
            graph,
            node_type="Hypothesis",
            text="A hybrid model can improve performance on the task.",
            role="MechanismProposer",
            branch_id=branch.id,
            confidence=0.8,
        )
        method = create_node(
            graph,
            node_type="Method",
            text="Use a hybrid neural architecture with physics constraints.",
            role="MechanismProposer",
            branch_id=branch.id,
            confidence=0.82,
        )
        eval_plan = create_node(
            graph,
            node_type="EvalPlan",
            text="Evaluate on realistic benchmark tasks and compare against strong baselines.",
            role="EvaluationDesigner",
            branch_id=branch.id,
            confidence=0.8,
        )
        novelty = create_node(
            graph,
            node_type="NoveltyClaim",
            text="The system integrates multiple components in a coherent graph.",
            role="NoveltyExaminer",
            branch_id=branch.id,
            confidence=0.78,
        )
        risk = create_node(
            graph,
            node_type="Risk",
            text="The generic hybrid design may fail under seasonal distribution shift.",
            role="FeasibilityCritic",
            branch_id=branch.id,
            confidence=0.76,
        )
        for source in (hypothesis, method, eval_plan, novelty):
            create_edge(
                graph,
                source_id=source.id,
                relation="supports",
                target_id=problem.id,
                role=source.role,
                branch_id=branch.id,
            )
            source.evidence.append("Benchmark keyword: meteorology")
        risk_edge = create_edge(
            graph,
            source_id=risk.id,
            relation="contradicts",
            target_id=method.id,
            role=risk.role,
            branch_id=branch.id,
        )
        risk_edge.resolved = True

        generic_snapshot = maturity_snapshot(graph)
        self.assertFalse(generic_snapshot.is_mature)

        hypothesis.text = "Physics-aware meteorology forecasting can improve severe-weather reliability."
        method.text = (
            "Combine satellite, radar, and reanalysis inputs with a physics-aware spatiotemporal forecasting model "
            "for meteorology."
        )
        eval_plan.text = (
            "Evaluate meteorology forecasting on reanalysis and satellite-based benchmark tasks, report RMSE, MAE, "
            "and CRPS, and run ablations on multimodal fusion and uncertainty calibration."
        )
        novelty.text = "The idea targets meteorology with uncertainty-aware multimodal forecasting rather than a generic hybrid model."
        risk.text = "Multimodal meteorology fusion may overfit rare seasonal events, so uncertainty calibration and seasonal transfer stress tests are necessary."

        graph.utility_history = [7.6]
        specific_round2_snapshot = maturity_snapshot(graph)
        self.assertFalse(specific_round2_snapshot.is_mature)

        graph.utility_history = [7.6, 7.8]
        specific_round3_snapshot = maturity_snapshot(graph)
        self.assertTrue(specific_round3_snapshot.is_mature)


if __name__ == "__main__":
    unittest.main()
