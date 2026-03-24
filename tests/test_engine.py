from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import ActionDecision
from idea_graph.engine import (
    build_seed_graphs,
    choose_round_action,
    merge_seed_graphs,
    run_experiment,
    synthesize_proposal,
    unresolved_contradiction_edges,
)
from idea_graph.models import FinalProposal, IdeaGraph, MaturitySnapshot


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
        self.assertEqual(len(graph.actions), 15)
        self.assertIn("seed_generation_error", graph.metadata)
        self.assertEqual(len(graph.metadata.get("action_errors", [])), 15)
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
        self.assertIn("PanoSUNCG (synthetic indoor)", proposal.evaluation)
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


if __name__ == "__main__":
    unittest.main()
