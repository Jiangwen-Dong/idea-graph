from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import ActionDecision
from idea_graph.action_candidates import (
    action_spec_from_action,
    enumerate_candidate_specs,
    flatten_candidate_text,
)
from idea_graph.claim_chain import select_claim_chain
from idea_graph.engine import (
    make_action,
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
    validate_action,
    unresolved_contradiction_edges,
    utility_breakdown,
)
from idea_graph.models import FinalProposal, GraphAction, IdeaGraph, MaturitySnapshot
from idea_graph.models import UtilityBreakdown
from idea_graph.relation_graph_runtime_critic import RelationGraphRuntimeConfig
from idea_graph.runtime_critic import TextCriticRuntimeConfig


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


class RuntimeCriticScoreStub:
    def score(self, texts):
        values = []
        for text in texts:
            if "kind=attach_evidence" in text:
                values.append(0.75)
            elif "kind=add_support_edge" in text:
                values.append(0.62)
            else:
                values.append(0.40)
        return values


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

    def test_action_spec_from_action_preserves_fields(self) -> None:
        action = GraphAction(
            id="A999",
            round_name="Round2",
            role="MechanismProposer",
            kind="attach_evidence",
            target_ids=["N001"],
            payload={"branch_id": "B001", "evidence": "paper://x"},
            rationale="Attach concrete prior support.",
        )

        spec = action_spec_from_action(action, candidate_source="legacy_policy")

        self.assertEqual(spec["kind"], action.kind)
        self.assertEqual(spec["target_ids"], action.target_ids)
        self.assertEqual(spec["payload"], action.payload)
        self.assertEqual(spec["candidate_source"], "legacy_policy")

    def test_enumerate_candidate_specs_default_includes_commit(self) -> None:
        graph = self._build_seed_graph()
        baseline_action = choose_round_action(graph, "Round1", "ImpactReframer")

        specs = enumerate_candidate_specs(
            graph,
            round_name="Round1",
            role="ImpactReframer",
            baseline_action=baseline_action,
        )

        self.assertTrue(
            any(
                spec.get("kind") == baseline_action.kind
                and list(spec.get("target_ids", [])) == baseline_action.target_ids
                and spec.get("candidate_source") == "legacy_policy"
                for spec in specs
            )
        )
        self.assertTrue(any(spec.get("kind") == "commit" for spec in specs))

    def test_enumerate_candidate_specs_live_safe_excludes_commit_and_validates(self) -> None:
        graph = self._build_seed_graph()
        role = "ImpactReframer"
        round_name = "Round1"
        baseline_action = choose_round_action(graph, round_name, role)

        self.assertNotIn("include_commit", inspect.signature(enumerate_candidate_specs).parameters)
        specs = enumerate_candidate_specs(
            graph,
            round_name=round_name,
            role=role,
            baseline_action=baseline_action,
        )
        live_specs = [spec for spec in specs if str(spec.get("kind", "")).strip() != "commit"]

        self.assertTrue(live_specs)
        self.assertTrue(all(spec.get("kind") != "commit" for spec in live_specs))
        for spec in live_specs:
            action = make_action(
                graph,
                round_name=round_name,
                role=role,
                kind=str(spec.get("kind", "")).strip(),
                target_ids=[str(item).strip() for item in spec.get("target_ids", []) if str(item).strip()],
                payload=dict(spec.get("payload", {}) or {}),
                rationale=str(spec.get("rationale", "")).strip(),
            )
            validate_action(graph, action)

    def test_flatten_candidate_text_uses_target_text_and_payload(self) -> None:
        graph = self._build_seed_graph()
        target = graph.active_nodes()[0]
        spec = {
            "kind": "attach_evidence",
            "target_ids": [target.id],
            "payload": {"branch_id": target.branch_id, "evidence": "grounding-snippet"},
            "rationale": "Ground this claim with concrete evidence.",
            "candidate_source": "utility_attach_evidence",
        }

        text = flatten_candidate_text(graph, spec)

        self.assertIn(target.type, text)
        self.assertIn(target.text, text)
        self.assertIn("grounding-snippet", text)
        self.assertIn("Ground this claim with concrete evidence.", text)

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

    def test_run_experiment_accepts_runtime_text_critic_reranker(self) -> None:
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            runtime_controller=RuntimeCriticScoreStub(),
            runtime_controller_metadata={
                "kind": "text_critic_rerank",
                "model_path": "memory://stub",
                "use_commit": False,
                "tau_override": 0.05,
                "config": TextCriticRuntimeConfig(tau_override=0.05, use_commit=False),
            },
            max_rounds=1,
            stop_when_mature=False,
        )

        self.assertIsNotNone(graph.final_proposal)
        self.assertEqual(graph.metadata["runtime_controller"]["kind"], "text_critic_rerank")
        self.assertTrue(graph.metadata.get("runtime_controller_log"))

    def test_runtime_text_critic_trace_is_saved_with_llm_backend(self) -> None:
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            collaboration_backend=MisalignedRepairBackend(),
            runtime_controller=RuntimeCriticScoreStub(),
            runtime_controller_metadata={
                "kind": "text_critic_rerank",
                "model_path": "memory://stub",
                "use_commit": False,
                "tau_override": 0.05,
                "config": TextCriticRuntimeConfig(tau_override=0.05, use_commit=False),
            },
            max_rounds=1,
            stop_when_mature=False,
        )

        self.assertIsNotNone(graph.final_proposal)
        controller_log = graph.metadata.get("runtime_controller_log")
        self.assertTrue(controller_log)
        self.assertEqual(controller_log[0]["round"], "Round1")
        self.assertIn("selected_source", controller_log[0])
        self.assertIn("used_heuristic_fallback", controller_log[0])

    def test_run_experiment_accepts_runtime_relation_graph_critic_reranker(self) -> None:
        def _mock_relation_graph_selector(*args, **kwargs):  # type: ignore[no-untyped-def]
            valid_candidates = list(kwargs["candidate_specs"])
            selected_id = str(valid_candidates[0]["candidate_id"])
            return SimpleNamespace(
                selected_spec={
                    "candidate_id": selected_id,
                    "critic_score": 0.91,
                },
                policy_decision=SimpleNamespace(
                    selected_candidate_id=selected_id,
                    selected_source="critic",
                    override_margin=0.05,
                    used_heuristic_fallback=False,
                ),
                scored_candidates=tuple(
                    {
                        **candidate,
                        "critic_score": 0.80 - (index * 0.1),
                    }
                    for index, candidate in enumerate(valid_candidates)
                ),
            )

        with patch(
            "idea_graph.engine.select_relation_graph_critic_candidate",
            side_effect=_mock_relation_graph_selector,
        ) as mock_selector:
            graph = run_experiment(
                topic="graph-based scientific ideation",
                literature=["paper a", "paper b", "paper c", "paper d"],
                runtime_controller=object(),
                runtime_controller_metadata={
                    "kind": "relation_graph_critic_rerank",
                    "model_dir": "memory://stub-bundle",
                    "use_commit": False,
                    "tau_override": 0.05,
                    "config": RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
                },
                max_rounds=1,
                stop_when_mature=False,
            )

        self.assertTrue(mock_selector.called)
        self.assertIsNotNone(graph.final_proposal)
        self.assertEqual(graph.metadata["runtime_controller"]["kind"], "relation_graph_critic_rerank")
        self.assertTrue(graph.metadata.get("runtime_controller_log"))

    def test_runtime_controller_trace_records_kind_and_fallback_reason_when_present(self) -> None:
        def _mock_relation_graph_selector(*args, **kwargs):  # type: ignore[no-untyped-def]
            valid_candidates = list(kwargs["candidate_specs"])
            selected_id = str(kwargs["heuristic_candidate_id"])
            return SimpleNamespace(
                selected_spec={
                    "candidate_id": selected_id,
                    "critic_score": float("-inf"),
                    "controller_fallback_reason": "unmapped_runtime_token",
                    "controller_fallback_candidate_ids": (selected_id,),
                },
                policy_decision=SimpleNamespace(
                    selected_candidate_id=selected_id,
                    selected_source="heuristic",
                    override_margin=float("-inf"),
                    used_heuristic_fallback=True,
                ),
                scored_candidates=tuple(
                    {
                        **candidate,
                        "critic_score": float("-inf"),
                        "controller_fallback_reason": "unmapped_runtime_token",
                    }
                    for candidate in valid_candidates
                ),
            )

        with patch(
            "idea_graph.engine.select_relation_graph_critic_candidate",
            side_effect=_mock_relation_graph_selector,
        ):
            graph = run_experiment(
                topic="graph-based scientific ideation",
                literature=["paper a", "paper b", "paper c", "paper d"],
                runtime_controller=object(),
                runtime_controller_metadata={
                    "kind": "relation_graph_critic_rerank",
                    "model_dir": "memory://stub-bundle",
                    "use_commit": False,
                    "tau_override": 0.05,
                    "config": RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
                },
                max_rounds=1,
                stop_when_mature=False,
            )

        controller_log = graph.metadata.get("runtime_controller_log")
        self.assertTrue(controller_log)
        self.assertEqual(controller_log[0]["controller_kind"], "relation_graph_critic_rerank")
        self.assertEqual(controller_log[0]["selected_fallback_reason"], "unmapped_runtime_token")
        self.assertEqual(
            controller_log[0]["selected_fallback_candidate_ids"],
            [controller_log[0]["selected_candidate_id"]],
        )

    def test_runtime_controller_trace_records_candidate_snapshots(self) -> None:
        def _mock_relation_graph_selector(*args, **kwargs):  # type: ignore[no-untyped-def]
            valid_candidates = [dict(candidate) for candidate in kwargs["candidate_specs"]]
            heuristic_id = str(kwargs["heuristic_candidate_id"])
            selected = next(
                candidate
                for candidate in valid_candidates
                if str(candidate.get("candidate_id")) != heuristic_id
            )
            return SimpleNamespace(
                selected_spec={
                    **selected,
                    "critic_score": 0.91,
                    "predicted_gain": 0.42,
                },
                policy_decision=SimpleNamespace(
                    selected_candidate_id=str(selected["candidate_id"]),
                    selected_source="critic",
                    override_margin=0.12,
                    used_heuristic_fallback=False,
                ),
                scored_candidates=tuple(
                    {
                        **candidate,
                        "critic_score": 0.80 - (index * 0.1),
                        "predicted_gain": 0.30 - (index * 0.05),
                    }
                    for index, candidate in enumerate(valid_candidates)
                ),
            )

        with patch(
            "idea_graph.engine.select_relation_graph_critic_candidate",
            side_effect=_mock_relation_graph_selector,
        ):
            graph = run_experiment(
                topic="graph-based scientific ideation",
                literature=["paper a", "paper b", "paper c", "paper d"],
                runtime_controller=object(),
                runtime_controller_metadata={
                    "kind": "relation_graph_critic_rerank",
                    "model_dir": "memory://stub-bundle",
                    "use_commit": False,
                    "tau_override": 0.05,
                    "config": RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
                },
                max_rounds=1,
                stop_when_mature=False,
            )

        controller_log = graph.metadata.get("runtime_controller_log")
        self.assertTrue(controller_log)
        first = controller_log[0]
        self.assertIn("heuristic_candidate", first)
        self.assertIn("selected_candidate", first)
        self.assertIn("target_ids", first["heuristic_candidate"])
        self.assertIn("payload", first["heuristic_candidate"])
        self.assertIn("candidate_source", first["heuristic_candidate"])
        self.assertIn("target_ids", first["selected_candidate"])
        self.assertIn("payload", first["selected_candidate"])
        self.assertIn("candidate_source", first["selected_candidate"])
        self.assertTrue(first["top_scored_candidates"])
        self.assertIn("target_ids", first["top_scored_candidates"][0])
        self.assertIn("payload", first["top_scored_candidates"][0])
        self.assertIn("candidate_source", first["top_scored_candidates"][0])

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

    def test_run_experiment_stores_prompt_safe_grounding_in_benchmark_mode(self) -> None:
        graph = run_experiment(
            topic="The topic of this paper is improving GUI grounding and OOD generalization for GUI agents.",
            literature=["SeeClick", "OSWorld"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_mode": True,
                "target_paper": "Hidden Target GUI Model",
                "method_summary": "Secret gold method.",
                "raw_record": {
                    "summary": {
                        "method": {
                            "datasets": "HiddenSet",
                            "metrics": "HiddenMetric",
                        }
                    }
                },
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "SeeClick",
                            "method": "Use screenshot-grounded interaction instead of structured text for GUI agents.",
                            "evaluation": "Evaluate on OSWorld and report success rate and error rate.",
                        }
                    ],
                },
                "generation_safe_metadata": {
                    "benchmark": "AI_Idea_Bench_2025",
                    "benchmark_mode": True,
                    "paper_grounding": {
                        "reference_paper_snippets": [
                            {
                                "resolved_title": "SeeClick",
                                "method": "Use screenshot-grounded interaction instead of structured text for GUI agents.",
                                "evaluation": "Evaluate on OSWorld and report success rate and error rate.",
                            }
                        ],
                    },
                },
            },
            max_rounds=1,
            stop_when_mature=False,
        )

        grounding = graph.metadata["literature_grounding"]
        self.assertEqual(grounding["target_paper"], "")
        self.assertNotIn("HiddenSet", grounding["dataset_items"])
        self.assertNotIn("HiddenMetric", grounding["metric_items"])
        self.assertNotIn("Hidden Target", grounding["existing_methods_summary"])
        self.assertIn("OSWorld", " ".join(grounding["dataset_items"] + grounding["metric_items"]))

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

    def test_run_experiment_accepts_parallel_runtime_protocol(self) -> None:
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            metadata={"runtime_protocol": "parallel_graph_v2"},
            max_rounds=1,
            stop_when_mature=False,
        )

        self.assertEqual(graph.metadata.get("runtime_protocol"), "parallel_graph_v2")
        self.assertEqual(graph.metadata.get("executed_round_count"), 1)

    def test_run_experiment_records_default_runtime_protocol(self) -> None:
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            max_rounds=1,
            stop_when_mature=False,
        )

        self.assertEqual(graph.metadata.get("runtime_protocol"), "sequential_v1")

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

    def test_claim_chain_prefers_concrete_method_over_generic_hypothesis_when_available(self) -> None:
        graph = IdeaGraph(
            topic="GUI grounding for out-of-distribution GUI agents.",
            literature=["SeeClick"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "generation_safe_metadata": {
                    "benchmark": "AI_Idea_Bench_2025",
                    "paper_grounding": {
                        "reference_paper_snippets": [
                            {
                                "resolved_title": "SeeClick",
                                "method": "Use screenshot-grounded interaction instead of structured text.",
                                "evaluation": "Evaluate on OSWorld and report success rate and error rate.",
                            }
                        ]
                    },
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
            text="GUI agents fail under layout and visual distribution shift.",
            role="ImpactReframer",
            branch_id=problem_branch.id,
            confidence=0.85,
        )
        gap = create_node(
            graph,
            node_type="NoveltyClaim",
            text="Visible references do not explicitly model cross-platform uncertainty in GUI grounding.",
            role="NoveltyExaminer",
            branch_id=gap_branch.id,
            confidence=0.82,
        )
        hypothesis = create_node(
            graph,
            node_type="Hypothesis",
            text="A better GUI grounding method should improve transfer.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.9,
        )
        generic_method = create_node(
            graph,
            node_type="Method",
            text="Use a generic multimodal architecture for the task.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.88,
        )
        concrete_method = create_node(
            graph,
            node_type="Method",
            text=(
                "Use uncertainty-aware screenshot-grounded element localization before action decoding to handle "
                "cross-platform GUI shift."
            ),
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.86,
        )
        evaluation = create_node(
            graph,
            node_type="EvalPlan",
            text="Evaluate on OSWorld and report success rate and error rate with cross-platform stress tests.",
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            confidence=0.84,
        )
        risk = create_node(
            graph,
            node_type="Risk",
            text="The grounding model may still fail on unseen interface widgets.",
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
            confidence=0.75,
        )

        for source, target in (
            (gap, problem),
            (hypothesis, gap),
            (generic_method, hypothesis),
            (concrete_method, hypothesis),
            (evaluation, concrete_method),
        ):
            create_edge(
                graph,
                source_id=source.id,
                relation="supports",
                target_id=target.id,
                role=source.role,
                branch_id=source.branch_id,
            )
        create_edge(
            graph,
            source_id=risk.id,
            relation="contradicts",
            target_id=concrete_method.id,
            role=risk.role,
            branch_id=risk_branch.id,
        )

        chain = select_claim_chain(graph)

        assert chain is not None
        self.assertEqual(chain["slots"]["mechanism"], concrete_method.id)

    def test_claim_chain_prefers_evaluation_node_with_visible_benchmark_anchors(self) -> None:
        graph = IdeaGraph(
            topic="GUI grounding for out-of-distribution GUI agents.",
            literature=["SeeClick"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "generation_safe_metadata": {
                    "benchmark": "AI_Idea_Bench_2025",
                    "paper_grounding": {
                        "reference_paper_snippets": [
                            {
                                "resolved_title": "SeeClick",
                                "method": "Use screenshot-grounded interaction instead of structured text.",
                                "evaluation": "Evaluate on OSWorld and report success rate and error rate.",
                            }
                        ]
                    },
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
            text="GUI agents fail under layout and visual distribution shift.",
            role="ImpactReframer",
            branch_id=problem_branch.id,
            confidence=0.85,
        )
        gap = create_node(
            graph,
            node_type="NoveltyClaim",
            text="Visible references do not explicitly model cross-platform uncertainty in GUI grounding.",
            role="NoveltyExaminer",
            branch_id=gap_branch.id,
            confidence=0.82,
        )
        method = create_node(
            graph,
            node_type="Method",
            text="Use uncertainty-aware screenshot-grounded element localization before action decoding.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.86,
        )
        generic_eval = create_node(
            graph,
            node_type="EvalPlan",
            text="Evaluate on benchmark datasets and report task-relevant metrics.",
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            confidence=0.9,
        )
        grounded_eval = create_node(
            graph,
            node_type="EvalPlan",
            text="Evaluate on OSWorld and report success rate and error rate with cross-platform stress tests.",
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            confidence=0.82,
        )
        risk = create_node(
            graph,
            node_type="Risk",
            text="The grounding model may still fail on unseen interface widgets.",
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
            confidence=0.75,
        )

        for source, target in (
            (gap, problem),
            (method, gap),
            (generic_eval, method),
            (grounded_eval, method),
        ):
            create_edge(
                graph,
                source_id=source.id,
                relation="supports",
                target_id=target.id,
                role=source.role,
                branch_id=source.branch_id,
            )
        create_edge(
            graph,
            source_id=risk.id,
            relation="contradicts",
            target_id=method.id,
            role=risk.role,
            branch_id=risk_branch.id,
        )

        chain = select_claim_chain(graph)

        assert chain is not None
        self.assertEqual(chain["slots"]["evaluation"], grounded_eval.id)

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

    def test_utility_prefers_specific_method_and_evaluation_over_generic_chain(self) -> None:
        def build_chain(*, method_text: str, evaluation_text: str) -> IdeaGraph:
            graph = IdeaGraph(
                topic="GUI grounding for out-of-distribution GUI agents",
                literature=["SeeClick"],
                metadata={
                    "benchmark_mode": True,
                    "generation_safe_metadata": {
                        "benchmark": "AI_Idea_Bench_2025",
                        "paper_grounding": {
                            "reference_paper_snippets": [
                                {
                                    "resolved_title": "SeeClick",
                                    "method": "Use screenshot-grounded interaction instead of structured text.",
                                    "evaluation": "Evaluate on OSWorld and report success rate and error rate.",
                                }
                            ]
                        },
                    },
                },
            )
            branch = create_branch(graph, "MechanismProposer")
            problem = create_node(
                graph,
                node_type="Problem",
                text="GUI agents fail to generalize when interface layouts and visual styles shift.",
                role="ImpactReframer",
                branch_id=branch.id,
                confidence=0.82,
            )
            hypothesis = create_node(
                graph,
                node_type="Hypothesis",
                text="A grounded visual interaction model can improve GUI-agent generalization.",
                role="MechanismProposer",
                branch_id=branch.id,
                confidence=0.82,
            )
            method = create_node(
                graph,
                node_type="Method",
                text=method_text,
                role="MechanismProposer",
                branch_id=branch.id,
                confidence=0.82,
            )
            evaluation = create_node(
                graph,
                node_type="EvalPlan",
                text=evaluation_text,
                role="EvaluationDesigner",
                branch_id=branch.id,
                confidence=0.82,
            )
            novelty = create_node(
                graph,
                node_type="NoveltyClaim",
                text="The idea targets cross-platform GUI grounding rather than only supervised click prediction.",
                role="NoveltyExaminer",
                branch_id=branch.id,
                confidence=0.82,
            )
            for source, target in (
                (hypothesis, problem),
                (method, hypothesis),
                (evaluation, method),
                (novelty, hypothesis),
            ):
                create_edge(
                    graph,
                    source_id=source.id,
                    relation="supports",
                    target_id=target.id,
                    role=source.role,
                    branch_id=branch.id,
                )
            return graph

        generic_breakdown = utility_breakdown(
            build_chain(
                method_text="Use a better agent architecture with a generic multimodal module.",
                evaluation_text="Evaluate on benchmark datasets and report task-relevant metrics.",
            )
        )
        specific_breakdown = utility_breakdown(
            build_chain(
                method_text=(
                    "Train an uncertainty-aware screenshot-grounded action planner that separates visual element "
                    "localization from command grounding and tests cross-platform GUI shifts."
                ),
                evaluation_text=(
                    "Evaluate on OSWorld and held-out GUI layouts, report success rate and error rate, and ablate "
                    "visual grounding and uncertainty calibration."
                ),
            )
        )

        self.assertGreater(specific_breakdown.total, generic_breakdown.total)

    def test_utility_penalizes_reference_copy_collapse(self) -> None:
        def build_chain(method_text: str) -> IdeaGraph:
            graph = IdeaGraph(
                topic="GUI grounding for out-of-distribution GUI agents",
                literature=["SeeClick"],
                metadata={
                    "benchmark_mode": True,
                    "generation_safe_metadata": {
                        "benchmark": "AI_Idea_Bench_2025",
                        "paper_grounding": {
                            "reference_paper_snippets": [
                                {
                                    "resolved_title": "SeeClick",
                                    "method": "Use screenshot-grounded interaction instead of structured text.",
                                    "evaluation": "Evaluate on OSWorld and report success rate.",
                                }
                            ]
                        },
                    },
                },
            )
            branch = create_branch(graph, "MechanismProposer")
            problem = create_node(
                graph,
                node_type="Problem",
                text="GUI agents fail when visual interfaces shift across platforms.",
                role="ImpactReframer",
                branch_id=branch.id,
                confidence=0.84,
            )
            hypothesis = create_node(
                graph,
                node_type="Hypothesis",
                text="A GUI grounding method with explicit uncertainty can improve transfer.",
                role="MechanismProposer",
                branch_id=branch.id,
                confidence=0.84,
            )
            method = create_node(
                graph,
                node_type="Method",
                text=method_text,
                role="MechanismProposer",
                branch_id=branch.id,
                confidence=0.84,
            )
            evaluation = create_node(
                graph,
                node_type="EvalPlan",
                text="Evaluate on OSWorld and report success rate with cross-platform stress tests.",
                role="EvaluationDesigner",
                branch_id=branch.id,
                confidence=0.84,
            )
            novelty = create_node(
                graph,
                node_type="NoveltyClaim",
                text="The method adds uncertainty-aware cross-platform transfer beyond the visible reference.",
                role="NoveltyExaminer",
                branch_id=branch.id,
                confidence=0.84,
            )
            for source, target in (
                (hypothesis, problem),
                (method, hypothesis),
                (evaluation, method),
                (novelty, hypothesis),
            ):
                create_edge(
                    graph,
                    source_id=source.id,
                    relation="supports",
                    target_id=target.id,
                    role=source.role,
                    branch_id=branch.id,
                )
            return graph

        copied_breakdown = utility_breakdown(
            build_chain("Use screenshot-grounded interaction instead of structured text.")
        )
        original_breakdown = utility_breakdown(
            build_chain(
                "Add uncertainty-aware cross-platform element localization before command grounding and reject "
                "ambiguous GUI targets during transfer."
            )
        )

        self.assertLess(copied_breakdown.total, original_breakdown.total)

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

    def test_benchmark_mode_maturity_requires_benchmark_specific_chain(self) -> None:
        graph = IdeaGraph(
            topic="GUI grounding for out-of-distribution GUI agents.",
            literature=["SeeClick"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "generation_safe_metadata": {
                    "benchmark": "AI_Idea_Bench_2025",
                    "paper_grounding": {
                        "reference_paper_snippets": [
                            {
                                "resolved_title": "SeeClick",
                                "method": "Use screenshot-grounded interaction instead of structured text.",
                                "evaluation": "Evaluate on OSWorld and report success rate and error rate.",
                            }
                        ]
                    },
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
            text="GUI agents fail under layout and visual distribution shift.",
            role="ImpactReframer",
            branch_id=problem_branch.id,
            confidence=0.85,
        )
        gap = create_node(
            graph,
            node_type="NoveltyClaim",
            text="Visible references do not explicitly model cross-platform uncertainty in GUI grounding.",
            role="NoveltyExaminer",
            branch_id=gap_branch.id,
            confidence=0.82,
            evidence=["Reference methods still fail under distribution shift."],
        )
        hypothesis = create_node(
            graph,
            node_type="Hypothesis",
            text="A stronger GUI grounding method can improve transfer.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.84,
        )
        method = create_node(
            graph,
            node_type="Method",
            text="Use a better multimodal architecture for the task.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.84,
        )
        evaluation = create_node(
            graph,
            node_type="EvalPlan",
            text="Evaluate on benchmark datasets and report task-relevant metrics.",
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            confidence=0.84,
            evidence=["Run a standard benchmark comparison."],
        )
        risk = create_node(
            graph,
            node_type="Risk",
            text="The model may still fail on unseen interface widgets.",
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
            confidence=0.75,
        )

        for source, target in (
            (gap, problem),
            (hypothesis, gap),
            (method, hypothesis),
            (evaluation, method),
        ):
            create_edge(
                graph,
                source_id=source.id,
                relation="supports",
                target_id=target.id,
                role=source.role,
                branch_id=source.branch_id,
            )

        graph.utility_history = [6.8, 6.8]
        generic_snapshot = maturity_snapshot(graph)
        self.assertFalse(generic_snapshot.is_mature)

        method.text = (
            "Use uncertainty-aware screenshot-grounded element localization before action decoding to handle "
            "cross-platform GUI shift."
        )
        evaluation.text = (
            "Evaluate on OSWorld and report success rate and error rate with cross-platform stress tests and "
            "ablations on uncertainty calibration."
        )

        graph.utility_history = [8.0, 8.1]
        specific_snapshot = maturity_snapshot(graph)
        self.assertTrue(specific_snapshot.is_mature)


if __name__ == "__main__":
    unittest.main()
