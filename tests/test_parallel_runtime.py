import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from idea_graph.agent_backend import ActionDecision
from idea_graph.engine import build_seed_graphs, merge_seed_graphs
from idea_graph.models import Branch, Edge, IdeaGraph, Node
from idea_graph.models import (
    ParallelCommitCheckRecord,
    ParallelEditPatchRecord,
    ParallelRoleDecisionRecord,
    ParallelRoleRoundResult,
)
from idea_graph.parallel_runtime import _runtime_commit_check, execute_parallel_role_round
from idea_graph.relation_graph_runtime_critic import RelationGraphRuntimeConfig
from idea_graph.signal_heuristic_control import SignalHeuristicController


def test_parallel_round_result_tracks_decisions_patches_and_commit_check() -> None:
    result = ParallelRoleRoundResult(
        round_name="Round1",
        active_roles=("MechanismProposer", "EvaluationDesigner"),
        skipped_roles=("EvaluationDesigner",),
        selected_role_decisions=(
            ParallelRoleDecisionRecord(
                role="MechanismProposer",
                kind="freeze_branch",
                target_ids=(),
                payload={"branch_id": "B001"},
                rationale="freeze",
            ),
        ),
        edit_patches=(
            ParallelEditPatchRecord(
                role="MechanismProposer",
                kind="freeze_branch",
                target_ids=(),
                payload={"branch_id": "B001"},
                is_empty=False,
            ),
        ),
        materialized_graph_actions=(),
        post_round_commit=ParallelCommitCheckRecord(
            round_name="Round1",
            state_kind="parallel_post_round",
            should_commit=False,
            source="maturity_snapshot",
        ),
    )

    assert result.round_name == "Round1"
    assert result.skipped_roles == ("EvaluationDesigner",)
    assert result.selected_role_decisions[0].role == "MechanismProposer"
    assert result.edit_patches[0].is_empty is False
    assert result.post_round_commit.should_commit is False


def test_parallel_runtime_uses_signal_heuristic_control_for_edits_and_commit_logging() -> None:
    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round3",
        collaboration_backend=None,
        runtime_controller=SignalHeuristicController(),
        runtime_controller_metadata={
            "kind": "signal_heuristic_control",
            "config": RelationGraphRuntimeConfig(
                use_edit=True,
                use_commit=True,
                min_commit_round=2,
                gamma_commit=0.55,
            ),
        },
        progress_callback=None,
    )

    assert result.selected_role_decisions
    assert result.post_round_commit.controller_kind == "signal_heuristic_control"
    assert "grounding" in result.post_round_commit.graph_signals
    assert "maturity" in result.post_round_commit.graph_signals


def test_parallel_runtime_avoids_duplicate_single_target_repairs_within_one_round() -> None:
    class DuplicateRepairPreferringController:
        def choose(self, *, round_name, role, candidate_specs, graph):  # type: ignore[no-untyped-def]
            del round_name, role, graph
            scored = []
            for index, candidate in enumerate(candidate_specs):
                kind = str(candidate.get("kind", "")).strip()
                score = 0.9 if kind == "propose_repair" else 0.1
                scored.append(
                    {
                        **dict(candidate),
                        "candidate_id": str(candidate.get("candidate_id", f"c{index}")),
                        "critic_score": score,
                        "critic_base_score": score,
                        "critic_score_calibrated": score,
                        "critic_calibration_bias": 0.0,
                    }
                )
            selected = max(
                scored,
                key=lambda row: (float(row.get("critic_score", 0.0)), str(row.get("candidate_id", ""))),
            )
            return {
                "candidate_id": str(selected["candidate_id"]),
                "selected_source": "signal_heuristic_control",
                "scored_candidates": tuple(scored),
            }

        def score_commit_graph(self, graph, *, snapshot=None):  # type: ignore[no-untyped-def]
            del graph, snapshot
            return 0.0

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round2",
        collaboration_backend=None,
        runtime_controller=DuplicateRepairPreferringController(),
        runtime_controller_metadata={
            "kind": "signal_heuristic_control",
            "config": RelationGraphRuntimeConfig(
                use_edit=True,
                use_commit=False,
                min_commit_round=2,
                gamma_commit=0.55,
            ),
        },
        progress_callback=None,
    )

    repair_decisions = [
        decision for decision in result.selected_role_decisions if decision.kind == "propose_repair"
    ]

    assert len(repair_decisions) >= 2
    assert len({decision.target_ids[0] for decision in repair_decisions}) == len(repair_decisions)


def test_parallel_role_round_uses_frozen_snapshot_for_all_roles() -> None:
    observed_node_counts = []

    class FrozenSnapshotBackend:
        name = "frozen-snapshot-test"

        def generate_seed(self, graph, role):
            raise RuntimeError("seed generation not used")

        def choose_action(self, graph, round_name, role):
            observed_node_counts.append((role, len(graph.active_nodes())))
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": next(branch.id for branch in graph.branches.values() if branch.role == role)},
                rationale="freeze for snapshot test",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=FrozenSnapshotBackend(),
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    counts = {count for _, count in observed_node_counts}
    assert len(counts) == 1


def test_parallel_role_round_converts_invalid_llm_action_to_logged_skip() -> None:
    class PartiallyInvalidBackend:
        name = "partially-invalid-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("seed generation not used")

        def choose_action(self, graph, round_name, role):
            if role == "NoveltyExaminer":
                raise ValueError("Action referenced unknown node id 'paper-003'.")
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": next(branch.id for branch in graph.branches.values() if branch.role == role)},
                rationale="valid freeze",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=PartiallyInvalidBackend(),
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    fallback_decision = next(
        decision for decision in result.selected_role_decisions if decision.role == "NoveltyExaminer"
    )
    assert fallback_decision.kind == "skip"
    assert "NoveltyExaminer" in result.skipped_roles
    assert any(
        trace.get("stage") == "Round1_action"
        and trace.get("role") == "NoveltyExaminer"
        and trace.get("fallback_action") == "skip"
        and trace.get("error_type") == "ValueError"
        for trace in graph.metadata.get("agent_traces", [])
    )


def test_parallel_role_round_converts_runtime_llm_failure_to_logged_skip() -> None:
    class PartiallyTimeoutBackend:
        name = "partially-timeout-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("seed generation not used")

        def choose_action(self, graph, round_name, role):
            if role == "NoveltyExaminer":
                raise RuntimeError("OpenAI-compatible request timed out after 3 attempts.")
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": next(branch.id for branch in graph.branches.values() if branch.role == role)},
                rationale="valid freeze",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=PartiallyTimeoutBackend(),
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    fallback_decision = next(
        decision for decision in result.selected_role_decisions if decision.role == "NoveltyExaminer"
    )
    assert fallback_decision.kind == "skip"
    assert "NoveltyExaminer" in result.skipped_roles
    assert any(
        trace.get("stage") == "Round1_action"
        and trace.get("role") == "NoveltyExaminer"
        and trace.get("fallback_action") == "skip"
        and trace.get("error_type") == "RuntimeError"
        for trace in graph.metadata.get("agent_traces", [])
    )


def test_parallel_round_materializes_non_skip_actions_in_role_order() -> None:
    class OrderedBackend:
        name = "ordered-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("not used")

        def choose_action(self, graph, round_name, role):
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            if role == "EvaluationDesigner":
                return ActionDecision(
                    kind="skip",
                    target_ids=[],
                    payload={"branch_id": branch_id},
                    rationale="skip",
                )
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale=role,
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=OrderedBackend(),
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    assert "EvaluationDesigner" in result.skipped_roles
    assert [decision.role for decision in result.selected_role_decisions] == sorted(
        decision.role for decision in result.selected_role_decisions
    )
    assert all(patch.is_empty for patch in result.edit_patches if patch.role == "EvaluationDesigner")
    assert [action.role for action in result.materialized_graph_actions] == sorted(
        action.role for action in result.materialized_graph_actions
    )
    assert result.post_round_commit.round_name == "Round1"


def test_parallel_runtime_logs_role_action_traces_for_usage_accounting() -> None:
    class TraceBackend:
        name = "trace-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("not used")

        def choose_action(self, graph, round_name, role):
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale=f"{role} action",
                trace={
                    "raw_response": {
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 7,
                            "total_tokens": 18,
                        }
                    }
                },
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=TraceBackend(),
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    traces = graph.metadata.get("agent_traces", [])
    assert isinstance(traces, list)
    round_action_traces = [trace for trace in traces if trace.get("stage") == "Round1_action"]
    assert len(round_action_traces) == len(result.active_roles)
    assert {trace.get("role") for trace in round_action_traces} == set(result.active_roles)
    assert sum(
        int(trace.get("raw_response", {}).get("usage", {}).get("total_tokens", 0))
        for trace in round_action_traces
    ) == 18 * len(result.active_roles)


def test_parallel_runtime_uses_controller_for_role_local_edit_selection() -> None:
    class FreezeBackend:
        name = "freeze-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("not used")

        def choose_action(self, graph, round_name, role):
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale="backend selected freeze",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    class SkipPreferringController:
        def build_runtime_batch(self, *, graph, candidate_specs, use_commit):
            self.last_candidate_specs = [dict(spec) for spec in candidate_specs]
            return SimpleNamespace(
                batch=object(),
                fallback_row_mask=torch.zeros(len(self.last_candidate_specs), dtype=torch.bool),
                diagnostics=(),
            )

        def runtime_token_status(self, runtime_batch):
            return {"ok": True}

        def score_runtime_batch(self, batch):
            return [
                10.0 if str(spec.get("kind")) == "skip" else 0.0
                for spec in self.last_candidate_specs
            ]

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=FreezeBackend(),
        runtime_controller=SkipPreferringController(),
        runtime_controller_metadata={
            "kind": "relation_graph_critic_rerank",
            "config": RelationGraphRuntimeConfig(tau_override=0.0, use_commit=False),
        },
        progress_callback=None,
    )

    assert result.active_roles
    assert set(result.skipped_roles) == set(result.active_roles)
    assert all(decision.kind == "skip" for decision in result.selected_role_decisions)
    assert all(row["label_source"] == "parallel_runtime_controller_v1" for row in result.edit_rows)
    assert graph.metadata["runtime_controller_log"]
    assert graph.metadata["runtime_controller_log"][0]["controller_kind"] == "relation_graph_critic_rerank"


def test_parallel_runtime_passes_utility_feedback_to_runtime_controller(monkeypatch) -> None:
    captured_state_features = []

    class FreezeBackend:
        name = "freeze-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("not used")

        def choose_action(self, graph, round_name, role):
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale="backend selected freeze",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    def fake_selector(
        graph,
        *,
        round_name,
        role,
        state_features,
        candidate_specs,
        heuristic_candidate_id,
        runtime_bundle,
        config,
    ):
        del graph, round_name, role, runtime_bundle, config
        captured_state_features.append(dict(state_features))
        selected = next(
            candidate
            for candidate in candidate_specs
            if str(candidate.get("candidate_id")) == str(heuristic_candidate_id)
        )
        return SimpleNamespace(
            selected_spec={**selected, "critic_score": 0.5},
            policy_decision=SimpleNamespace(
                selected_candidate_id=str(heuristic_candidate_id),
                selected_source="heuristic",
                override_margin=0.0,
                used_heuristic_fallback=True,
            ),
            scored_candidates=tuple({**candidate, "critic_score": 0.5} for candidate in candidate_specs),
        )

    monkeypatch.setattr(
        "idea_graph.parallel_runtime.select_relation_graph_critic_candidate",
        fake_selector,
    )

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=FreezeBackend(),
        runtime_controller=object(),
        runtime_controller_metadata={
            "kind": "relation_graph_critic_rerank",
            "config": RelationGraphRuntimeConfig(tau_override=0.0, use_commit=False),
        },
        progress_callback=None,
    )

    assert captured_state_features
    first_state = captured_state_features[0]
    assert "utility" in first_state
    assert "utility_stable" in first_state
    assert "utility_breakdown" in first_state
    assert "support" in first_state["utility_breakdown"]
    assert "evidence" in first_state["utility_breakdown"]
    assert "coherence" in first_state["utility_breakdown"]


def test_parallel_runtime_uses_nonlearned_policy_controller_for_role_local_edit_selection() -> None:
    class FreezeBackend:
        name = "freeze-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("not used")

        def choose_action(self, graph, round_name, role):
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale="backend selected freeze",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    class SkipPolicyController:
        def __init__(self):
            self.calls = []

        def choose(self, *, round_name, role, candidate_specs):
            self.calls.append((round_name, role, [dict(spec) for spec in candidate_specs]))
            return next(spec for spec in candidate_specs if str(spec.get("kind")) == "skip")

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)
    controller = SkipPolicyController()

    result = execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=FreezeBackend(),
        runtime_controller=controller,
        runtime_controller_metadata={
            "kind": "fixed_control",
            "config": SimpleNamespace(use_edit=True, use_commit=False),
        },
        progress_callback=None,
    )

    assert controller.calls
    assert set(result.skipped_roles) == set(result.active_roles)
    assert all(decision.kind == "skip" for decision in result.selected_role_decisions)
    assert all(row["label_source"] == "parallel_runtime_controller_v1" for row in result.edit_rows)
    assert result.post_round_commit.source == "maturity_snapshot"
    assert graph.metadata["runtime_controller_log"][0]["controller_kind"] == "fixed_control"
    assert graph.metadata["runtime_controller_log"][0]["selected_source"] == "fixed_control_policy"


def test_parallel_runtime_no_edit_uses_heuristic_edits_but_learned_commit() -> None:
    class FreezeBackend:
        name = "freeze-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("not used")

        def choose_action(self, graph, round_name, role):
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale="backend selected freeze",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    class SkipEditCommitController:
        def build_runtime_batch(self, *, graph, candidate_specs, use_commit):
            self.last_candidate_specs = [dict(spec) for spec in candidate_specs]
            return SimpleNamespace(
                batch=object(),
                fallback_row_mask=torch.zeros(len(self.last_candidate_specs), dtype=torch.bool),
                diagnostics=(),
            )

        def runtime_token_status(self, runtime_batch):
            return {"ok": True}

        def score_runtime_batch(self, batch):
            return [
                10.0 if str(spec.get("kind")) == "skip" else 0.0
                for spec in self.last_candidate_specs
            ]

        def score_commit_graph(self, graph, *, snapshot=None):
            return 0.91

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round2",
        collaboration_backend=FreezeBackend(),
        runtime_controller=SkipEditCommitController(),
        runtime_controller_metadata={
            "kind": "relation_graph_two_head_critic",
            "config": RelationGraphRuntimeConfig(
                use_edit=False,
                use_commit=True,
                min_commit_round=2,
                gamma_commit=0.80,
            ),
        },
        progress_callback=None,
    )

    assert result.active_roles
    assert result.post_round_commit.source == "runtime_controller_commit"
    assert result.post_round_commit.should_commit is True
    assert any(decision.kind != "skip" for decision in result.selected_role_decisions)
    assert result.materialized_graph_actions
    assert not graph.metadata.get("runtime_controller_log")


def test_parallel_runtime_uses_two_head_commit_score_after_round() -> None:
    class FreezeBackend:
        name = "freeze-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("not used")

        def choose_action(self, graph, round_name, role):
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale="backend selected freeze",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    class CommitPreferringController:
        def build_runtime_batch(self, *, graph, candidate_specs, use_commit):
            self.last_candidate_specs = [dict(spec) for spec in candidate_specs]
            return SimpleNamespace(
                batch=object(),
                fallback_row_mask=torch.zeros(len(self.last_candidate_specs), dtype=torch.bool),
                diagnostics=(),
            )

        def runtime_token_status(self, runtime_batch):
            return {"ok": True}

        def score_runtime_batch(self, batch):
            return [0.0 for _spec in self.last_candidate_specs]

        def score_commit_graph(self, graph, *, snapshot=None):
            return 0.91

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round2",
        collaboration_backend=FreezeBackend(),
        runtime_controller=CommitPreferringController(),
        runtime_controller_metadata={
            "kind": "relation_graph_two_head_critic",
            "config": RelationGraphRuntimeConfig(
                min_commit_round=2,
                gamma_commit=0.80,
                use_commit=True,
            ),
        },
        progress_callback=None,
    )

    assert result.post_round_commit.should_commit is True
    assert result.post_round_commit.source == "runtime_controller_commit"
    assert result.post_round_commit.controller_kind == "relation_graph_two_head_critic"
    assert result.post_round_commit.commit_probability == 0.91
    assert result.post_round_commit_rows[0]["commit_supervision"]["source"] == "runtime_controller_commit"
    assert result.post_round_commit_rows[0]["commit_probability"] == 0.91


def test_parallel_runtime_commit_check_uses_round_threshold_and_guards() -> None:
    class FreezeBackend:
        name = "freeze-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("not used")

        def choose_action(self, graph, round_name, role):
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale="backend selected freeze",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    class ConfidentCommitController:
        def score_commit_graph(self, graph, *, snapshot=None):
            return 0.91

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round3",
        collaboration_backend=FreezeBackend(),
        runtime_controller=ConfidentCommitController(),
        runtime_controller_metadata={
            "kind": "relation_graph_two_head_critic",
            "config": RelationGraphRuntimeConfig(
                use_edit=False,
                use_commit=True,
                min_commit_round=2,
                gamma_commit=0.60,
                gamma_commit_by_round={3: 0.90},
                guard_commit_support_threshold=0.99,
            ),
        },
        progress_callback=None,
    )

    assert result.post_round_commit.should_commit is False
    assert result.post_round_commit.source == "runtime_controller_continue"
    assert result.post_round_commit.commit_threshold == 0.90
    assert result.post_round_commit.commit_guard_reason == "support_below_commit_guard"
    assert result.post_round_commit_rows[0]["commit_guard_reason"] == "support_below_commit_guard"


def _build_graph_signal_commit_graph(*, healthy: bool) -> IdeaGraph:
    graph = IdeaGraph(topic="topic", literature=["paper a"])
    graph.branches["B001"] = Branch(id="B001", role="MechanismProposer")
    graph.nodes["N001"] = Node(
        id="N001",
        type="Problem",
        text="Problem node.",
        role="ImpactReframer",
        branch_id="B001",
        confidence=0.8,
        evidence=["problem-evidence"] if healthy else [],
    )
    graph.nodes["N002"] = Node(
        id="N002",
        type="Hypothesis",
        text="Hypothesis node.",
        role="MechanismProposer",
        branch_id="B001",
        confidence=0.8,
        evidence=["hypothesis-evidence"] if healthy else [],
    )
    graph.nodes["N003"] = Node(
        id="N003",
        type="Method",
        text="Method node.",
        role="MechanismProposer",
        branch_id="B001",
        confidence=0.8,
        evidence=["method-evidence"] if healthy else [],
    )
    graph.nodes["N004"] = Node(
        id="N004",
        type="EvalPlan",
        text="Eval plan node.",
        role="EvaluationDesigner",
        branch_id="B001",
        confidence=0.8,
        evidence=["eval-evidence"] if healthy else [],
    )
    graph.nodes["N005"] = Node(
        id="N005",
        type="NoveltyClaim",
        text="Novelty claim node.",
        role="NoveltyExaminer",
        branch_id="B001",
        confidence=0.8,
        evidence=["novelty-evidence"] if healthy else [],
    )
    if healthy:
        graph.edges.extend(
            [
                Edge(
                    id="E001",
                    source_id="N002",
                    relation="supports",
                    target_id="N001",
                    role="MechanismProposer",
                    branch_id="B001",
                ),
                Edge(
                    id="E002",
                    source_id="N003",
                    relation="supports",
                    target_id="N002",
                    role="MechanismProposer",
                    branch_id="B001",
                ),
                Edge(
                    id="E003",
                    source_id="N004",
                    relation="depends_on",
                    target_id="N003",
                    role="EvaluationDesigner",
                    branch_id="B001",
                ),
                Edge(
                    id="E004",
                    source_id="N005",
                    relation="overlaps_prior",
                    target_id="N003",
                    role="NoveltyExaminer",
                    branch_id="B001",
                ),
            ]
        )
    else:
        graph.edges.append(
            Edge(
                id="E900",
                source_id="N003",
                relation="contradicts",
                target_id="N002",
                role="FeasibilityCritic",
                branch_id="B001",
            )
        )
    return graph


def _build_graph_signal_blend_graph() -> IdeaGraph:
    graph = _build_graph_signal_commit_graph(healthy=True)
    graph.edges.extend(
        [
            Edge(
                id="E100",
                source_id="N003",
                relation="contradicts",
                target_id="N002",
                role="FeasibilityCritic",
                branch_id="B001",
            ),
            Edge(
                id="E101",
                source_id="N004",
                relation="repairs",
                target_id="N002",
                role="EvaluationDesigner",
                branch_id="B001",
            ),
            Edge(
                id="E102",
                source_id="N005",
                relation="requires_evidence",
                target_id="N005",
                role="NoveltyExaminer",
                branch_id="B001",
            ),
        ]
    )
    return graph


def test_runtime_commit_check_records_graph_signals_and_deficits_without_controller() -> None:
    graph = _build_graph_signal_commit_graph(healthy=False)
    snapshot = SimpleNamespace(
        support_coverage=0.28,
        unresolved_contradiction_ratio=1.0,
        utility=3.2,
        utility_stable=False,
        completeness=False,
        is_mature=False,
        utility_breakdown=SimpleNamespace(
            support=0.28,
            evidence=0.05,
            coherence=0.22,
            novelty=0.50,
        ),
    )

    record = _runtime_commit_check(
        graph,
        round_name="Round2",
        post_round_snapshot=snapshot,
        runtime_controller=None,
        runtime_controller_metadata=None,
    )

    assert record.source == "maturity_snapshot"
    assert set(record.graph_signals) >= {
        "support",
        "dependency",
        "grounding",
        "challenge",
        "positioning",
        "repair",
        "completeness",
        "maturity",
        "contradiction_load",
    }
    assert set(record.graph_signal_deficits) >= {
        "support",
        "dependency",
        "grounding",
        "challenge",
        "positioning",
        "repair",
        "completeness",
        "maturity",
    }


def test_runtime_commit_check_blends_graph_structure_into_signal_values() -> None:
    graph = _build_graph_signal_blend_graph()
    snapshot = SimpleNamespace(
        support_coverage=0.40,
        unresolved_contradiction_ratio=0.25,
        utility=6.4,
        utility_stable=False,
        completeness=True,
        is_mature=False,
        utility_breakdown=SimpleNamespace(
            support=0.40,
            evidence=0.15,
            coherence=0.58,
            novelty=0.72,
        ),
    )

    record = _runtime_commit_check(
        graph,
        round_name="Round4",
        post_round_snapshot=snapshot,
        runtime_controller=None,
        runtime_controller_metadata=None,
    )

    assert record.graph_signals["support"] > 0.40
    assert record.graph_signals["dependency"] >= 0.60
    assert record.graph_signals["grounding"] > 0.15
    assert record.graph_signals["challenge"] >= 0.35
    assert record.graph_signals["positioning"] >= 0.50
    assert record.graph_signals["repair"] >= 0.60


def test_runtime_commit_check_applies_positive_graph_signal_calibration() -> None:
    class BorderlineCommitController:
        def score_commit_graph(self, graph, *, snapshot=None):
            return 0.46

    graph = _build_graph_signal_commit_graph(healthy=True)
    snapshot = SimpleNamespace(
        support_coverage=0.95,
        unresolved_contradiction_ratio=0.0,
        utility=7.9,
        utility_stable=True,
        completeness=True,
        is_mature=True,
        utility_breakdown=SimpleNamespace(
            support=0.95,
            evidence=0.90,
            coherence=0.84,
            novelty=0.86,
        ),
    )

    record = _runtime_commit_check(
        graph,
        round_name="Round3",
        post_round_snapshot=snapshot,
        runtime_controller=BorderlineCommitController(),
        runtime_controller_metadata={
            "kind": "relation_graph_two_head_critic",
            "config": RelationGraphRuntimeConfig(
                use_edit=False,
                use_commit=True,
                min_commit_round=2,
                gamma_commit=0.50,
                use_action_score_calibration=True,
                action_score_calibration_strength=0.50,
                action_score_calibration_max_bias=0.35,
            ),
        },
    )

    assert record.should_commit is True
    assert record.source == "runtime_controller_commit"
    assert record.commit_probability == 0.46
    assert record.commit_probability_calibrated is not None
    assert record.commit_probability_calibrated > 0.50
    assert record.commit_calibration_bias > 0.0
    assert record.commit_calibration_enabled is True
    assert "maturity" in record.commit_calibration_feedback
    assert "benchmark_specificity" not in record.commit_calibration_feedback
    assert "experiment_alignment" not in record.commit_calibration_feedback


def test_runtime_commit_check_suppresses_commit_when_graph_signals_remain_weak() -> None:
    class OverconfidentCommitController:
        def score_commit_graph(self, graph, *, snapshot=None):
            return 0.62

    graph = _build_graph_signal_commit_graph(healthy=False)
    snapshot = SimpleNamespace(
        support_coverage=0.28,
        unresolved_contradiction_ratio=1.0,
        utility=3.2,
        utility_stable=False,
        completeness=False,
        is_mature=False,
        utility_breakdown=SimpleNamespace(
            support=0.28,
            evidence=0.05,
            coherence=0.22,
            novelty=0.50,
        ),
    )

    record = _runtime_commit_check(
        graph,
        round_name="Round3",
        post_round_snapshot=snapshot,
        runtime_controller=OverconfidentCommitController(),
        runtime_controller_metadata={
            "kind": "relation_graph_two_head_critic",
            "config": RelationGraphRuntimeConfig(
                use_edit=False,
                use_commit=True,
                min_commit_round=2,
                gamma_commit=0.50,
                use_action_score_calibration=True,
                action_score_calibration_strength=0.50,
                action_score_calibration_max_bias=0.35,
            ),
        },
    )

    assert record.should_commit is False
    assert record.source == "runtime_controller_continue"
    assert record.commit_probability == 0.62
    assert record.commit_probability_calibrated is not None
    assert record.commit_probability_calibrated < 0.50
    assert record.commit_calibration_bias < 0.0
    assert record.commit_calibration_enabled is True


def test_runtime_commit_check_allows_late_commit_when_graph_is_ready_but_not_perfect() -> None:
    class LateRoundCommitController:
        def score_commit_graph(self, graph, *, snapshot=None):
            return 0.53

    graph = _build_graph_signal_commit_graph(healthy=True)
    graph.edges = [edge for edge in graph.edges if edge.relation != "overlaps_prior"]
    snapshot = SimpleNamespace(
        support_coverage=0.92,
        unresolved_contradiction_ratio=0.0,
        utility=7.5,
        utility_stable=True,
        completeness=True,
        is_mature=True,
        utility_breakdown=SimpleNamespace(
            support=0.92,
            evidence=0.88,
            coherence=0.85,
            novelty=0.81,
        ),
    )

    record = _runtime_commit_check(
        graph,
        round_name="Round5",
        post_round_snapshot=snapshot,
        runtime_controller=LateRoundCommitController(),
        runtime_controller_metadata={
            "kind": "relation_graph_two_head_critic",
            "config": RelationGraphRuntimeConfig(
                use_edit=False,
                use_commit=True,
                min_commit_round=3,
                gamma_commit=0.50,
                use_action_score_calibration=True,
                action_score_calibration_strength=0.50,
                action_score_calibration_max_bias=0.35,
            ),
        },
    )

    assert record.commit_calibration_feedback.get("positioning") == 0.0
    assert record.should_commit is True
    assert record.source == "runtime_controller_commit"
    assert record.commit_probability_calibrated is not None
    assert record.commit_probability_calibrated >= 0.50
