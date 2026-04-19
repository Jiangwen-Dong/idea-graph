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
from idea_graph.models import IdeaGraph
from idea_graph.models import (
    ParallelCommitCheckRecord,
    ParallelEditPatchRecord,
    ParallelRoleDecisionRecord,
    ParallelRoleRoundResult,
)
from idea_graph.parallel_runtime import execute_parallel_role_round
from idea_graph.relation_graph_runtime_critic import RelationGraphRuntimeConfig


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
