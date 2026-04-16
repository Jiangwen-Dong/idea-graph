from __future__ import annotations

from copy import deepcopy

from .agent_backend import ActionDecision
from .engine import action_from_decision, apply_action, choose_round_action
from .models import ParallelRoleRoundResult
from .parallel_replay import build_parallel_edit_rows
from .parallel_role_executor import collect_parallel_role_decisions
from .role_activation import active_roles_for_round


def _decision_from_graph_action(action) -> ActionDecision:
    return ActionDecision(
        kind=action.kind,
        target_ids=list(action.target_ids),
        payload=dict(action.payload),
        rationale=action.rationale,
    )


def execute_parallel_role_round(
    graph,
    *,
    round_name,
    collaboration_backend,
    runtime_controller,
    runtime_controller_metadata,
    progress_callback,
):
    del runtime_controller, runtime_controller_metadata, progress_callback
    node_count_before = len(graph.nodes)
    edge_count_before = len(graph.edges)
    action_count_before = len(graph.actions)
    roles = active_roles_for_round(graph, round_name)
    snapshot = deepcopy(graph)
    if collaboration_backend is None:
        raw_decisions = [
            (
                role,
                _decision_from_graph_action(choose_round_action(snapshot, round_name, role)),
            )
            for role in roles
        ]
        action_source = "parallel_deterministic"
        label_source = "parallel_protocol_teacher_v1"
    else:
        raw_decisions = collect_parallel_role_decisions(
            graph,
            round_name,
            collaboration_backend,
            roles,
        )
        action_source = "parallel_llm"
        label_source = "parallel_runtime_logged_v1"
    edit_rows = build_parallel_edit_rows(
        snapshot,
        round_name=round_name,
        role_decisions=raw_decisions,
        runtime_protocol="parallel_graph_v2",
        label_source=label_source,
    )
    selected_actions = []
    skipped_roles = []
    for role, decision in sorted(raw_decisions, key=lambda item: item[0]):
        if str(decision.kind).strip() == "skip":
            skipped_roles.append(role)
            continue
        action = action_from_decision(
            graph,
            round_name=round_name,
            role=role,
            decision=decision,
        )
        action.source = action_source
        apply_action(graph, action)
        selected_actions.append(action)
    return ParallelRoleRoundResult(
        round_name=round_name,
        active_roles=tuple(role for role, _ in raw_decisions),
        skipped_roles=tuple(skipped_roles),
        selected_actions=tuple(selected_actions),
        termination_reason="continue",
        edit_rows=tuple(dict(row) for row in edit_rows),
        node_count_before=node_count_before,
        node_count_after=len(graph.nodes),
        edge_count_before=edge_count_before,
        edge_count_after=len(graph.edges),
        action_count_before=action_count_before,
        action_count_after=len(graph.actions),
    )


__all__ = ["ParallelRoleRoundResult", "execute_parallel_role_round"]
