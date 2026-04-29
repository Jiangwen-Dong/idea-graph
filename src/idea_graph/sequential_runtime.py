from __future__ import annotations

from copy import deepcopy
from typing import Any

from .agent_backend import ActionDecision, append_agent_trace
from .engine import action_from_decision, apply_action, choose_round_action, maturity_snapshot
from .models import ParallelRoleRoundResult
from .parallel_replay import build_parallel_edit_rows, build_post_round_commit_row
from .parallel_role_executor import _skip_decision_from_role_error
from .parallel_runtime import (
    _decision_from_graph_action,
    _decision_record,
    _maybe_apply_runtime_controller,
    _patch_record,
    _runtime_commit_check,
)
from .role_activation import active_roles_for_round
from .runtime_protocols import ROLE_ORDER_PRESETS, SEQUENTIAL_GRAPH_V2, resolve_role_order


def _branch_id_for_role(graph, role: str) -> str:
    return next(branch.id for branch in graph.branches.values() if branch.role == role)


def _effective_role_order_metadata(graph) -> tuple[str, list[str]]:
    role_order_id = str(graph.metadata.get("runtime_role_order_id", "")).strip() or "order_a_canonical"
    try:
        role_order = list(ROLE_ORDER_PRESETS[role_order_id])
    except KeyError as exc:
        options = ", ".join(sorted(ROLE_ORDER_PRESETS))
        raise ValueError(
            f"Unknown role order '{role_order_id}'. Expected one of: {options}."
        ) from exc
    graph.metadata["runtime_role_order_id"] = role_order_id
    graph.metadata["runtime_role_order"] = list(role_order)
    return role_order_id, role_order


def _raw_role_decision(
    graph,
    *,
    role_snapshot,
    round_name: str,
    role: str,
    collaboration_backend,
) -> tuple[ActionDecision, str, str]:
    if collaboration_backend is None:
        return (
            _decision_from_graph_action(choose_round_action(role_snapshot, round_name, role)),
            "sequential_deterministic",
            "sequential_protocol_teacher_v1",
        )

    try:
        decision = collaboration_backend.choose_action(role_snapshot, round_name, role)
    except (RuntimeError, ValueError) as exc:
        decision = _skip_decision_from_role_error(role, exc)
    append_agent_trace(
        graph,
        stage=f"{round_name}_action",
        role=role,
        trace=decision.trace,
    )
    return decision, "sequential_llm", "sequential_runtime_logged_v1"


def execute_sequential_role_round(
    graph,
    *,
    round_name,
    collaboration_backend,
    runtime_controller,
    runtime_controller_metadata,
    progress_callback,
):
    del progress_callback
    node_count_before = len(graph.nodes)
    edge_count_before = len(graph.edges)
    action_count_before = len(graph.actions)

    active_roles = active_roles_for_round(graph, round_name)
    role_order_id, _role_order = _effective_role_order_metadata(graph)
    resolved_roles = resolve_role_order(role_order_id, active_roles)

    selected_role_decisions = []
    edit_patches = []
    materialized_graph_actions = []
    skipped_roles = []
    edit_rows = []

    for role in resolved_roles:
        role_snapshot = deepcopy(graph)
        raw_decision, action_source, label_source = _raw_role_decision(
            graph,
            role_snapshot=role_snapshot,
            round_name=round_name,
            role=role,
            collaboration_backend=collaboration_backend,
        )
        controlled_decisions, used_controller = _maybe_apply_runtime_controller(
            graph,
            snapshot=role_snapshot,
            round_name=round_name,
            raw_decisions=[(role, raw_decision)],
            runtime_controller=runtime_controller,
            runtime_controller_metadata=runtime_controller_metadata,
            runtime_protocol=SEQUENTIAL_GRAPH_V2,
        )
        decision = controlled_decisions[0][1]
        if used_controller:
            action_source = "sequential_controller"
            label_source = "sequential_runtime_controller_v1"

        edit_rows.extend(
            build_parallel_edit_rows(
                role_snapshot,
                round_name=round_name,
                role_decisions=[(role, decision)],
                runtime_protocol=SEQUENTIAL_GRAPH_V2,
                label_source=label_source,
                state_kind="sequential_pre_action",
                runtime_role_order_id=role_order_id,
                resolved_role_sequence=resolved_roles,
            )
        )

        selected_role_decisions.append(_decision_record(role, decision))
        if str(decision.kind).strip() == "skip":
            skipped_roles.append(role)
            edit_patches.append(_patch_record(role, decision, is_empty=True))
            continue

        edit_patches.append(_patch_record(role, decision, is_empty=False))
        action = action_from_decision(
            graph,
            round_name=round_name,
            role=role,
            decision=decision,
        )
        if not action.payload.get("branch_id"):
            action.payload["branch_id"] = _branch_id_for_role(graph, role)
        action.source = action_source
        apply_action(graph, action)
        materialized_graph_actions.append(action)

    post_round_snapshot = maturity_snapshot(graph)
    post_round_commit = _runtime_commit_check(
        graph,
        round_name=round_name,
        post_round_snapshot=post_round_snapshot,
        runtime_controller=runtime_controller,
        runtime_controller_metadata=runtime_controller_metadata,
        state_kind="sequential_post_round",
    )
    post_round_commit_row = build_post_round_commit_row(
        graph,
        round_name=round_name,
        commit_check=post_round_commit,
        runtime_protocol=SEQUENTIAL_GRAPH_V2,
        label_source=post_round_commit.source,
        state_kind="sequential_post_round",
        runtime_role_order_id=role_order_id,
        resolved_role_sequence=resolved_roles,
    )
    return ParallelRoleRoundResult(
        round_name=round_name,
        active_roles=tuple(resolved_roles),
        skipped_roles=tuple(skipped_roles),
        selected_role_decisions=tuple(selected_role_decisions),
        edit_patches=tuple(edit_patches),
        materialized_graph_actions=tuple(materialized_graph_actions),
        post_round_commit=post_round_commit,
        edit_rows=tuple(dict(row) for row in edit_rows),
        post_round_commit_rows=(dict(post_round_commit_row),),
        node_count_before=node_count_before,
        node_count_after=len(graph.nodes),
        edge_count_before=edge_count_before,
        edge_count_after=len(graph.edges),
        action_count_before=action_count_before,
        action_count_after=len(graph.actions),
    )


__all__ = ["execute_sequential_role_round"]
