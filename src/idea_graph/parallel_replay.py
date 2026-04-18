from __future__ import annotations

from typing import Any, Mapping, Sequence

from .action_candidates import enumerate_edit_candidate_specs, flatten_candidate_text
from .models import GraphAction, IdeaGraph


def append_parallel_round_trace(metadata: dict[str, object], payload: dict[str, object]) -> None:
    traces = metadata.setdefault("parallel_round_traces", [])
    if isinstance(traces, list):
        traces.append(dict(payload))


def append_parallel_edit_rows(
    metadata: dict[str, object],
    rows: Sequence[Mapping[str, object]],
) -> None:
    payload = metadata.setdefault("parallel_edit_rows", [])
    if isinstance(payload, list):
        payload.extend(dict(row) for row in rows)


def append_post_round_commit_rows(
    metadata: dict[str, object],
    rows: Sequence[Mapping[str, object]],
) -> None:
    payload = metadata.setdefault("post_round_commit_rows", [])
    if isinstance(payload, list):
        payload.extend(dict(row) for row in rows)


def _state_snapshot(
    graph: IdeaGraph,
    *,
    state_kind: str = "parallel_pre_action",
    action_id: str = "parallel_round_pre_action",
    action_timestamp: object | None = None,
) -> dict[str, Any]:
    included_nodes = {
        node_id: {
            "id": node.id,
            "type": node.type,
            "text": node.text,
            "role": node.role,
            "branch_id": node.branch_id,
            "confidence": node.confidence,
            "evidence": list(node.evidence),
            "status": node.status,
            "created_at": node.created_at,
            "provenance": [
                {
                    "role": item.role,
                    "branch_id": item.branch_id,
                    "source": item.source,
                }
                for item in node.provenance
            ],
        }
        for node_id, node in sorted(graph.nodes.items())
    }

    included_edges: list[dict[str, Any]] = []
    contradiction_count = 0
    support_edge_count = 0
    for edge in graph.edges:
        if edge.relation == "supports":
            support_edge_count += 1
        if edge.relation == "contradicts" and not edge.resolved:
            contradiction_count += 1
        included_edges.append(
            {
                "id": edge.id,
                "source_id": edge.source_id,
                "relation": edge.relation,
                "target_id": edge.target_id,
                "role": edge.role,
                "branch_id": edge.branch_id,
                "evidence_id": edge.evidence_id,
                "note": edge.note,
                "resolved": edge.resolved,
                "created_at": edge.created_at,
            }
        )

    return {
        "action_id": action_id,
        "action_index": len(graph.actions),
        "action_timestamp": action_timestamp,
        "state_kind": state_kind,
        "nodes": included_nodes,
        "edges": included_edges,
        "node_count": len(included_nodes),
        "edge_count": len(included_edges),
        "contradiction_count": contradiction_count,
        "support_edge_count": support_edge_count,
        "action_count": len(graph.actions),
    }


def _flatten_state_text(graph: IdeaGraph) -> str:
    node_fragments = [
        f"{node.id}:{node.type}:{node.text}"
        for node in sorted(graph.nodes.values(), key=lambda item: item.id)
    ]
    edge_fragments = [
        f"{edge.source_id}-{edge.relation}->{edge.target_id}"
        for edge in sorted(graph.edges, key=lambda item: (item.source_id, item.relation, item.target_id, item.id))
    ]
    return (
        f"nodes={len(graph.nodes)};"
        f"edges={len(graph.edges)};"
        f"contradictions={sum(1 for edge in graph.edges if edge.relation == 'contradicts' and not edge.resolved)};"
        f"node_details=[{' || '.join(node_fragments)}];"
        f"edge_details=[{' || '.join(edge_fragments)}]"
    )


def _metadata_string(graph: IdeaGraph, key: str, default: str = "unknown") -> str:
    value = graph.metadata.get(key, default)
    text = str(value).strip()
    return text or default


def _synthetic_action(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    decision,
) -> GraphAction:
    payload = dict(decision.payload or {})
    return GraphAction(
        id=f"parallel::{round_name}::{role}",
        round_name=round_name,
        role=role,
        kind=str(decision.kind).strip(),
        target_ids=[str(item).strip() for item in decision.target_ids if str(item).strip()],
        payload=payload,
        rationale=str(decision.rationale).strip(),
        source="parallel_label_builder",
    )


def _build_candidate_rows(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    selected_action: GraphAction,
) -> tuple[list[dict[str, object]], str]:
    candidate_specs = enumerate_edit_candidate_specs(
        graph,
        round_name=round_name,
        role=role,
        baseline_action=selected_action,
    )
    candidate_rows: list[dict[str, object]] = []
    selected_candidate_id = ""
    for candidate_index, spec in enumerate(candidate_specs):
        candidate_id = f"parallel::{round_name}::{role}::candidate:{candidate_index:04d}"
        is_selected = (
            str(spec.get("kind", "")).strip() == selected_action.kind
            and [str(item).strip() for item in spec.get("target_ids", []) if str(item).strip()] == selected_action.target_ids
            and dict(spec.get("payload", {}) or {}) == dict(selected_action.payload)
        )
        if is_selected:
            selected_candidate_id = candidate_id
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "candidate_kind": str(spec.get("kind", "")).strip(),
                "candidate_target_ids": [
                    str(item).strip() for item in spec.get("target_ids", []) if str(item).strip()
                ],
                "candidate_payload": dict(spec.get("payload", {}) or {}),
                "candidate_rationale": str(spec.get("rationale", "")).strip(),
                "candidate_source": str(spec.get("candidate_source", "")).strip(),
                "candidate_text": flatten_candidate_text(graph, spec),
                "is_selected": is_selected,
            }
        )
    if not selected_candidate_id:
        raise ValueError(f"Parallel edit slate for {round_name}/{role} is missing the selected candidate.")
    return candidate_rows, selected_candidate_id


def build_parallel_edit_rows(
    graph: IdeaGraph,
    *,
    round_name: str,
    role_decisions: Sequence[tuple[str, Any]],
    runtime_protocol: str,
    label_source: str,
) -> list[dict[str, object]]:
    state_snapshot = _state_snapshot(graph)
    state_text = _flatten_state_text(graph)
    rows: list[dict[str, object]] = []
    for role, decision in role_decisions:
        selected_action = _synthetic_action(
            graph,
            round_name=round_name,
            role=role,
            decision=decision,
        )
        candidates, selected_candidate_id = _build_candidate_rows(
            graph,
            round_name=round_name,
            role=role,
            selected_action=selected_action,
        )
        rows.append(
            {
                "schema_version": "parallel_edit_row_v1",
                "state_id": f"parallel::{round_name}::{role}",
                "runtime_protocol": runtime_protocol,
                "label_source": label_source,
                "benchmark": _metadata_string(graph, "benchmark"),
                "instance_name": _metadata_string(graph, "instance_name", graph.topic),
                "baseline_name": _metadata_string(graph, "baseline_name"),
                "topic": graph.topic,
                "round_name": round_name,
                "role": role,
                "state_kind": "parallel_pre_action",
                "state_text": state_text,
                "state_snapshot": state_snapshot,
                "state_node_count": state_snapshot["node_count"],
                "state_edge_count": state_snapshot["edge_count"],
                "state_action_count": state_snapshot["action_count"],
                "candidate_count": len(candidates),
                "selected_candidate_id": selected_candidate_id,
                "selected_action_kind": selected_action.kind,
                "selected_action_targets": list(selected_action.target_ids),
                "selected_action_payload": dict(selected_action.payload),
                "selected_action_source": str(label_source),
                "candidates": candidates,
            }
        )
    return rows


def build_post_round_commit_row(
    graph: IdeaGraph,
    *,
    round_name: str,
    commit_check: Any,
    runtime_protocol: str,
    label_source: str,
) -> dict[str, object]:
    state_snapshot = _state_snapshot(
        graph,
        state_kind="parallel_post_round",
        action_id="parallel_round_post_action",
    )
    should_commit = bool(getattr(commit_check, "should_commit", False))
    return {
        "schema_version": "post_round_commit_row_v1",
        "state_id": f"parallel::{round_name}::post_round_commit",
        "runtime_protocol": runtime_protocol,
        "label_source": label_source,
        "benchmark": _metadata_string(graph, "benchmark"),
        "instance_name": _metadata_string(graph, "instance_name", graph.topic),
        "baseline_name": _metadata_string(graph, "baseline_name"),
        "topic": graph.topic,
        "round_name": round_name,
        "state_kind": "parallel_post_round",
        "state_text": _flatten_state_text(graph),
        "state_snapshot": state_snapshot,
        "state_node_count": state_snapshot["node_count"],
        "state_edge_count": state_snapshot["edge_count"],
        "state_action_count": state_snapshot["action_count"],
        "commit_supervision": {
            "available": True,
            "label": 1 if should_commit else 0,
            "source": str(getattr(commit_check, "source", label_source)).strip() or label_source,
        },
        "support_coverage": float(getattr(commit_check, "support_coverage", 0.0)),
        "unresolved_contradiction_ratio": float(
            getattr(commit_check, "unresolved_contradiction_ratio", 0.0)
        ),
        "utility": float(getattr(commit_check, "utility", 0.0)),
        "controller_kind": str(getattr(commit_check, "controller_kind", "") or ""),
        "commit_probability": getattr(commit_check, "commit_probability", None),
        "commit_threshold": getattr(commit_check, "commit_threshold", None),
    }
