from __future__ import annotations

import json

from .models import GraphAction, IdeaGraph


def build_action_spec(
    *,
    kind: str,
    target_ids: list[str],
    payload: dict[str, object] | None = None,
    rationale: str = "",
    candidate_source: str = "heuristic",
) -> dict[str, object]:
    return {
        "kind": kind,
        "target_ids": list(target_ids),
        "payload": dict(payload or {}),
        "rationale": rationale,
        "candidate_source": candidate_source,
    }


def action_spec_from_action(action: GraphAction, *, candidate_source: str) -> dict[str, object]:
    return build_action_spec(
        kind=action.kind,
        target_ids=list(action.target_ids),
        payload=dict(action.payload),
        rationale=action.rationale,
        candidate_source=candidate_source,
    )


def _normalize_spec_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple((str(key), _normalize_spec_value(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_normalize_spec_value(item) for item in value)
    return str(value)


def _action_spec_signature(spec: dict[str, object]) -> tuple[object, ...]:
    payload = spec.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    return (
        str(spec.get("kind", "")).strip(),
        tuple(str(item).strip() for item in spec.get("target_ids", []) if str(item).strip()),
        tuple((str(key), _normalize_spec_value(value)) for key, value in sorted(payload.items())),
    )


def dedupe_action_specs(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        signature = _action_spec_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(candidate)
    return deduped


def enumerate_candidate_specs(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    baseline_action: GraphAction,
) -> list[dict[str, object]]:
    from .engine import branch_for_role, generic_candidate_action_specs

    branch = branch_for_role(graph, role)
    baseline_candidate = action_spec_from_action(baseline_action, candidate_source="legacy_policy")
    candidates = [baseline_candidate, *generic_candidate_action_specs(graph, round_name, role, branch)]
    candidates.append(
        build_action_spec(
            kind="commit",
            target_ids=[],
            payload={"branch_id": branch.id},
            rationale="Commit to the current graph state without applying an additional edit.",
            candidate_source="commit",
        )
    )
    return dedupe_action_specs(candidates)


def flatten_candidate_text(graph: IdeaGraph, spec: dict[str, object]) -> str:
    target_descriptions: list[str] = []
    for target_id in [str(item).strip() for item in spec.get("target_ids", []) if str(item).strip()]:
        node = graph.nodes.get(target_id)
        if node is None:
            target_descriptions.append(target_id)
            continue
        target_descriptions.append(f"{node.type}: {node.text}")

    payload = spec.get("payload", {})
    payload_text = ""
    if isinstance(payload, dict) and payload:
        payload_text = json.dumps(payload, ensure_ascii=True, sort_keys=True)

    parts = [f"kind={str(spec.get('kind', '')).strip()}"]
    if target_descriptions:
        parts.append(f"targets={'; '.join(target_descriptions)}")
    if payload_text:
        parts.append(f"payload={payload_text}")
    rationale = str(spec.get("rationale", "")).strip()
    if rationale:
        parts.append(f"rationale={rationale}")
    candidate_source = str(spec.get("candidate_source", "")).strip()
    if candidate_source:
        parts.append(f"source={candidate_source}")
    return " | ".join(parts)
