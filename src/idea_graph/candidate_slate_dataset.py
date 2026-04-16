from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_candidates import enumerate_candidate_specs, flatten_candidate_text
from .critic_dataset import (
    _as_object_dict,
    assign_group_splits,
    build_group_manifest,
    make_group_id,
    package_labels_from_manifest_row,
)
from .fs_utils import read_text_file, write_text_file
from .models import Branch, Edge, GraphAction, IdeaGraph, Node, Provenance


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_timestamp(value: object) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return datetime.now(timezone.utc)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = read_text_file(path)
    for line_index, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_index} must contain a JSON object.")
        rows.append(dict(payload))
    return rows


def _load_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    candidate_path = Path(path)
    if not candidate_path.exists():
        return []
    return _load_jsonl(candidate_path)


def load_g1_state_snapshot(g1_dataset_dir: Path, transition: Mapping[str, Any]) -> dict[str, Any]:
    relative_path = str(transition.get("before_state_snapshot", "")).strip()
    if not relative_path:
        raise ValueError("Transition row is missing required before_state_snapshot.")
    snapshot_path = Path(g1_dataset_dir) / relative_path
    payload = json.loads(read_text_file(snapshot_path))
    if not isinstance(payload, dict):
        raise ValueError(f"{snapshot_path} does not contain a JSON object.")
    return dict(payload)


def _role_for_terminal_enumeration(graph: IdeaGraph, transition: Mapping[str, Any], default_role: str) -> str:
    state_kind = str(transition.get("state_kind", "")).strip()
    if state_kind != "terminal_commit":
        return default_role
    payload = _as_object_dict(transition.get("selected_action_payload"))
    branch_id = str(payload.get("branch_id", "")).strip() or str(transition.get("selected_action_branch_id", "")).strip()
    if branch_id and branch_id in graph.branches and graph.branches[branch_id].role:
        return graph.branches[branch_id].role
    if graph.branches:
        return next(iter(graph.branches.values())).role
    return default_role


def state_id_from_transition(transition: Mapping[str, Any]) -> str:
    run_dir = str(transition.get("run_dir", "")).strip()
    round_name = str(transition.get("round_name", "")).strip()
    role = str(transition.get("role", "")).strip()
    step_index = _safe_int(transition.get("step_index"))
    return f"{run_dir}::step:{step_index:04d}::{round_name}::{role}"


def flatten_state_text(snapshot: Mapping[str, Any]) -> str:
    nodes_payload = snapshot.get("nodes", {})
    edges_payload = snapshot.get("edges", [])
    node_fragments: list[str] = []
    edge_fragments: list[str] = []

    if isinstance(nodes_payload, Mapping):
        for node_id, node_payload in sorted(nodes_payload.items(), key=lambda item: str(item[0])):
            node_obj = node_payload if isinstance(node_payload, Mapping) else {}
            node_type = str(node_obj.get("type", "")).strip()
            node_text = str(node_obj.get("text", "")).strip()
            node_fragments.append(f"{node_id}:{node_type}:{node_text}")

    if isinstance(edges_payload, list):
        for edge_payload in edges_payload:
            if not isinstance(edge_payload, Mapping):
                continue
            source_id = str(edge_payload.get("source_id", "")).strip()
            relation = str(edge_payload.get("relation", "")).strip()
            target_id = str(edge_payload.get("target_id", "")).strip()
            edge_fragments.append(f"{source_id}-{relation}->{target_id}")
    edge_fragments.sort()

    node_text = " || ".join(node_fragments)
    edge_text = " || ".join(edge_fragments)
    return (
        f"nodes={_safe_int(snapshot.get('node_count'))};"
        f"edges={_safe_int(snapshot.get('edge_count'))};"
        f"contradictions={_safe_int(snapshot.get('contradiction_count'))};"
        f"node_details=[{node_text}];edge_details=[{edge_text}]"
    )


def _build_branches(snapshot: Mapping[str, Any], *, transition_role: str | None = None) -> dict[str, Branch]:
    branches: dict[str, Branch] = {}
    nodes_payload = snapshot.get("nodes", {})
    edges_payload = snapshot.get("edges", [])

    if isinstance(nodes_payload, Mapping):
        for node_id, node_payload in nodes_payload.items():
            if not isinstance(node_payload, Mapping):
                continue
            branch_id = str(node_payload.get("branch_id", "")).strip()
            if not branch_id:
                continue
            role = str(node_payload.get("role", "")).strip() or (transition_role or "UnknownRole")
            branch = branches.setdefault(branch_id, Branch(id=branch_id, role=role))
            if str(node_id) not in branch.node_ids:
                branch.node_ids.append(str(node_id))

    if isinstance(edges_payload, list):
        for edge_payload in edges_payload:
            if not isinstance(edge_payload, Mapping):
                continue
            edge_id = str(edge_payload.get("id", "")).strip()
            branch_id = str(edge_payload.get("branch_id", "")).strip()
            if not branch_id:
                continue
            role = str(edge_payload.get("role", "")).strip() or (transition_role or "UnknownRole")
            branch = branches.setdefault(branch_id, Branch(id=branch_id, role=role))
            if edge_id and edge_id not in branch.edge_ids:
                branch.edge_ids.append(edge_id)

    if transition_role and all(branch.role != transition_role for branch in branches.values()):
        new_branch_id = f"B_{transition_role}"
        suffix = 0
        while new_branch_id in branches:
            suffix += 1
            new_branch_id = f"B_{transition_role}_{suffix}"
        branches[new_branch_id] = Branch(id=new_branch_id, role=transition_role)

    return branches


def graph_from_state_snapshot(
    snapshot: Mapping[str, Any],
    *,
    topic: str = "",
    literature: Sequence[str] | None = None,
    role: str | None = None,
    round_name: str | None = None,
) -> IdeaGraph:
    graph = IdeaGraph(
        topic=topic,
        literature=[str(item).strip() for item in (literature or []) if str(item).strip()],
        metadata={},
    )
    transition_role = role or ""
    graph.branches = _build_branches(snapshot, transition_role=transition_role or None)

    nodes_payload = snapshot.get("nodes", {})
    if isinstance(nodes_payload, Mapping):
        for node_id, node_payload in nodes_payload.items():
            if not isinstance(node_payload, Mapping):
                continue
            branch_id = str(node_payload.get("branch_id", "")).strip()
            if branch_id and branch_id not in graph.branches:
                graph.branches[branch_id] = Branch(
                    id=branch_id,
                    role=str(node_payload.get("role", "")).strip() or transition_role or "UnknownRole",
                )

            provenance_payload = node_payload.get("provenance", [])
            provenance: list[Provenance] = []
            if isinstance(provenance_payload, list):
                for item in provenance_payload:
                    if not isinstance(item, Mapping):
                        continue
                    provenance.append(
                        Provenance(
                            role=str(item.get("role", "")).strip(),
                            branch_id=str(item.get("branch_id", "")).strip(),
                            source=str(item.get("source", "")).strip(),
                        )
                    )

            node = Node(
                id=str(node_payload.get("id", str(node_id))).strip() or str(node_id),
                type=str(node_payload.get("type", "")).strip(),
                text=str(node_payload.get("text", "")).strip(),
                role=str(node_payload.get("role", "")).strip() or transition_role or "UnknownRole",
                branch_id=branch_id,
                confidence=_safe_float(node_payload.get("confidence"), default=0.0),
                evidence=[str(item) for item in node_payload.get("evidence", [])] if isinstance(node_payload.get("evidence"), list) else [],
                status=str(node_payload.get("status", "active")).strip() or "active",
                created_at=_parse_timestamp(node_payload.get("created_at")),
                provenance=provenance,
            )
            graph.nodes[node.id] = node
            if branch_id in graph.branches and node.id not in graph.branches[branch_id].node_ids:
                graph.branches[branch_id].node_ids.append(node.id)

    edges_payload = snapshot.get("edges", [])
    if isinstance(edges_payload, list):
        for edge_payload in edges_payload:
            if not isinstance(edge_payload, Mapping):
                continue
            edge = Edge(
                id=str(edge_payload.get("id", "")).strip(),
                source_id=str(edge_payload.get("source_id", "")).strip(),
                relation=str(edge_payload.get("relation", "")).strip(),
                target_id=str(edge_payload.get("target_id", "")).strip(),
                role=str(edge_payload.get("role", "")).strip() or transition_role or "UnknownRole",
                branch_id=str(edge_payload.get("branch_id", "")).strip(),
                evidence_id=(
                    str(edge_payload.get("evidence_id", "")).strip() or None
                    if edge_payload.get("evidence_id") is not None
                    else None
                ),
                note=str(edge_payload.get("note", "")).strip(),
                resolved=bool(edge_payload.get("resolved", False)),
                created_at=_parse_timestamp(edge_payload.get("created_at")),
            )
            graph.edges.append(edge)
            if edge.branch_id in graph.branches and edge.id and edge.id not in graph.branches[edge.branch_id].edge_ids:
                graph.branches[edge.branch_id].edge_ids.append(edge.id)

    graph.actions = []
    graph.round_summaries = []
    graph.metadata["round_name"] = round_name or ""
    return graph


def _infer_branch_id(graph: IdeaGraph, transition: Mapping[str, Any]) -> str:
    payload_hint = _as_object_dict(transition.get("selected_action_payload"))
    hinted_branch_id = str(payload_hint.get("branch_id", "")).strip()
    if hinted_branch_id:
        return hinted_branch_id

    transition_branch_id = str(transition.get("selected_action_branch_id", "")).strip()
    if transition_branch_id:
        return transition_branch_id

    targets = transition.get("selected_action_targets", [])
    if isinstance(targets, list):
        for target_id in targets:
            node = graph.nodes.get(str(target_id))
            if node is not None and node.branch_id:
                return node.branch_id

    role = str(transition.get("role", "")).strip()
    if role:
        for branch in graph.branches.values():
            if branch.role == role:
                return branch.id

    if graph.branches:
        return next(iter(graph.branches.values())).id

    fallback = f"B_{role or 'fallback'}"
    graph.branches[fallback] = Branch(id=fallback, role=role or "UnknownRole")
    return fallback


def action_from_transition(transition: Mapping[str, Any], graph: IdeaGraph) -> GraphAction:
    kind = str(transition.get("selected_action_kind", "")).strip()
    round_name = str(transition.get("round_name", "")).strip()
    role = str(transition.get("role", "")).strip()
    targets = [str(item).strip() for item in transition.get("selected_action_targets", []) if str(item).strip()]
    payload = _as_object_dict(transition.get("selected_action_payload"))
    branch_id = str(payload.get("branch_id", "")).strip()
    if not branch_id:
        branch_id = _infer_branch_id(graph, transition)
        payload["branch_id"] = branch_id
    rationale = str(transition.get("selected_action_rationale", "")).strip()
    return GraphAction(
        id=f"logged::{state_id_from_transition(transition)}",
        round_name=round_name,
        role=role,
        kind=kind,
        target_ids=targets,
        payload=payload,
        rationale=rationale,
        source=str(transition.get("selected_action_source", "")).strip() or "logged",
        timestamp=datetime.now(timezone.utc),
    )


def _action_signature(kind: str, target_ids: Sequence[str], payload: Mapping[str, object]) -> tuple[object, ...]:
    normalized_targets = tuple(str(item).strip() for item in target_ids if str(item).strip())
    normalized_payload = tuple((str(key), json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)) for key, value in sorted(payload.items()))
    return (kind.strip(), normalized_targets, normalized_payload)


def _transition_sort_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        str(row.get("group_id", "")).strip(),
        str(row.get("run_dir", "")).strip(),
        _safe_int(row.get("step_index")),
        str(row.get("round_name", "")).strip(),
        str(row.get("role", "")).strip(),
    )


def build_candidate_dataset_rows(
    *,
    g1_dataset_dir: Path,
    critic_rows: Sequence[Mapping[str, Any]],
    terminal_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    state_manifest: list[dict[str, Any]] = []

    merged_rows = [dict(row) for row in critic_rows]
    if terminal_rows:
        merged_rows.extend(dict(row) for row in terminal_rows)
    sorted_rows = sorted(merged_rows, key=_transition_sort_key)
    for transition in sorted_rows:
        state_id = state_id_from_transition(transition)
        snapshot = load_g1_state_snapshot(Path(g1_dataset_dir), transition)
        round_name = str(transition.get("round_name", "")).strip()
        role = str(transition.get("role", "")).strip()
        state_kind = str(transition.get("state_kind", "")).strip() or "pre_action"
        graph = graph_from_state_snapshot(
            snapshot,
            topic=str(transition.get("topic", "")).strip(),
            literature=[
                str(item).strip()
                for item in transition.get("state_literature", [])
                if str(item).strip()
            ]
            if isinstance(transition.get("state_literature"), list)
            else [],
            role=role,
            round_name=round_name,
        )
        enumeration_role = _role_for_terminal_enumeration(graph, transition, role)
        logged_action = action_from_transition(transition, graph)
        candidate_specs = enumerate_candidate_specs(
            graph,
            round_name=round_name,
            role=enumeration_role,
            baseline_action=logged_action,
        )

        logged_signature = _action_signature(
            logged_action.kind,
            logged_action.target_ids,
            _as_object_dict(logged_action.payload),
        )
        logged_selected_count = 0

        for candidate_index, spec in enumerate(candidate_specs):
            kind = str(spec.get("kind", "")).strip()
            target_ids = [str(item).strip() for item in spec.get("target_ids", []) if str(item).strip()]
            payload = _as_object_dict(spec.get("payload"))
            is_logged_selected = _action_signature(kind, target_ids, payload) == logged_signature
            if is_logged_selected:
                logged_selected_count += 1
            row = {
                "state_id": state_id,
                "candidate_id": f"{state_id}::candidate:{candidate_index:04d}",
                "group_id": str(transition.get("group_id", "")).strip(),
                "split": str(transition.get("split", "train")).strip() or "train",
                "benchmark": str(transition.get("benchmark", "unknown")).strip() or "unknown",
                "instance_name": str(transition.get("instance_name", "unknown")).strip() or "unknown",
                "run_dir": str(transition.get("run_dir", "")).strip(),
                "step_index": _safe_int(transition.get("step_index")),
                "round_name": round_name,
                "role": role,
                "state_kind": state_kind,
                "state_text": flatten_state_text(snapshot),
                "candidate_index": candidate_index,
                "candidate_count": len(candidate_specs),
                "candidate_kind": kind,
                "candidate_target_ids": target_ids,
                "candidate_payload": payload,
                "candidate_source": str(spec.get("candidate_source", "")).strip(),
                "candidate_text": flatten_candidate_text(graph, spec),
                "is_commit": kind == "commit",
                "is_logged_selected": is_logged_selected,
                "is_commit_positive_state": bool(
                    _as_object_dict(transition.get("commit_supervision")).get("label") == 1
                ),
                "commit_supervision": _as_object_dict(transition.get("commit_supervision")),
                "targets": _as_object_dict(transition.get("targets")),
                "weak_local": _as_object_dict(transition.get("weak_local")),
                "native": _as_object_dict(transition.get("native")),
                "label_availability": _as_object_dict(transition.get("label_availability")),
            }
            candidate_rows.append(row)

        if logged_selected_count != 1:
            raise ValueError(
                f"State '{state_id}' must have exactly one logged-selected candidate; found {logged_selected_count}."
            )

        state_manifest.append(
            {
                "state_id": state_id,
                "group_id": str(transition.get("group_id", "")).strip(),
                "split": str(transition.get("split", "train")).strip() or "train",
                "benchmark": str(transition.get("benchmark", "unknown")).strip() or "unknown",
                "instance_name": str(transition.get("instance_name", "unknown")).strip() or "unknown",
                "run_dir": str(transition.get("run_dir", "")).strip(),
                "step_index": _safe_int(transition.get("step_index")),
                "round_name": round_name,
                "role": role,
                "state_kind": state_kind,
                "candidate_count": len(candidate_specs),
            }
        )

    return candidate_rows, state_manifest


def build_candidate_dataset_stats(
    candidate_rows: Sequence[Mapping[str, Any]],
    state_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    state_count = len(state_manifest)
    candidate_count = len(candidate_rows)
    commit_count = sum(1 for row in candidate_rows if bool(row.get("is_commit", False)))
    logged_selected_count = sum(1 for row in candidate_rows if bool(row.get("is_logged_selected", False)))
    commit_positive_count = sum(
        1
        for row in candidate_rows
        if bool(row.get("is_commit", False))
        and bool(row.get("is_logged_selected", False))
        and bool(row.get("is_commit_positive_state", False))
    )
    terminal_state_count = sum(
        1 for row in state_manifest if str(row.get("state_kind", "")).strip() == "terminal_commit"
    )

    split_state_counts: dict[str, int] = {}
    split_candidate_counts: dict[str, int] = {}
    for row in state_manifest:
        split = str(row.get("split", "train")).strip() or "train"
        split_state_counts[split] = split_state_counts.get(split, 0) + 1
    for row in candidate_rows:
        split = str(row.get("split", "train")).strip() or "train"
        split_candidate_counts[split] = split_candidate_counts.get(split, 0) + 1

    def _fraction(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    return {
        "state_count": state_count,
        "candidate_count": candidate_count,
        "commit_count": commit_count,
        "logged_selected_count": logged_selected_count,
        "commit_positive_count": commit_positive_count,
        "terminal_state_count": terminal_state_count,
        "commit_fraction": _fraction(commit_count, candidate_count),
        "logged_selected_fraction": _fraction(logged_selected_count, candidate_count),
        "mean_candidates_per_state": _fraction(candidate_count, state_count),
        "split_state_counts": split_state_counts,
        "split_candidate_counts": split_candidate_counts,
    }


def _jsonl_lines(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in rows)


def _readme_text(dataset_name: str) -> str:
    return "\n".join(
        [
            f"# {dataset_name}",
            "",
            "G2.5 candidate-slate dataset derived from frozen G1 state snapshots and G2 critic rows.",
            "",
            "Files:",
            "- candidate_dataset.jsonl",
            "- state_manifest.jsonl",
            "- candidate_schema.json",
            "- dataset_stats.json",
            "",
        ]
    )


def build_candidate_schema() -> dict[str, str]:
    return {
        "candidate_id": "str",
        "state_id": "str",
        "group_id": "str",
        "split": "str",
        "benchmark": "str",
        "instance_name": "str",
        "run_dir": "str",
        "step_index": "int",
        "round_name": "str",
        "role": "str",
        "state_kind": "str",
        "state_text": "str",
        "candidate_index": "int",
        "candidate_count": "int",
        "candidate_kind": "str",
        "candidate_target_ids": "list[str]",
        "candidate_payload": "object",
        "candidate_source": "str",
        "candidate_text": "str",
        "is_commit": "bool",
        "is_logged_selected": "bool",
        "is_commit_positive_state": "bool",
        "commit_supervision": "object",
        "targets": "object",
        "weak_local": "object",
        "native": "object",
        "label_availability": "object",
        "runtime_protocol": "str",
        "label_source": "str",
    }


def load_g2_critic_rows(g2_dataset_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(Path(g2_dataset_dir) / "critic_dataset.jsonl")


def _split_lookup(split_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in split_rows:
        group_id = str(row.get("group_id", "")).strip()
        if group_id:
            lookup[group_id] = str(row.get("split", "train")).strip() or "train"
    return lookup


def build_terminal_rows_from_g1(
    manifest_rows: Sequence[Mapping[str, Any]],
    terminal_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_run_dir = {
        str(row.get("run_dir", "")).strip(): row
        for row in manifest_rows
        if str(row.get("run_dir", "")).strip()
    }
    split_by_group = _split_lookup(split_rows)
    output_rows: list[dict[str, Any]] = []
    if not terminal_rows:
        return output_rows
    for terminal in terminal_rows:
        terminal_copy = dict(terminal)
        run_dir = str(terminal_copy.get("run_dir", "")).strip()
        manifest_row = manifest_by_run_dir.get(run_dir)
        if manifest_row is None:
            raise ValueError(f"Terminal state row references run_dir '{run_dir}' missing from run manifest.")
        group_id = make_group_id(manifest_row)
        split = split_by_group.get(group_id)
        if split is None:
            raise ValueError(f"Missing split assignment for terminal state group_id '{group_id}'.")
        label_package = package_labels_from_manifest_row(manifest_row)
        terminal_copy["group_id"] = group_id
        terminal_copy["split"] = split
        terminal_copy["weak_local"] = label_package["weak_local"]
        terminal_copy["native"] = label_package["native"]
        terminal_copy["label_availability"] = label_package["label_availability"]
        terminal_copy["targets"] = label_package["targets"]
        terminal_copy["state_kind"] = str(terminal_copy.get("state_kind", "")).strip() or "terminal_commit"
        terminal_copy["commit_supervision"] = _as_object_dict(terminal_copy.get("commit_supervision"))
        output_rows.append(terminal_copy)
    return output_rows


@dataclass(frozen=True)
class CandidateSlateDatasetBuildResult:
    dataset_dir: Path
    state_count: int
    candidate_count: int


@dataclass(frozen=True)
class ParallelTwoHeadDatasetBuildResult:
    dataset_dir: Path
    edit_state_count: int
    edit_candidate_count: int
    commit_state_count: int


def _group_parallel_rows_by_state(
    parallel_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in parallel_rows:
        state_id = str(row.get("state_id", "")).strip()
        if not state_id:
            raise ValueError("Parallel edit row is missing required state_id.")
        grouped.setdefault(state_id, []).append(dict(row))
    return grouped


def _validate_parallel_candidate_state_rows(
    state_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _safe_int(row.get("candidate_index")),
            str(row.get("candidate_id", "")).strip(),
        ),
    )
    if not ordered_rows:
        raise ValueError(f"Parallel state '{state_id}' is empty.")

    selected_count = sum(1 for row in ordered_rows if bool(row.get("is_logged_selected", False)))
    if selected_count != 1:
        raise ValueError(
            f"Parallel state '{state_id}' must have exactly one logged-selected candidate; found {selected_count}."
        )

    candidate_count = _safe_int(ordered_rows[0].get("candidate_count"))
    if candidate_count != len(ordered_rows):
        raise ValueError(
            f"Parallel state '{state_id}' candidate_count={candidate_count} does not match actual rows={len(ordered_rows)}."
        )

    selected_candidate_id = str(ordered_rows[0].get("selected_candidate_id", "")).strip()
    shared_fields = {
        "group_id": make_group_id(ordered_rows[0]),
        "benchmark": str(ordered_rows[0].get("benchmark", "unknown")).strip() or "unknown",
        "instance_name": str(ordered_rows[0].get("instance_name", "unknown")).strip() or "unknown",
        "run_dir": str(ordered_rows[0].get("run_dir", "")).strip(),
        "step_index": _safe_int(ordered_rows[0].get("parallel_state_index")),
        "round_name": str(ordered_rows[0].get("round_name", "")).strip(),
        "role": str(ordered_rows[0].get("role", "")).strip(),
        "state_kind": str(ordered_rows[0].get("state_kind", "parallel_pre_action")).strip() or "parallel_pre_action",
        "state_text": str(ordered_rows[0].get("state_text", "")).strip(),
        "candidate_count": candidate_count,
        "runtime_protocol": str(ordered_rows[0].get("runtime_protocol", "")).strip(),
        "label_source": str(ordered_rows[0].get("label_source", "")).strip(),
        "selected_candidate_id": selected_candidate_id,
    }
    for row in ordered_rows[1:]:
        if str(row.get("selected_candidate_id", "")).strip() != selected_candidate_id:
            raise ValueError(f"Parallel state '{state_id}' has inconsistent selected_candidate_id values.")
        if _safe_int(row.get("candidate_count")) != candidate_count:
            raise ValueError(f"Parallel state '{state_id}' has inconsistent candidate_count values.")
        if str(row.get("run_dir", "")).strip() != shared_fields["run_dir"]:
            raise ValueError(f"Parallel state '{state_id}' mixes multiple run_dir values.")

    return ordered_rows, shared_fields


def build_parallel_candidate_dataset_rows(
    *,
    g1_dataset_dir: Path,
    parallel_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_by_run_dir = {
        str(row.get("run_dir", "")).strip(): row
        for row in manifest_rows
        if str(row.get("run_dir", "")).strip()
    }
    split_by_group = _split_lookup(split_rows)
    candidate_rows: list[dict[str, Any]] = []
    state_manifest: list[dict[str, Any]] = []

    grouped = _group_parallel_rows_by_state(parallel_rows)
    for state_id in sorted(grouped):
        state_rows, shared = _validate_parallel_candidate_state_rows(state_id, grouped[state_id])
        run_dir = str(shared["run_dir"])
        manifest_row = manifest_by_run_dir.get(run_dir)
        if manifest_row is None:
            raise ValueError(f"Parallel state '{state_id}' references run_dir '{run_dir}' missing from run manifest.")
        group_id = make_group_id(manifest_row)
        split = split_by_group.get(group_id)
        if split is None:
            raise ValueError(f"Missing split assignment for parallel state group_id '{group_id}'.")
        label_package = package_labels_from_manifest_row(manifest_row)

        for row in state_rows:
            candidate_rows.append(
                {
                    "state_id": state_id,
                    "candidate_id": str(row.get("candidate_id", "")).strip(),
                    "group_id": group_id,
                    "split": split,
                    "benchmark": shared["benchmark"],
                    "instance_name": shared["instance_name"],
                    "run_dir": run_dir,
                    "step_index": shared["step_index"],
                    "round_name": shared["round_name"],
                    "role": shared["role"],
                    "state_kind": shared["state_kind"],
                    "state_text": shared["state_text"],
                    "candidate_index": _safe_int(row.get("candidate_index")),
                    "candidate_count": shared["candidate_count"],
                    "candidate_kind": str(row.get("candidate_kind", "")).strip(),
                    "candidate_target_ids": [str(item).strip() for item in row.get("candidate_target_ids", []) if str(item).strip()],
                    "candidate_payload": _as_object_dict(row.get("candidate_payload")),
                    "candidate_source": str(row.get("candidate_source", "")).strip(),
                    "candidate_text": str(row.get("candidate_text", "")).strip(),
                    "is_commit": False,
                    "is_logged_selected": bool(row.get("is_logged_selected", False)),
                    "is_commit_positive_state": False,
                    "commit_supervision": {},
                    "targets": label_package["targets"],
                    "weak_local": label_package["weak_local"],
                    "native": label_package["native"],
                    "label_availability": label_package["label_availability"],
                    "runtime_protocol": shared["runtime_protocol"],
                    "label_source": shared["label_source"],
                }
            )

        state_manifest.append(
            {
                "state_id": state_id,
                "group_id": group_id,
                "split": split,
                "benchmark": shared["benchmark"],
                "instance_name": shared["instance_name"],
                "run_dir": run_dir,
                "step_index": shared["step_index"],
                "round_name": shared["round_name"],
                "role": shared["role"],
                "state_kind": shared["state_kind"],
                "candidate_count": shared["candidate_count"],
                "runtime_protocol": shared["runtime_protocol"],
                "label_source": shared["label_source"],
            }
        )

    return candidate_rows, state_manifest


def _group_commit_rows_by_state(
    commit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in commit_rows:
        state_id = str(row.get("state_id", "")).strip()
        if not state_id:
            raise ValueError("Post-round commit row is missing required state_id.")
        grouped.setdefault(state_id, []).append(dict(row))
    return grouped


def _validate_parallel_commit_state_rows(
    state_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _safe_int(row.get("post_round_state_index")),
            str(row.get("state_id", "")).strip(),
        ),
    )
    if not ordered_rows:
        raise ValueError(f"Post-round commit state '{state_id}' is empty.")
    if len(ordered_rows) != 1:
        raise ValueError(f"Post-round commit state '{state_id}' must contain exactly one row.")

    row = ordered_rows[0]
    commit_supervision = _as_object_dict(row.get("commit_supervision"))
    if not bool(commit_supervision.get("available", False)):
        raise ValueError(f"Post-round commit state '{state_id}' is missing available commit supervision.")
    label = _safe_int(commit_supervision.get("label"))
    if label not in (0, 1):
        raise ValueError(f"Post-round commit state '{state_id}' must use binary commit supervision.")
    return row


def build_parallel_commit_dataset_rows(
    *,
    commit_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_by_run_dir = {
        str(row.get("run_dir", "")).strip(): row
        for row in manifest_rows
        if str(row.get("run_dir", "")).strip()
    }
    split_by_group = _split_lookup(split_rows)
    packaged_rows: list[dict[str, Any]] = []
    state_manifest: list[dict[str, Any]] = []

    grouped = _group_commit_rows_by_state(commit_rows)
    for state_id in sorted(grouped):
        row = _validate_parallel_commit_state_rows(state_id, grouped[state_id])
        run_dir = str(row.get("run_dir", "")).strip()
        manifest_row = manifest_by_run_dir.get(run_dir)
        if manifest_row is None:
            raise ValueError(
                f"Post-round commit state '{state_id}' references run_dir '{run_dir}' missing from run manifest."
            )
        group_id = make_group_id(manifest_row)
        split = split_by_group.get(group_id)
        if split is None:
            raise ValueError(f"Missing split assignment for post-round commit group_id '{group_id}'.")
        label_package = package_labels_from_manifest_row(manifest_row)
        commit_supervision = _as_object_dict(row.get("commit_supervision"))
        packaged_rows.append(
            {
                "state_id": state_id,
                "group_id": group_id,
                "split": split,
                "benchmark": str(row.get("benchmark", "unknown")).strip() or "unknown",
                "instance_name": str(row.get("instance_name", "unknown")).strip() or "unknown",
                "run_dir": run_dir,
                "step_index": _safe_int(row.get("post_round_state_index")),
                "round_name": str(row.get("round_name", "")).strip(),
                "role": str(row.get("role", "CommitController")).strip() or "CommitController",
                "state_kind": str(row.get("state_kind", "parallel_post_round")).strip() or "parallel_post_round",
                "state_text": str(row.get("state_text", "")).strip(),
                "before_state_snapshot": str(row.get("before_state_snapshot", "")).strip(),
                "commit_supervision": commit_supervision,
                "commit_label": _safe_int(commit_supervision.get("label")),
                "targets": label_package["targets"],
                "weak_local": label_package["weak_local"],
                "native": label_package["native"],
                "label_availability": label_package["label_availability"],
                "runtime_protocol": str(row.get("runtime_protocol", "")).strip(),
                "label_source": str(row.get("label_source", "")).strip(),
            }
        )
        state_manifest.append(
            {
                "state_id": state_id,
                "group_id": group_id,
                "split": split,
                "benchmark": str(row.get("benchmark", "unknown")).strip() or "unknown",
                "instance_name": str(row.get("instance_name", "unknown")).strip() or "unknown",
                "run_dir": run_dir,
                "step_index": _safe_int(row.get("post_round_state_index")),
                "round_name": str(row.get("round_name", "")).strip(),
                "role": str(row.get("role", "CommitController")).strip() or "CommitController",
                "state_kind": str(row.get("state_kind", "parallel_post_round")).strip() or "parallel_post_round",
                "runtime_protocol": str(row.get("runtime_protocol", "")).strip(),
                "label_source": str(row.get("label_source", "")).strip(),
            }
        )

    return packaged_rows, state_manifest


def build_parallel_candidate_dataset_from_export(
    *,
    g1_dataset_dir: Path,
    output_dir: Path,
    dataset_name: str,
    validation_fraction: float = 0.2,
    split_overrides_path: Path | None = None,
) -> CandidateSlateDatasetBuildResult:
    manifest_rows = _load_optional_jsonl(Path(g1_dataset_dir) / "run_manifest.jsonl")
    parallel_rows = _load_optional_jsonl(Path(g1_dataset_dir) / "parallel_edit_examples.jsonl")
    split_override_rows = _load_optional_jsonl(Path(split_overrides_path)) if split_overrides_path else []
    group_rows = build_group_manifest(manifest_rows, parallel_rows)
    split_rows = assign_group_splits(
        group_rows,
        validation_fraction=validation_fraction,
        split_override_rows=split_override_rows,
    )
    candidate_rows, state_manifest = build_parallel_candidate_dataset_rows(
        g1_dataset_dir=g1_dataset_dir,
        parallel_rows=parallel_rows,
        manifest_rows=manifest_rows,
        split_rows=split_rows,
    )
    candidate_schema = build_candidate_schema()
    dataset_stats = build_candidate_dataset_stats(candidate_rows, state_manifest)

    dataset_dir = Path(output_dir) / dataset_name
    write_text_file(dataset_dir / "candidate_dataset.jsonl", _jsonl_lines(candidate_rows))
    write_text_file(dataset_dir / "state_manifest.jsonl", _jsonl_lines(state_manifest))
    write_text_file(dataset_dir / "split_manifest.jsonl", _jsonl_lines(split_rows))
    write_text_file(
        dataset_dir / "candidate_schema.json",
        json.dumps(candidate_schema, indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(
        dataset_dir / "dataset_stats.json",
        json.dumps(dataset_stats, indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(dataset_dir / "README.md", _readme_text(dataset_name))

    return CandidateSlateDatasetBuildResult(
        dataset_dir=dataset_dir,
        state_count=len(state_manifest),
        candidate_count=len(candidate_rows),
    )


def _two_head_readme_text(dataset_name: str) -> str:
    return "\n".join(
        [
            f"# {dataset_name}",
            "",
            "Parallel two-head critic dataset derived from frozen edit states and post-round commit states.",
            "",
            "Files:",
            "- edit_head_rows.jsonl",
            "- edit_state_manifest.jsonl",
            "- commit_head_rows.jsonl",
            "- commit_state_manifest.jsonl",
            "- split_manifest.jsonl",
            "- dataset_stats.json",
            "",
        ]
    )


def build_parallel_two_head_dataset_from_export(
    *,
    g1_dataset_dir: Path,
    output_dir: Path,
    dataset_name: str,
    validation_fraction: float = 0.2,
    split_overrides_path: Path | None = None,
) -> ParallelTwoHeadDatasetBuildResult:
    manifest_rows = _load_optional_jsonl(Path(g1_dataset_dir) / "run_manifest.jsonl")
    parallel_rows = _load_optional_jsonl(Path(g1_dataset_dir) / "parallel_edit_examples.jsonl")
    commit_rows = _load_optional_jsonl(Path(g1_dataset_dir) / "post_round_commit_examples.jsonl")
    split_override_rows = _load_optional_jsonl(Path(split_overrides_path)) if split_overrides_path else []
    group_rows = build_group_manifest(manifest_rows, [*parallel_rows, *commit_rows])
    split_rows = assign_group_splits(
        group_rows,
        validation_fraction=validation_fraction,
        split_override_rows=split_override_rows,
    )
    edit_rows, edit_state_manifest = build_parallel_candidate_dataset_rows(
        g1_dataset_dir=g1_dataset_dir,
        parallel_rows=parallel_rows,
        manifest_rows=manifest_rows,
        split_rows=split_rows,
    )
    packaged_commit_rows, commit_state_manifest = build_parallel_commit_dataset_rows(
        commit_rows=commit_rows,
        manifest_rows=manifest_rows,
        split_rows=split_rows,
    )
    dataset_stats = {
        "edit_state_count": len(edit_state_manifest),
        "edit_candidate_count": len(edit_rows),
        "commit_state_count": len(commit_state_manifest),
        "commit_positive_count": sum(1 for row in packaged_commit_rows if _safe_int(row.get("commit_label")) == 1),
        "commit_continue_count": sum(1 for row in packaged_commit_rows if _safe_int(row.get("commit_label")) == 0),
        "split_counts": {
            split: sum(1 for row in edit_state_manifest if str(row.get("split", "")).strip() == split)
            + sum(1 for row in commit_state_manifest if str(row.get("split", "")).strip() == split)
            for split in sorted(
                {
                    *(str(row.get("split", "train")).strip() or "train" for row in edit_state_manifest),
                    *(str(row.get("split", "train")).strip() or "train" for row in commit_state_manifest),
                }
            )
        },
    }

    dataset_dir = Path(output_dir) / dataset_name
    write_text_file(dataset_dir / "edit_head_rows.jsonl", _jsonl_lines(edit_rows))
    write_text_file(dataset_dir / "edit_state_manifest.jsonl", _jsonl_lines(edit_state_manifest))
    write_text_file(dataset_dir / "commit_head_rows.jsonl", _jsonl_lines(packaged_commit_rows))
    write_text_file(dataset_dir / "commit_state_manifest.jsonl", _jsonl_lines(commit_state_manifest))
    write_text_file(dataset_dir / "split_manifest.jsonl", _jsonl_lines(split_rows))
    write_text_file(
        dataset_dir / "dataset_stats.json",
        json.dumps(dataset_stats, indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(dataset_dir / "README.md", _two_head_readme_text(dataset_name))

    return ParallelTwoHeadDatasetBuildResult(
        dataset_dir=dataset_dir,
        edit_state_count=len(edit_state_manifest),
        edit_candidate_count=len(edit_rows),
        commit_state_count=len(commit_state_manifest),
    )


def build_graph_critic_candidate_dataset(
    *,
    g1_dataset_dir: Path,
    g2_dataset_dir: Path,
    output_dir: Path,
    dataset_name: str,
) -> CandidateSlateDatasetBuildResult:
    critic_rows = load_g2_critic_rows(g2_dataset_dir)
    raw_terminal_rows = _load_optional_jsonl(Path(g1_dataset_dir) / "terminal_state_manifest.jsonl")
    manifest_rows = _load_optional_jsonl(Path(g1_dataset_dir) / "run_manifest.jsonl")
    split_rows = _load_optional_jsonl(Path(g2_dataset_dir) / "split_manifest.jsonl")
    terminal_rows = build_terminal_rows_from_g1(
        manifest_rows,
        raw_terminal_rows,
        split_rows,
    )
    candidate_rows, state_manifest = build_candidate_dataset_rows(
        g1_dataset_dir=g1_dataset_dir,
        critic_rows=critic_rows,
        terminal_rows=terminal_rows,
    )
    candidate_schema = build_candidate_schema()
    dataset_stats = build_candidate_dataset_stats(candidate_rows, state_manifest)

    dataset_dir = Path(output_dir) / dataset_name
    write_text_file(dataset_dir / "candidate_dataset.jsonl", _jsonl_lines(candidate_rows))
    write_text_file(dataset_dir / "state_manifest.jsonl", _jsonl_lines(state_manifest))
    write_text_file(
        dataset_dir / "candidate_schema.json",
        json.dumps(candidate_schema, indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(
        dataset_dir / "dataset_stats.json",
        json.dumps(dataset_stats, indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(dataset_dir / "README.md", _readme_text(dataset_name))

    return CandidateSlateDatasetBuildResult(
        dataset_dir=dataset_dir,
        state_count=len(state_manifest),
        candidate_count=len(candidate_rows),
    )
