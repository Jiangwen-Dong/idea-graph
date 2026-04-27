from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .collaboration_protocol import resolve_round_phase
from .fs_utils import ensure_dir, read_text_file, write_text_file


@dataclass(frozen=True)
class CuratedDatasetQuotas:
    edit_phase_kind: dict[tuple[str, str], int] = field(default_factory=dict)
    commit_round_label: dict[tuple[str, int], int] = field(default_factory=dict)


@dataclass(frozen=True)
class CuratedDatasetResult:
    dataset_dir: Path
    edit_label_count: int
    commit_label_count: int


DEFAULT_CURATED_DATASET_QUOTAS = CuratedDatasetQuotas(
    edit_phase_kind={
        ("structure", "add_support_edge"): 14,
        ("structure", "add_contradiction_edge"): 14,
        ("structure", "add_dependency_edge"): 20,
        ("structure", "request_evidence"): 12,
        ("structure", "mark_overlap"): 8,
        ("structure", "skip"): 4,
        ("stress_test", "attach_evidence"): 18,
        ("stress_test", "request_evidence"): 10,
        ("stress_test", "add_contradiction_edge"): 8,
        ("stress_test", "mark_overlap"): 12,
        ("stress_test", "add_support_edge"): 8,
        ("stress_test", "skip"): 8,
        ("repair", "propose_repair"): 24,
        ("repair", "attach_evidence"): 10,
        ("repair", "add_support_edge"): 6,
        ("repair", "skip"): 16,
    },
    commit_round_label={
        ("Round2", 0): 16,
        ("Round2", 1): 4,
        ("Round3", 0): 10,
        ("Round3", 1): 18,
        ("Round4", 0): 6,
        ("Round4", 1): 8,
        ("Round5", 1): 2,
    },
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, raw_line in enumerate(read_text_file(path).splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_index} must contain a JSON object.")
        rows.append(dict(payload))
    return rows


def _jsonl_lines(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)


def _stable_sort_key(*parts: object) -> str:
    digest = hashlib.sha256("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest


def _phase_key(round_name: str) -> str:
    return resolve_round_phase(round_name).key


def _commit_label(row: Mapping[str, Any]) -> int:
    value = row.get("commit_label")
    if isinstance(value, Mapping):
        value = value.get("label")
    try:
        return int(value)
    except (TypeError, ValueError):
        supervision = row.get("commit_supervision")
        if isinstance(supervision, Mapping):
            try:
                return int(supervision.get("label", 0))
            except (TypeError, ValueError):
                return 0
        return 0


def _group_edit_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("candidate_kind", "")).strip() == "freeze_branch":
            continue
        state_id = str(row.get("state_id", "")).strip()
        if state_id:
            grouped[state_id].append(dict(row))
    return grouped


def _candidate_kind_to_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("candidate_kind", "")).strip()].append(dict(row))
    return grouped


def _select_edit_states(
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    quotas: CuratedDatasetQuotas,
) -> dict[str, str]:
    selected_state_to_kind: dict[str, str] = {}
    ordered_quotas = sorted(quotas.edit_phase_kind.items(), key=lambda item: (item[0][0], item[0][1]))
    for (phase_key, candidate_kind), required_count in ordered_quotas:
        eligible: list[tuple[tuple[int, str], str]] = []
        for state_id, rows in grouped_rows.items():
            if state_id in selected_state_to_kind:
                continue
            if not rows:
                continue
            round_name = str(rows[0].get("round_name", "")).strip()
            if _phase_key(round_name) != phase_key:
                continue
            by_kind = _candidate_kind_to_rows(rows)
            candidates = by_kind.get(candidate_kind, [])
            if not candidates:
                continue
            original_match = 0 if any(bool(row.get("is_logged_selected", False)) for row in candidates) else 1
            eligible.append(((original_match, _stable_sort_key(phase_key, candidate_kind, state_id)), state_id))
        if len(eligible) < required_count:
            raise ValueError(
                f"Not enough edit states for phase='{phase_key}', kind='{candidate_kind}'. "
                f"Required {required_count}, found {len(eligible)}."
            )
        for _sort_key, state_id in sorted(eligible)[:required_count]:
            selected_state_to_kind[state_id] = candidate_kind
    return selected_state_to_kind


def _select_commit_rows(
    rows: Sequence[Mapping[str, Any]],
    quotas: CuratedDatasetQuotas,
) -> list[dict[str, Any]]:
    used_state_ids: set[str] = set()
    selected_rows: list[dict[str, Any]] = []
    ordered_quotas = sorted(quotas.commit_round_label.items(), key=lambda item: (item[0][0], item[0][1]))
    for (round_name, label), required_count in ordered_quotas:
        eligible: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            state_id = str(row.get("state_id", "")).strip()
            if not state_id or state_id in used_state_ids:
                continue
            if str(row.get("round_name", "")).strip() != round_name:
                continue
            if _commit_label(row) != label:
                continue
            eligible.append((_stable_sort_key(round_name, label, state_id), dict(row)))
        if len(eligible) < required_count:
            raise ValueError(
                f"Not enough commit rows for round='{round_name}', label='{label}'. "
                f"Required {required_count}, found {len(eligible)}."
            )
        for _sort_key, row in sorted(eligible)[:required_count]:
            state_id = str(row.get("state_id", "")).strip()
            used_state_ids.add(state_id)
            selected_rows.append(row)
    return selected_rows


def _curate_edit_rows(
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_state_to_kind: Mapping[str, str],
) -> list[dict[str, Any]]:
    curated_rows: list[dict[str, Any]] = []
    for state_id in sorted(selected_state_to_kind):
        selected_kind = str(selected_state_to_kind[state_id]).strip()
        state_rows = [dict(row) for row in grouped_rows[state_id]]
        state_rows.sort(
            key=lambda row: (
                int(row.get("candidate_index", 0)),
                str(row.get("candidate_id", "")),
            )
        )
        selected_candidate_id = ""
        for row in state_rows:
            if str(row.get("candidate_kind", "")).strip() == selected_kind:
                selected_candidate_id = str(row.get("candidate_id", "")).strip()
                break
        if not selected_candidate_id:
            raise ValueError(f"State '{state_id}' is missing selected kind '{selected_kind}'.")
        total_candidates = len(state_rows)
        for index, row in enumerate(state_rows):
            row["original_is_logged_selected"] = bool(row.get("is_logged_selected", False))
            row["is_logged_selected"] = str(row.get("candidate_id", "")).strip() == selected_candidate_id
            row["candidate_index"] = index
            row["candidate_count"] = total_candidates
            row["curation_source"] = "gold256_phase_quota_v1"
            row["curated_positive_kind"] = selected_kind
            row["curation_phase"] = _phase_key(str(row.get("round_name", "")).strip())
            curated_rows.append(row)
    return curated_rows


def _curate_commit_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    curated_rows: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["curation_source"] = "gold256_commit_quota_v1"
        payload["curated_commit_label"] = _commit_label(payload)
        curated_rows.append(payload)
    return curated_rows


def _dataset_stats(edit_rows: Sequence[Mapping[str, Any]], commit_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    edit_state_ids = {str(row.get("state_id", "")).strip() for row in edit_rows if str(row.get("state_id", "")).strip()}
    commit_state_ids = {str(row.get("state_id", "")).strip() for row in commit_rows if str(row.get("state_id", "")).strip()}
    split_counts: Counter[str] = Counter()
    for row in [*edit_rows, *commit_rows]:
        split_counts[str(row.get("split", "train")).strip() or "train"] += 1
    return {
        "edit_state_count": len(edit_state_ids),
        "edit_candidate_count": len(edit_rows),
        "edit_positive_count": sum(1 for row in edit_rows if bool(row.get("is_logged_selected", False))),
        "commit_state_count": len(commit_state_ids),
        "commit_positive_count": sum(1 for row in commit_rows if _commit_label(row) == 1),
        "commit_continue_count": sum(1 for row in commit_rows if _commit_label(row) == 0),
        "split_counts": dict(sorted(split_counts.items())),
    }


def _audit_payload(
    *,
    source_dataset_dir: Path,
    curated_edit_rows: Sequence[Mapping[str, Any]],
    curated_commit_rows: Sequence[Mapping[str, Any]],
    quotas: CuratedDatasetQuotas,
) -> dict[str, Any]:
    edit_positive_counts: Counter[str] = Counter(
        str(row.get("candidate_kind", "")).strip()
        for row in curated_edit_rows
        if bool(row.get("is_logged_selected", False))
    )
    commit_label_counts: Counter[str] = Counter(str(_commit_label(row)) for row in curated_commit_rows)
    role_counts: Counter[str] = Counter(
        str(row.get("role", "")).strip()
        for row in curated_edit_rows
        if bool(row.get("is_logged_selected", False))
    )
    phase_counts: Counter[str] = Counter(
        str(row.get("curation_phase", "")).strip()
        for row in curated_edit_rows
        if bool(row.get("is_logged_selected", False))
    )
    return {
        "source_dataset_dir": str(source_dataset_dir.resolve()),
        "edit_label_count": sum(edit_positive_counts.values()),
        "commit_label_count": len(curated_commit_rows),
        "edit_positive_counts": dict(sorted(edit_positive_counts.items())),
        "commit_label_counts": dict(sorted(commit_label_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "requested_edit_phase_kind_quotas": {
            f"{phase}::{kind}": count for (phase, kind), count in sorted(quotas.edit_phase_kind.items())
        },
        "requested_commit_round_label_quotas": {
            f"{round_name}::{label}": count
            for (round_name, label), count in sorted(quotas.commit_round_label.items())
        },
    }


def curate_two_head_critic_dataset(
    *,
    dataset_dir: Path,
    output_dir: Path,
    dataset_name: str,
    quotas: CuratedDatasetQuotas = DEFAULT_CURATED_DATASET_QUOTAS,
) -> CuratedDatasetResult:
    if not dataset_name.strip():
        raise ValueError("dataset_name must not be empty.")
    source_dir = Path(dataset_dir)
    edit_rows = _load_jsonl(source_dir / "edit_head_rows.jsonl")
    commit_rows = _load_jsonl(source_dir / "commit_head_rows.jsonl")
    grouped_edit_rows = _group_edit_rows(edit_rows)
    selected_state_to_kind = _select_edit_states(grouped_edit_rows, quotas)
    curated_edit_rows = _curate_edit_rows(grouped_edit_rows, selected_state_to_kind)
    curated_commit_rows = _curate_commit_rows(_select_commit_rows(commit_rows, quotas))
    dataset_output_dir = Path(output_dir) / dataset_name
    ensure_dir(dataset_output_dir)
    write_text_file(dataset_output_dir / "edit_head_rows.jsonl", _jsonl_lines(curated_edit_rows))
    write_text_file(dataset_output_dir / "commit_head_rows.jsonl", _jsonl_lines(curated_commit_rows))
    stats = _dataset_stats(curated_edit_rows, curated_commit_rows)
    audit = _audit_payload(
        source_dataset_dir=source_dir,
        curated_edit_rows=curated_edit_rows,
        curated_commit_rows=curated_commit_rows,
        quotas=quotas,
    )
    write_text_file(dataset_output_dir / "dataset_stats.json", json.dumps(stats, indent=2, ensure_ascii=False))
    write_text_file(dataset_output_dir / "curation_audit.json", json.dumps(audit, indent=2, ensure_ascii=False))
    write_text_file(
        dataset_output_dir / "README.md",
        (
            f"# {dataset_name}\n\n"
            "Curated two-head critic dataset derived from an existing parallel two-head package.\n\n"
            f"- edit_label_count: {audit['edit_label_count']}\n"
            f"- commit_label_count: {audit['commit_label_count']}\n"
            "- curation_source: gold256_phase_quota_v1 / gold256_commit_quota_v1\n"
        ),
    )
    return CuratedDatasetResult(
        dataset_dir=dataset_output_dir,
        edit_label_count=int(audit["edit_label_count"]),
        commit_label_count=int(audit["commit_label_count"]),
    )


__all__ = [
    "CuratedDatasetQuotas",
    "CuratedDatasetResult",
    "DEFAULT_CURATED_DATASET_QUOTAS",
    "curate_two_head_critic_dataset",
]
