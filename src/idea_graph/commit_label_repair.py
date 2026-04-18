from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fs_utils import read_text_file


@dataclass(frozen=True)
class OutcomeGroundedCommitConfig:
    commit_margin: float = 0.15
    continue_margin: float = 0.35
    positive_support_threshold: float = 0.75
    positive_unresolved_threshold: float = 0.05
    positive_utility_threshold: float = 7.0
    minimum_positive_signals: int = 2


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_object_dict(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _group_rows_by_run(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        run_dir = str(row.get("run_dir", "")).strip() or str(row.get("group_id", "")).strip()
        grouped.setdefault(run_dir, []).append(dict(row))
    return grouped


def _positive_signal_count(row: Mapping[str, Any], config: OutcomeGroundedCommitConfig) -> tuple[int, dict[str, bool]]:
    flags = {
        "support_ready": _safe_float(row.get("support_coverage")) >= config.positive_support_threshold,
        "contradictions_resolved": _safe_float(row.get("unresolved_contradiction_ratio")) <= config.positive_unresolved_threshold,
        "utility_ready": _safe_float(row.get("utility")) >= config.positive_utility_threshold,
    }
    return sum(1 for value in flags.values() if value), flags


def relabel_scored_commit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: OutcomeGroundedCommitConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active_config = config or OutcomeGroundedCommitConfig()
    grouped = _group_rows_by_run(rows)
    repaired_rows: list[dict[str, Any]] = []
    original_positive_count = 0
    repaired_positive_count = 0
    repaired_continue_count = 0
    ambiguous_drop_count = 0
    flipped_negative_to_positive_count = 0
    flipped_positive_to_negative_count = 0
    future_gains: list[float] = []
    runs_with_positive_boundary_count = 0

    for _run_dir, run_rows in sorted(grouped.items()):
        ordered_rows = sorted(
            (dict(row) for row in run_rows),
            key=lambda row: (_safe_int(row.get("post_round_state_index")), str(row.get("state_id", ""))),
        )
        scores = [_safe_float(row.get("outcome_local_overall")) for row in ordered_rows]
        episode_best = max(scores) if scores else 0.0
        positive_candidates: list[bool] = []
        candidate_future_gains: list[float] = []
        candidate_signal_counts: list[int] = []
        candidate_positive_flags: list[dict[str, bool]] = []
        for index, row in enumerate(ordered_rows):
            current_score = scores[index]
            later_scores = scores[index + 1 :]
            future_best = max(later_scores) if later_scores else current_score
            future_gain = max(0.0, future_best - current_score)
            positive_signal_count, positive_flags = _positive_signal_count(row, active_config)
            candidate_future_gains.append(future_gain)
            candidate_signal_counts.append(positive_signal_count)
            candidate_positive_flags.append(positive_flags)
            positive_candidates.append(
                future_gain <= active_config.commit_margin
                and positive_signal_count >= active_config.minimum_positive_signals
            )
        positive_boundary_index = next(
            (index for index, is_positive_candidate in enumerate(positive_candidates) if is_positive_candidate),
            None,
        )
        if positive_boundary_index is not None:
            runs_with_positive_boundary_count += 1
        for index, row in enumerate(ordered_rows):
            original_supervision = _as_object_dict(row.get("commit_supervision"))
            original_label = _safe_int(original_supervision.get("label"))
            if original_label == 1:
                original_positive_count += 1

            current_score = scores[index]
            later_scores = scores[index + 1 :]
            future_best = max(later_scores) if later_scores else current_score
            future_gain = candidate_future_gains[index]
            positive_signal_count = candidate_signal_counts[index]
            positive_flags = candidate_positive_flags[index]
            future_gains.append(future_gain)

            if positive_boundary_index is not None and index < positive_boundary_index:
                available = True
                repaired_label = 0
                status = "continue_before_positive_boundary"
            elif positive_candidates[index]:
                available = True
                repaired_label = 1
                status = "commit_competitive_and_structured"
            elif future_gain >= active_config.continue_margin:
                available = True
                repaired_label = 0
                status = "continue_future_improves"
            else:
                available = False
                repaired_label = 0
                status = "ambiguous_boundary"

            if available and repaired_label == 1:
                repaired_positive_count += 1
            elif available:
                repaired_continue_count += 1
            else:
                ambiguous_drop_count += 1
            if original_label == 0 and available and repaired_label == 1:
                flipped_negative_to_positive_count += 1
            if original_label == 1 and (not available or repaired_label == 0):
                flipped_positive_to_negative_count += 1

            repaired_row = dict(row)
            repaired_row["logged_commit_supervision"] = original_supervision
            repaired_row["label_source"] = "outcome_grounded_local_eval"
            repaired_row["commit_supervision"] = {
                "available": available,
                "label": repaired_label,
                "source": "outcome_grounded_local_eval",
                "logged_label": original_label,
                "repair_status": status,
            }
            repaired_row["outcome_grounded_commit_label"] = {
                "available": available,
                "label": repaired_label,
                "status": status,
                "current_local_overall": round(current_score, 4),
                "future_best_local_overall": round(future_best, 4),
                "episode_best_local_overall": round(episode_best, 4),
                "future_improvement": round(future_gain, 4),
                "positive_signal_count": positive_signal_count,
                "positive_flags": positive_flags,
                "config": asdict(active_config),
            }
            repaired_rows.append(repaired_row)

    available_commit_state_count = repaired_positive_count + repaired_continue_count
    audit = {
        "commit_label_mode": "outcome_grounded",
        "input_commit_state_count": len(rows),
        "available_commit_state_count": available_commit_state_count,
        "original_positive_count": original_positive_count,
        "repaired_positive_count": repaired_positive_count,
        "repaired_continue_count": repaired_continue_count,
        "ambiguous_drop_count": ambiguous_drop_count,
        "flipped_negative_to_positive_count": flipped_negative_to_positive_count,
        "flipped_positive_to_negative_count": flipped_positive_to_negative_count,
        "mean_future_improvement": (
            round(sum(future_gains) / len(future_gains), 4) if future_gains else 0.0
        ),
        "runs_with_positive_boundary_count": runs_with_positive_boundary_count,
        "config": asdict(active_config),
    }
    return repaired_rows, audit


def _load_run_context(row: Mapping[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    run_dir = Path(str(row.get("run_dir", "")).strip())
    graph_path = run_dir / "graph.json"
    if graph_path.exists():
        payload = json.loads(read_text_file(graph_path))
        if isinstance(payload, Mapping):
            metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {}
            topic = str(payload.get("topic", row.get("topic", ""))).strip()
            literature = [
                str(item).strip()
                for item in payload.get("literature", [])
                if str(item).strip()
            ] if isinstance(payload.get("literature"), list) else []
            return topic, literature, metadata

    return str(row.get("topic", "")).strip(), [], {}


def score_post_round_commit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    g1_dataset_dir: Path,
) -> list[dict[str, Any]]:
    from .candidate_slate_dataset import graph_from_state_snapshot
    from .engine import select_final_subgraph, synthesize_proposal
    from .evaluation import evaluate_graph

    scored_rows: list[dict[str, Any]] = []
    run_context_cache: dict[str, tuple[str, list[str], dict[str, Any]]] = {}
    for row in rows:
        row_copy = dict(row)
        relative_snapshot = str(row.get("before_state_snapshot", "")).strip()
        if not relative_snapshot:
            row_copy["outcome_scoring_error"] = "missing before_state_snapshot"
            row_copy["outcome_local_overall"] = 0.0
            scored_rows.append(row_copy)
            continue
        snapshot_path = Path(g1_dataset_dir) / relative_snapshot
        try:
            snapshot_payload = json.loads(read_text_file(snapshot_path))
            if not isinstance(snapshot_payload, Mapping):
                raise ValueError("snapshot is not a JSON object")
            run_key = str(row.get("run_dir", "")).strip()
            if run_key not in run_context_cache:
                run_context_cache[run_key] = _load_run_context(row)
            topic, literature, metadata = run_context_cache[run_key]
            graph = graph_from_state_snapshot(
                dict(snapshot_payload),
                topic=topic or str(row.get("topic", "")).strip(),
                literature=literature,
                role="CommitController",
                round_name=str(row.get("round_name", "")).strip(),
            )
            graph.metadata.update(metadata)
            graph.metadata["benchmark"] = str(row.get("benchmark", graph.metadata.get("benchmark", ""))).strip()
            graph.metadata["instance_name"] = str(
                row.get("instance_name", graph.metadata.get("instance_name", ""))
            ).strip()
            graph.final_subgraph = select_final_subgraph(graph)
            graph.final_proposal = synthesize_proposal(graph, graph.final_subgraph)
            evaluation = evaluate_graph(graph)
            row_copy["outcome_local_overall"] = float(evaluation.overall_score)
            row_copy["outcome_local_category_scores"] = dict(evaluation.category_scores)
            row_copy["outcome_synthesis_mode"] = "deterministic_snapshot"
        except Exception as exc:
            row_copy["outcome_scoring_error"] = str(exc)
            row_copy["outcome_local_overall"] = 0.0
        scored_rows.append(row_copy)
    return scored_rows


def score_and_relabel_commit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    g1_dataset_dir: Path,
    config: OutcomeGroundedCommitConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored_rows = score_post_round_commit_rows(rows, g1_dataset_dir=g1_dataset_dir)
    repaired_rows, audit = relabel_scored_commit_rows(scored_rows, config=config)
    audit["scoring_error_count"] = sum(1 for row in repaired_rows if str(row.get("outcome_scoring_error", "")).strip())
    return repaired_rows, audit
