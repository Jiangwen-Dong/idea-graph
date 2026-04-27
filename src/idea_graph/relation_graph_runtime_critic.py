from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .critic_policy import (
    CriticPolicyDecision,
    SafeCriticPolicyConfig,
    ScoredCandidate,
    choose_critic_action,
    commit_threshold_for_round,
)
from .models import IdeaGraph
from .relation_graph_critic_data import (
    HashTextEmbeddingBackend,
    RelationGraphBatch,
    RelationGraphRuntimeBatch,
    RelationGraphRuntimeRowDiagnostics,
    RelationGraphVocabularies,
    SentenceTransformerEmbeddingBackend,
    build_relation_graph_runtime_batch,
    build_relation_graph_vocabularies,
)
from .relation_graph_critic_model import RelationGraphCritic


@dataclass(frozen=True)
class RelationGraphRuntimeConfig:
    tau_override: float = 0.05
    tau_override_by_round: Mapping[int | str, float] | None = None
    tau_commit: float = 0.08
    gamma_commit: float = 0.50
    gamma_commit_by_round: Mapping[int | str, float] | None = None
    min_commit_round: int = 3
    use_edit: bool = True
    use_commit: bool = False
    use_low_signal_kind_swap_guard: bool = False
    guard_support_threshold: float = 0.66
    guard_support_gain_floor: float = 0.10
    guard_requires_contradiction_progress: bool = False
    guard_commit_support_threshold: float = 0.0
    guard_commit_utility_floor: float = 0.0
    use_action_score_calibration: bool = False
    action_score_calibration_strength: float = 0.35
    action_score_calibration_max_bias: float = 0.35


@dataclass(frozen=True)
class RelationGraphRuntimeDecision:
    selected_spec: dict[str, object]
    policy_decision: CriticPolicyDecision
    scored_candidates: tuple[dict[str, object], ...]


def _parse_round_index(round_name: str) -> int:
    text = str(round_name).strip()
    if not text.lower().startswith("round"):
        return 0
    suffix = text[5:]
    try:
        return int(suffix)
    except ValueError:
        return 0


def _normalized_candidate_specs(
    candidate_specs: Sequence[Mapping[str, Any]],
    *,
    use_commit: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(candidate_specs):
        kind = str(spec.get("kind", spec.get("candidate_kind", ""))).strip() or "unknown"
        if kind == "commit" and not use_commit:
            continue
        row = dict(spec)
        row["kind"] = kind
        row["candidate_id"] = (
            str(spec.get("candidate_id", f"runtime-candidate:{index:03d}")).strip()
            or f"runtime-candidate:{index:03d}"
        )
        rows.append(row)
    return rows


def _candidate_float(spec: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(spec.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _candidate_after_is_mature(spec: Mapping[str, Any]) -> bool:
    if "after_is_mature" in spec:
        return bool(spec.get("after_is_mature"))
    after_subgraph = spec.get("after_subgraph")
    if not isinstance(after_subgraph, Mapping):
        return False
    return bool(after_subgraph.get("is_mature", False))


def _action_family(kind: str) -> str:
    normalized = str(kind).strip()
    if normalized in {"add_support_edge", "add_dependency_edge"}:
        return "support"
    if normalized in {"request_evidence", "attach_evidence"}:
        return "grounding"
    if normalized in {"add_contradiction_edge", "mark_overlap"}:
        return "critique"
    if normalized == "propose_repair":
        return "repair"
    if normalized in {"skip", "freeze_branch"}:
        return "abstain"
    if normalized == "commit":
        return "commit"
    return "other"


def _aligned_signal(kind: str) -> str:
    normalized = str(kind).strip()
    if normalized == "add_support_edge":
        return "support"
    if normalized == "add_dependency_edge":
        return "dependency"
    if normalized in {"request_evidence", "attach_evidence"}:
        return "grounding"
    if normalized == "add_contradiction_edge":
        return "challenge"
    if normalized == "mark_overlap":
        return "positioning"
    if normalized == "propose_repair":
        return "repair"
    if normalized in {"skip", "freeze_branch", "commit"}:
        return "maturity"
    return "other"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_state_float(
    state: Mapping[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    try:
        return float(state.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _safe_state_bool(state: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = state.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _optional_breakdown_float(
    state: Mapping[str, Any],
    key: str,
) -> float | None:
    breakdown = state.get("utility_breakdown")
    value: Any = None
    if isinstance(breakdown, Mapping):
        value = breakdown.get(key)
    elif breakdown is not None and hasattr(breakdown, key):
        value = getattr(breakdown, key)
    if value is None:
        value = state.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _deficit(value: float | None, target: float) -> float:
    if value is None or target <= 0.0:
        return 0.0
    return _clamp((target - float(value)) / target, 0.0, 1.0)


def _round_phase_key(round_index: int) -> str:
    if round_index <= 1:
        return "structure"
    if round_index == 2:
        return "stress_test"
    return "repair"


_GRAPH_SIGNAL_TARGETS: dict[str, float] = {
    "support": 0.62,
    "dependency": 0.52,
    "grounding": 0.36,
    "challenge": 0.30,
    "positioning": 0.28,
    "repair": 0.46,
    "completeness": 0.76,
    "maturity": 0.60,
}


def _graph_signal_feedback(
    graph: IdeaGraph,
    state_features: Mapping[str, Any] | None,
) -> dict[str, object]:
    state = dict(state_features or {})
    active_nodes = [
        node for node in graph.nodes.values() if getattr(node, "status", "active") == "active"
    ]
    active_node_ids = {str(node.id) for node in active_nodes}
    relevant_edges = [
        edge
        for edge in graph.edges
        if str(edge.source_id) in active_node_ids and str(edge.target_id) in active_node_ids
    ]
    active_types = {str(node.type) for node in active_nodes}

    tracked_nodes = [
        node
        for node in active_nodes
        if str(node.type) in {"Problem", "Hypothesis", "Method", "EvalPlan", "NoveltyClaim"}
    ]
    support_nodes = [
        node
        for node in active_nodes
        if str(node.type) in {"Hypothesis", "Method", "NoveltyClaim", "EvalPlan"}
    ]
    support_node_ids = {str(node.id) for node in support_nodes}
    support_edges = [edge for edge in relevant_edges if str(edge.relation) == "supports"]
    supported_count = 0
    for node in support_nodes:
        incoming = [edge for edge in relevant_edges if str(edge.target_id) == str(node.id)]
        if any(str(edge.relation) in {"supports", "repairs"} for edge in incoming) or list(node.evidence):
            supported_count += 1
    support_graph_coverage = (supported_count / len(support_nodes)) if support_nodes else 1.0
    support_edge_density = (
        _clamp(
            len(
                [
                    edge
                    for edge in support_edges
                    if str(edge.target_id) in support_node_ids or str(edge.source_id) in support_node_ids
                ]
            )
            / max(1, len(support_nodes)),
            0.0,
            1.0,
        )
        if support_nodes
        else 1.0
    )
    support_prior = max(
        _safe_state_float(state, "support_coverage", 0.0),
        _optional_breakdown_float(state, "support") or 0.0,
    )
    support_value = _clamp(
        (0.45 * support_graph_coverage)
        + (0.25 * support_edge_density)
        + (0.30 * support_prior),
        0.0,
        1.0,
    )

    dependency_nodes = [
        node
        for node in active_nodes
        if str(node.type) in {"Hypothesis", "Method", "EvalPlan"}
    ]
    primary_dependency_nodes = [
        node for node in dependency_nodes if str(node.type) in {"Method", "EvalPlan"}
    ]
    secondary_dependency_nodes = [
        node for node in dependency_nodes if str(node.type) == "Hypothesis"
    ]
    dependency_edges = [
        edge for edge in relevant_edges if str(edge.relation) == "depends_on"
    ]
    dependency_touched_ids = {str(edge.source_id) for edge in dependency_edges} | {
        str(edge.target_id) for edge in dependency_edges
    }
    primary_dependency_coverage = (
        len(
            {
                str(node.id)
                for node in primary_dependency_nodes
                if str(node.id) in dependency_touched_ids
            }
        )
        / len(primary_dependency_nodes)
        if primary_dependency_nodes
        else 1.0
    )
    secondary_dependency_coverage = (
        len(
            {
                str(node.id)
                for node in secondary_dependency_nodes
                if str(node.id) in dependency_touched_ids
            }
        )
        / len(secondary_dependency_nodes)
        if secondary_dependency_nodes
        else 1.0
    )
    dependency_density = (
        _clamp(
            len(dependency_edges) / max(1, len(primary_dependency_nodes)),
            0.0,
            1.0,
        )
        if dependency_nodes
        else 1.0
    )
    dependency_value = _clamp(
        (0.55 * primary_dependency_coverage)
        + (0.20 * secondary_dependency_coverage)
        + (0.25 * dependency_density),
        0.0,
        1.0,
    )

    grounded_count = sum(1 for node in tracked_nodes if list(node.evidence))
    requires_evidence_edges = [
        edge for edge in relevant_edges if str(edge.relation) == "requires_evidence"
    ]
    requested_grounding_ids = {
        str(edge.target_id)
        for edge in requires_evidence_edges
        if str(edge.target_id) in {str(node.id) for node in tracked_nodes}
    }
    grounded_coverage = (
        grounded_count / len(tracked_nodes)
        if tracked_nodes
        else 1.0
    )
    grounding_request_coverage = (
        len(requested_grounding_ids) / len(tracked_nodes)
        if tracked_nodes
        else 1.0
    )
    grounding_prior = _optional_breakdown_float(state, "evidence")
    if grounding_prior is None:
        grounding_value = _clamp(
            (0.75 * grounded_coverage) + (0.25 * grounding_request_coverage),
            0.0,
            1.0,
        )
    else:
        grounding_value = _clamp(
            (0.45 * grounded_coverage)
            + (0.20 * grounding_request_coverage)
            + (0.35 * grounding_prior),
            0.0,
            1.0,
        )

    contradiction_edges = [
        edge for edge in relevant_edges if str(edge.relation) == "contradicts"
    ]
    unresolved_contradiction_ratio = _safe_state_float(
        state,
        "unresolved_contradiction_ratio",
        (
            sum(1 for edge in contradiction_edges if not bool(getattr(edge, "resolved", False)))
            / len(contradiction_edges)
        )
        if contradiction_edges
        else 0.0,
    )
    challenge_nodes = [
        node
        for node in active_nodes
        if str(node.type)
        in {"Hypothesis", "Method", "NoveltyClaim", "EvalPlan", "Risk", "Assumption"}
    ]
    critique_anchor_nodes = [
        node for node in active_nodes if str(node.type) in {"Risk", "Assumption"}
    ]
    challenge_node_ids = {str(node.id) for node in challenge_nodes}
    contradiction_node_ids = {str(edge.source_id) for edge in contradiction_edges} | {
        str(edge.target_id) for edge in contradiction_edges
    }
    challenge_coverage = (
        len({node_id for node_id in contradiction_node_ids if node_id in challenge_node_ids})
        / len(challenge_nodes)
        if challenge_nodes
        else 1.0
    )
    contradiction_presence = _clamp(len(contradiction_edges) / 2.0, 0.0, 1.0)
    critique_anchor_presence = _clamp(len(critique_anchor_nodes) / 2.0, 0.0, 1.0)
    challenge_value = _clamp(
        (0.25 * critique_anchor_presence)
        + (0.50 * contradiction_presence)
        + (0.25 * challenge_coverage),
        0.0,
        1.0,
    )

    positioning_nodes = [
        node
        for node in active_nodes
        if str(node.type) in {"NoveltyClaim", "Hypothesis", "Method"}
    ]
    positioning_node_ids = {str(node.id) for node in positioning_nodes}
    overlap_edges = [
        edge for edge in relevant_edges if str(edge.relation) == "overlaps_prior"
    ]
    overlap_node_ids = {str(edge.source_id) for edge in overlap_edges} | {
        str(edge.target_id) for edge in overlap_edges
    }
    positioning_coverage = (
        len({node_id for node_id in overlap_node_ids if node_id in positioning_node_ids})
        / len(positioning_nodes)
        if positioning_nodes
        else 1.0
    )
    overlap_presence = _clamp(len(overlap_edges) / 2.0, 0.0, 1.0)
    positioning_value = _clamp(
        (0.60 * overlap_presence) + (0.40 * positioning_coverage),
        0.0,
        1.0,
    )

    repair_edges = [edge for edge in relevant_edges if str(edge.relation) == "repairs"]
    if contradiction_edges:
        repaired_target_ids = {
            str(edge.target_id)
            for edge in repair_edges
            if str(edge.target_id) in contradiction_node_ids
        }
        repair_activity = _clamp(
            len(repair_edges) / max(1, len(contradiction_edges)),
            0.0,
            1.0,
        )
        repair_value = _clamp(
            (0.40 * (1.0 - unresolved_contradiction_ratio))
            + (0.35 * (len(repaired_target_ids) / max(1, len(contradiction_node_ids))))
            + (0.25 * repair_activity),
            0.0,
            1.0,
        )
    else:
        repair_value = 1.0

    completeness_flag = _safe_state_bool(state, "completeness", False)
    core_type_fraction = sum(
        1 for node_type in {"Problem", "Hypothesis", "Method", "EvalPlan"} if node_type in active_types
    ) / 4.0
    backbone_checks: list[float] = []
    if "Problem" in active_types:
        backbone_checks.append(
            1.0
            if any(
                str(edge.relation) == "supports"
                and str(graph.nodes.get(str(edge.target_id), object()).type) == "Problem"
                for edge in relevant_edges
                if str(edge.target_id) in graph.nodes
            )
            else 0.0
        )
    if "Hypothesis" in active_types:
        backbone_checks.append(
            1.0
            if any(
                str(edge.relation) in {"supports", "depends_on"}
                and (
                    str(graph.nodes.get(str(edge.target_id), object()).type) == "Hypothesis"
                    or str(graph.nodes.get(str(edge.source_id), object()).type) == "Hypothesis"
                )
                for edge in relevant_edges
                if str(edge.target_id) in graph.nodes and str(edge.source_id) in graph.nodes
            )
            else 0.0
        )
    if "Method" in active_types:
        backbone_checks.append(
            1.0
            if any(
                str(edge.relation) in {"supports", "depends_on"}
                and (
                    str(graph.nodes.get(str(edge.target_id), object()).type) == "Method"
                    or str(graph.nodes.get(str(edge.source_id), object()).type) == "Method"
                )
                for edge in relevant_edges
                if str(edge.target_id) in graph.nodes and str(edge.source_id) in graph.nodes
            )
            else 0.0
        )
    if "EvalPlan" in active_types:
        backbone_checks.append(
            1.0
            if any(
                (
                    str(edge.relation) == "depends_on"
                    and (
                        str(graph.nodes.get(str(edge.target_id), object()).type) == "EvalPlan"
                        or str(graph.nodes.get(str(edge.source_id), object()).type) == "EvalPlan"
                    )
                )
                or (
                    str(edge.relation) == "supports"
                    and str(graph.nodes.get(str(edge.target_id), object()).type) == "EvalPlan"
                )
                for edge in relevant_edges
                if str(edge.target_id) in graph.nodes and str(edge.source_id) in graph.nodes
            )
            else 0.0
        )
    backbone_fraction = (
        sum(backbone_checks) / len(backbone_checks)
        if backbone_checks
        else 0.0
    )
    completeness_value = _clamp(
        (0.45 * core_type_fraction)
        + (0.30 * backbone_fraction)
        + (0.25 * (1.0 if completeness_flag else 0.0)),
        0.0,
        1.0,
    )
    utility_stable = _safe_state_bool(state, "utility_stable", False)
    challenge_health = _clamp(
        (0.60 * challenge_value) + (0.40 * (1.0 - unresolved_contradiction_ratio)),
        0.0,
        1.0,
    )
    maturity_value = _clamp(
        (0.18 * support_value)
        + (0.14 * dependency_value)
        + (0.12 * float(grounding_value))
        + (0.10 * challenge_health)
        + (0.09 * positioning_value)
        + (0.11 * repair_value)
        + (0.16 * completeness_value)
        + (0.08 * (1.0 - unresolved_contradiction_ratio))
        + (0.02 if utility_stable else 0.0),
        0.0,
        1.0,
    )
    feedback: dict[str, object] = {
        "round_index": int(_safe_state_float(state, "round_index", 0.0)),
        "support_coverage": round(_clamp(support_value, 0.0, 1.0), 4),
        "unresolved_contradiction_ratio": round(
            _clamp(unresolved_contradiction_ratio, 0.0, 1.0),
            4,
        ),
        "completeness_flag": completeness_flag,
        "is_mature": _safe_state_bool(state, "is_mature", False),
        "utility_stable": utility_stable,
        "support": round(_clamp(support_value, 0.0, 1.0), 4),
        "dependency": round(_clamp(dependency_value, 0.0, 1.0), 4),
        "grounding": round(_clamp(float(grounding_value), 0.0, 1.0), 4),
        "challenge": round(_clamp(challenge_value, 0.0, 1.0), 4),
        "positioning": round(_clamp(positioning_value, 0.0, 1.0), 4),
        "repair": round(_clamp(repair_value, 0.0, 1.0), 4),
        "completeness": round(_clamp(completeness_value, 0.0, 1.0), 4),
        "maturity": round(_clamp(maturity_value, 0.0, 1.0), 4),
        "contradiction_load": round(_clamp(unresolved_contradiction_ratio, 0.0, 1.0), 4),
        "utility": round(_safe_state_float(state, "utility", 0.0), 4),
    }
    for key in ("evidence", "coherence", "novelty"):
        value = _optional_breakdown_float(state, key)
        feedback[key] = None if value is None else round(float(value), 4)
    feedback["round_phase"] = _round_phase_key(int(feedback["round_index"]))
    return feedback


def _feedback_float(feedback: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = feedback.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _feedback_optional_float(feedback: Mapping[str, object], key: str) -> float | None:
    value = feedback.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _signal_deficits(feedback: Mapping[str, object]) -> dict[str, float]:
    deficits: dict[str, float] = {}
    for signal, target in _GRAPH_SIGNAL_TARGETS.items():
        deficits[signal] = round(
            _deficit(_feedback_optional_float(feedback, signal), target),
            4,
        )
    return deficits


def _graph_signal_payload(
    graph: IdeaGraph,
    state_features: Mapping[str, Any] | None,
) -> tuple[dict[str, object], dict[str, float]]:
    feedback = _graph_signal_feedback(graph, state_features)
    return feedback, _signal_deficits(feedback)


def _phase_multiplier(signal: str, phase: str) -> float:
    if phase == "structure":
        if signal in {"support", "dependency"}:
            return 1.15
        if signal == "grounding":
            return 0.95
        if signal == "repair":
            return 0.85
        if signal in {"challenge", "positioning"}:
            return 0.90
        if signal == "maturity":
            return 0.85
    elif phase == "stress_test":
        if signal in {"grounding", "challenge", "positioning"}:
            return 1.15
        if signal in {"support", "dependency"}:
            return 0.90
    elif phase == "repair":
        if signal == "repair":
            return 1.20
        if signal == "challenge":
            return 0.75
        if signal == "positioning":
            return 0.85
        if signal in {"support", "dependency", "grounding"}:
            return 0.85
        if signal == "maturity":
            return 1.10
    return 1.0


def _raw_action_calibration_bias(kind: str, feedback: Mapping[str, object]) -> float:
    aligned_signal = _aligned_signal(kind)
    phase = str(feedback.get("round_phase", "")).strip() or "structure"
    deficits = _signal_deficits(feedback)
    mature_graph = bool(feedback.get("is_mature", False))
    support_coverage = _feedback_float(feedback, "support")
    maturity = _feedback_float(feedback, "maturity")
    utility_stable = bool(feedback.get("utility_stable", False))
    contradiction_pressure = _feedback_float(feedback, "contradiction_load")
    support_deficit = deficits["support"]
    dependency_deficit = deficits["dependency"]
    grounding_deficit = deficits["grounding"]
    challenge_deficit = deficits["challenge"]
    positioning_deficit = deficits["positioning"]
    repair_deficit = deficits["repair"]
    completeness_deficit = deficits["completeness"]
    residual_need = (
        support_deficit
        + dependency_deficit
        + grounding_deficit
        + (0.5 * challenge_deficit)
        + (0.5 * positioning_deficit)
        + repair_deficit
        + completeness_deficit
    ) / 6.0
    need_pressure = max(
        support_deficit,
        dependency_deficit,
        grounding_deficit,
        challenge_deficit,
        positioning_deficit,
        repair_deficit,
        contradiction_pressure,
        completeness_deficit,
    )

    raw = 0.0
    if kind == "add_support_edge":
        raw += 0.85 * support_deficit
        raw += 0.15 * completeness_deficit
        raw -= 0.10 * contradiction_pressure
        raw -= 0.18 * maturity
        raw -= 0.50 if mature_graph else 0.0
    elif kind == "add_dependency_edge":
        raw += 0.90 * dependency_deficit
        raw += 0.20 * completeness_deficit
        raw -= 0.12 * maturity
        raw -= 0.35 if mature_graph else 0.0
    elif kind == "request_evidence":
        raw += 0.75 * grounding_deficit
        raw += 0.20 * support_deficit
        raw += 0.10 * completeness_deficit
        raw -= 0.10 * maturity
        raw -= 0.25 if mature_graph else 0.0
    elif kind == "attach_evidence":
        raw += 0.85 * grounding_deficit
        raw += 0.10 * support_deficit
        raw -= 0.10 * maturity
        raw -= 0.25 if mature_graph else 0.0
    elif kind == "add_contradiction_edge":
        challenge_gap = max(0.0, challenge_deficit - (0.85 * contradiction_pressure))
        raw += 0.80 * challenge_gap
        raw -= 0.35 * contradiction_pressure
        raw -= 0.10 * maturity
        raw -= 0.22 if mature_graph else 0.0
    elif kind == "mark_overlap":
        raw += 1.00 * positioning_deficit
        raw -= 0.08 * maturity
        raw -= 0.22 if mature_graph else 0.0
    elif kind == "propose_repair":
        raw += 0.70 * repair_deficit
        raw += 0.60 * contradiction_pressure
        raw += 0.15 * completeness_deficit
        raw -= 0.10 * maturity
        raw -= 0.16 if mature_graph else 0.0
    elif kind in {"skip", "freeze_branch"}:
        raw += 1.05 * maturity
        raw += 0.20 if utility_stable else 0.0
        raw += 0.10 if bool(feedback.get("is_mature", False)) else 0.0
        raw -= 0.20 * residual_need

    return raw * _phase_multiplier(aligned_signal, phase)


def _calibrated_scored_row(
    graph: IdeaGraph,
    row: Mapping[str, object],
    *,
    state_features: Mapping[str, Any] | None,
    config: RelationGraphRuntimeConfig,
) -> dict[str, object]:
    updated = dict(row)
    base_score = float(updated.get("critic_base_score", updated.get("critic_score", 0.0)))
    family = _action_family(str(updated.get("kind", "")).strip())
    aligned_signal = _aligned_signal(str(updated.get("kind", "")).strip())
    feedback, deficits = _graph_signal_payload(graph, state_features)
    enabled = (
        bool(config.use_action_score_calibration)
        and family not in {"commit", "other"}
        and math.isfinite(base_score)
    )
    max_bias = max(0.0, float(config.action_score_calibration_max_bias))
    strength = max(0.0, float(config.action_score_calibration_strength))
    bias = 0.0
    if enabled and max_bias > 0.0 and strength > 0.0:
        raw_bias = _raw_action_calibration_bias(str(updated.get("kind", "")).strip(), feedback)
        bias = _clamp(strength * raw_bias, -max_bias, max_bias)
    calibrated_score = base_score + bias
    updated["critic_base_score"] = base_score
    updated["critic_calibration_bias"] = round(bias, 6)
    updated["critic_score_calibrated"] = calibrated_score
    updated["critic_score"] = calibrated_score
    updated["critic_action_family"] = family
    updated["critic_aligned_signal"] = aligned_signal
    updated["critic_calibration_enabled"] = bool(enabled)
    updated["critic_calibration_feedback"] = feedback
    updated["critic_calibration_deficits"] = deficits
    return updated


def calibrate_commit_probability(
    graph: IdeaGraph,
    *,
    raw_probability: float,
    state_features: Mapping[str, Any] | None,
    config: RelationGraphRuntimeConfig,
    feedback: Mapping[str, object] | None = None,
    deficits: Mapping[str, float] | None = None,
) -> tuple[float, float, bool, dict[str, object]]:
    probability = _clamp(float(raw_probability), 0.0, 1.0)
    if feedback is None or deficits is None:
        feedback, deficits = _graph_signal_payload(graph, state_features)
    enabled = bool(config.use_action_score_calibration)
    if not enabled:
        return probability, 0.0, False, dict(feedback)

    strength = max(0.0, float(config.action_score_calibration_strength))
    max_bias = max(0.0, min(0.18, float(config.action_score_calibration_max_bias) * 0.50))
    if strength <= 0.0 or max_bias <= 0.0:
        return probability, 0.0, False, dict(feedback)

    support_deficit = float(deficits["support"])
    dependency_deficit = float(deficits["dependency"])
    grounding_deficit = float(deficits["grounding"])
    challenge_deficit = float(deficits["challenge"])
    positioning_deficit = float(deficits["positioning"])
    repair_deficit = float(deficits["repair"])
    completeness_deficit = float(deficits["completeness"])
    contradiction_load = _feedback_float(feedback, "contradiction_load")
    support = _feedback_float(feedback, "support")
    dependency = _feedback_float(feedback, "dependency", 1.0)
    grounding = _feedback_float(feedback, "grounding")
    challenge = _feedback_float(feedback, "challenge")
    positioning = _feedback_float(feedback, "positioning")
    repair = _feedback_float(feedback, "repair", 1.0)
    completeness = _feedback_float(feedback, "completeness")
    maturity = _feedback_float(feedback, "maturity")
    utility_stable = bool(feedback.get("utility_stable", False))
    is_mature = bool(feedback.get("is_mature", False))
    round_index = int(_feedback_float(feedback, "round_index", 0.0))
    round_bonus = _clamp(
        0.05 * max(0, round_index - int(config.min_commit_round) + 1),
        0.0,
        0.15,
    )
    resolution_readiness = _clamp(
        (0.55 * repair) + (0.45 * (1.0 - contradiction_load)),
        0.0,
        1.0,
    )
    readiness = _clamp(
        (0.20 * support)
        + (0.10 * dependency)
        + (0.16 * grounding)
        + (0.22 * resolution_readiness)
        + (0.14 * completeness)
        + (0.18 * maturity),
        0.0,
        1.0,
    )
    average_deficit = (
        support_deficit
        + dependency_deficit
        + (1.20 * grounding_deficit)
        + (0.50 * challenge_deficit)
        + (0.50 * positioning_deficit)
        + (1.35 * repair_deficit)
        + completeness_deficit
    ) / 6.55
    blocking_need = max(
        support_deficit,
        grounding_deficit,
        repair_deficit,
        contradiction_load,
    )
    raw_bias = (
        (0.65 * (readiness - 0.56))
        + (0.35 * ((1.0 - average_deficit) - 0.58))
        + (0.10 if utility_stable else 0.0)
        + (0.05 if is_mature else 0.0)
        + round_bonus
        - (0.24 * blocking_need)
        - (0.16 * contradiction_load)
    )
    bias = _clamp(strength * raw_bias, -max_bias, max_bias)
    calibrated = _clamp(probability + bias, 0.0, 1.0)
    return calibrated, bias, True, dict(feedback)


def _policy_candidate_from_scored_spec(spec: Mapping[str, Any]) -> ScoredCandidate:
    return ScoredCandidate(
        candidate_id=str(spec["candidate_id"]),
        score=float(spec["critic_score"]),
        is_commit=str(spec.get("kind", "")).strip() == "commit",
        confidence=float(spec["critic_score"]),
        predicted_gain=_candidate_float(spec, "predicted_gain"),
        support_gain=_candidate_float(spec, "support_gain"),
        contradiction_gain=_candidate_float(spec, "contradiction_gain"),
        maturity_gain=_candidate_float(spec, "maturity_gain"),
        after_is_mature=_candidate_after_is_mature(spec),
    )


def _runtime_fallback_reason(diagnostics: RelationGraphRuntimeRowDiagnostics) -> str:
    if diagnostics.used_vocab_fallback:
        return "unmapped_runtime_token"
    if diagnostics.missing_target_ids:
        return "missing_runtime_target"
    return "runtime_token_unsafe"


def _unsafe_runtime_candidates_from_batch(
    runtime_batch: RelationGraphRuntimeBatch,
) -> dict[str, str]:
    unsafe_by_id: dict[str, str] = {}
    for index, is_unsafe in enumerate(runtime_batch.fallback_row_mask.tolist()):
        if not bool(is_unsafe):
            continue
        diagnostics = runtime_batch.diagnostics[index]
        unsafe_by_id[str(diagnostics.candidate_id)] = _runtime_fallback_reason(diagnostics)
    return unsafe_by_id


def _build_runtime_batch(
    runtime_bundle: Any,
    *,
    graph: IdeaGraph,
    candidate_specs: Sequence[Mapping[str, Any]],
    use_commit: bool,
) -> RelationGraphRuntimeBatch:
    build_fn = getattr(runtime_bundle, "build_runtime_batch", None)
    if callable(build_fn):
        return build_fn(
            graph=graph,
            candidate_specs=candidate_specs,
            use_commit=use_commit,
        )

    vocabs = getattr(runtime_bundle, "vocabs", None) or getattr(runtime_bundle, "vocabularies", None)
    text_backend = getattr(runtime_bundle, "text_backend", None)
    if vocabs is None or text_backend is None:
        raise ValueError(
            "runtime_bundle must provide build_runtime_batch(...) or provide "
            "both vocabularies/vocabs and text_backend."
        )
    return build_relation_graph_runtime_batch(
        graph=graph,
        candidate_specs=candidate_specs,
        text_backend=text_backend,
        vocabularies=vocabs,
        use_commit=use_commit,
    )


def _unsafe_runtime_candidates_from_bundle_status(
    runtime_bundle: Any,
    runtime_batch: RelationGraphRuntimeBatch,
) -> tuple[dict[str, str], str | None]:
    status_fn = getattr(runtime_bundle, "runtime_token_status", None)
    if not callable(status_fn):
        return {}, None
    status = status_fn(runtime_batch)
    if not isinstance(status, Mapping):
        return {}, None
    ok = bool(status.get("ok", True))
    if ok:
        return {}, None
    reason = str(status.get("reason", "")).strip() or "unmapped_runtime_token"
    candidate_ids_raw = status.get("candidate_ids", ())
    candidate_ids: list[str] = []
    if isinstance(candidate_ids_raw, Sequence) and not isinstance(candidate_ids_raw, (str, bytes)):
        candidate_ids = [str(candidate_id).strip() for candidate_id in candidate_ids_raw if str(candidate_id).strip()]
    if candidate_ids:
        return ({candidate_id: reason for candidate_id in candidate_ids}, None)
    return ({}, reason)


class LoadedRelationGraphRuntimeCritic:
    def __init__(
        self,
        *,
        model: RelationGraphCritic,
        vocabs: RelationGraphVocabularies,
        text_backend: Any,
        device: torch.device,
    ) -> None:
        self.model = model
        self.vocabs = vocabs
        self.text_backend = text_backend
        self.device = device

    def build_runtime_batch(
        self,
        *,
        graph: IdeaGraph,
        candidate_specs: Sequence[Mapping[str, Any]],
        use_commit: bool,
    ) -> RelationGraphRuntimeBatch:
        return build_relation_graph_runtime_batch(
            graph=graph,
            candidate_specs=candidate_specs,
            text_backend=self.text_backend,
            vocabularies=self.vocabs,
            use_commit=use_commit,
        )

    def runtime_token_status(
        self,
        runtime_batch: RelationGraphRuntimeBatch,
    ) -> dict[str, object]:
        unsafe_by_id = _unsafe_runtime_candidates_from_batch(runtime_batch)
        if not unsafe_by_id:
            return {"ok": True, "reason": "", "candidate_ids": ()}
        return {
            "ok": False,
            "reason": "unmapped_runtime_token",
            "candidate_ids": tuple(sorted(unsafe_by_id)),
        }

    def score_runtime_batch(self, batch: RelationGraphBatch) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            scores = self.model(batch.to(self.device)).detach().cpu().tolist()
        return [float(value) for value in scores]


def _require_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_text_backend(training_config: Mapping[str, Any]) -> Any:
    backend_name = str(training_config.get("text_backend", "")).strip()
    if backend_name == "hash":
        return HashTextEmbeddingBackend(dim=int(training_config.get("embedding_dim", 0) or 0))
    if backend_name == "sentence-transformer":
        model_name = str(training_config.get("text_model_name", "")).strip()
        if not model_name:
            raise ValueError("training_config.text_model_name must be set for sentence-transformer backend.")
        return SentenceTransformerEmbeddingBackend(model_name)
    raise ValueError(f"Unsupported runtime text backend '{backend_name}'.")


def _normalized_vocab_mapping(raw: object, *, field_name: str) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"vocabularies.json field '{field_name}' must be a JSON object.")
    mapping: dict[str, int] = {}
    for key, value in raw.items():
        token = str(key).strip()
        if not token:
            continue
        try:
            mapping[token] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"vocabularies.json field '{field_name}' has non-integer id for token '{token}'."
            ) from exc
    return mapping


def _vocabularies_snapshot_payload(vocabs: RelationGraphVocabularies) -> dict[str, object]:
    return {
        "node_type_to_id": dict(vocabs.node_type_to_id),
        "role_to_id": dict(vocabs.role_to_id),
        "edge_type_to_id": dict(vocabs.edge_type_to_id),
        "candidate_kind_to_id": dict(vocabs.candidate_kind_to_id),
    }


def _load_vocabularies_from_snapshot(path: Path) -> RelationGraphVocabularies:
    payload = _require_json_object(path)
    return RelationGraphVocabularies(
        node_type_to_id=_normalized_vocab_mapping(payload.get("node_type_to_id", {}), field_name="node_type_to_id"),
        role_to_id=_normalized_vocab_mapping(payload.get("role_to_id", {}), field_name="role_to_id"),
        edge_type_to_id=_normalized_vocab_mapping(payload.get("edge_type_to_id", {}), field_name="edge_type_to_id"),
        candidate_kind_to_id=_normalized_vocab_mapping(
            payload.get("candidate_kind_to_id", {}),
            field_name="candidate_kind_to_id",
        ),
    )


def _try_write_vocabularies_snapshot(path: Path, vocabs: RelationGraphVocabularies) -> None:
    try:
        path.write_text(
            json.dumps(_vocabularies_snapshot_payload(vocabs), indent=2),
            encoding="utf-8",
        )
    except OSError:
        return


def _load_relation_graph_vocabularies(
    resolved_model_dir: Path,
    training_config: Mapping[str, Any],
) -> RelationGraphVocabularies:
    snapshot_path = resolved_model_dir / "vocabularies.json"
    if snapshot_path.exists():
        return _load_vocabularies_from_snapshot(snapshot_path)

    vocabs = build_relation_graph_vocabularies(
        candidate_dataset_dir=Path(str(training_config["candidate_dataset_dir"])),
        g1_dataset_dir=Path(str(training_config["g1_dataset_dir"])),
        partition_manifest_path=Path(str(training_config["partition_manifest"])),
    )
    _try_write_vocabularies_snapshot(snapshot_path, vocabs)
    return vocabs


def _load_torch_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _require_trailing_unknown_bucket(vocab: Mapping[str, int], *, vocab_name: str) -> int:
    unknown_id = vocab.get("unknown")
    if unknown_id is None:
        raise ValueError(f"Runtime vocabulary '{vocab_name}' is missing required trailing 'unknown' bucket.")
    normalized_unknown_id = int(unknown_id)
    expected_unknown_id = len(vocab) - 1
    if normalized_unknown_id != expected_unknown_id:
        raise ValueError(
            f"Runtime vocabulary '{vocab_name}' must place 'unknown' at id {expected_unknown_id}, "
            f"found {normalized_unknown_id}."
        )
    return normalized_unknown_id


def _edge_linear_weight_key(layer_index: int, edge_type_id: int) -> str:
    return f"layers.{layer_index}.edge_linears.{edge_type_id}.weight"


def _edge_linear_indices_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    layer_index: int,
) -> list[int]:
    prefix = f"layers.{layer_index}.edge_linears."
    suffix = ".weight"
    indices: set[int] = set()
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        middle = key[len(prefix): -len(suffix)]
        try:
            edge_type_id = int(middle)
        except ValueError:
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Checkpoint entry '{key}' must be a tensor.")
        indices.add(edge_type_id)
    return sorted(indices)


def _maybe_pad_legacy_unknown_embedding_row(
    *,
    state_dict: dict[str, Any],
    model_state_dict: Mapping[str, Any],
    key: str,
    vocab_name: str,
    unknown_id: int,
) -> None:
    checkpoint_tensor = state_dict.get(key)
    if not isinstance(checkpoint_tensor, torch.Tensor):
        raise ValueError(f"Checkpoint is missing tensor '{key}'.")
    runtime_tensor = model_state_dict.get(key)
    if not isinstance(runtime_tensor, torch.Tensor):
        raise ValueError(f"Runtime model is missing tensor '{key}'.")

    checkpoint_shape = tuple(int(value) for value in checkpoint_tensor.shape)
    runtime_shape = tuple(int(value) for value in runtime_tensor.shape)
    if checkpoint_shape == runtime_shape:
        return

    if (
        len(checkpoint_shape) == 2
        and len(runtime_shape) == 2
        and checkpoint_shape[1] == runtime_shape[1]
        and checkpoint_shape[0] + 1 == runtime_shape[0]
        and unknown_id == runtime_shape[0] - 1
    ):
        padding = torch.zeros(
            (1, checkpoint_shape[1]),
            dtype=checkpoint_tensor.dtype,
            device=checkpoint_tensor.device,
        )
        state_dict[key] = torch.cat([checkpoint_tensor, padding], dim=0)
        return

    raise ValueError(
        f"Checkpoint tensor '{key}' does not match runtime vocab shape for {vocab_name}: "
        f"checkpoint={checkpoint_shape}, runtime={runtime_shape}. "
        "Only legacy trailing 'unknown' bucket compatibility is supported."
    )


def _maybe_pad_legacy_unknown_edge_linears(
    *,
    state_dict: dict[str, Any],
    model_state_dict: Mapping[str, Any],
    layer_count: int,
    edge_type_count: int,
    unknown_edge_type_id: int,
) -> None:
    expected_indices = list(range(edge_type_count))
    legacy_indices = expected_indices[:-1]
    for layer_index in range(layer_count):
        present_indices = _edge_linear_indices_from_state_dict(
            state_dict,
            layer_index=layer_index,
        )
        if present_indices == expected_indices:
            continue
        if present_indices == legacy_indices and unknown_edge_type_id == edge_type_count - 1:
            target_key = _edge_linear_weight_key(layer_index, unknown_edge_type_id)
            runtime_tensor = model_state_dict.get(target_key)
            if not isinstance(runtime_tensor, torch.Tensor):
                raise ValueError(f"Runtime model is missing tensor '{target_key}'.")
            template_tensor: torch.Tensor | None = None
            for edge_type_id in legacy_indices:
                source_key = _edge_linear_weight_key(layer_index, edge_type_id)
                source_value = state_dict.get(source_key)
                if isinstance(source_value, torch.Tensor):
                    template_tensor = source_value
                    break
            if template_tensor is None:
                template_tensor = runtime_tensor
            state_dict[target_key] = torch.zeros(
                tuple(int(value) for value in runtime_tensor.shape),
                dtype=template_tensor.dtype,
                device=template_tensor.device,
            )
            continue
        raise ValueError(
            "Checkpoint edge-type linears do not match runtime vocab shape: "
            f"layer={layer_index}, checkpoint_edge_ids={present_indices}, "
            f"runtime_edge_ids={expected_indices}. "
            "Only legacy trailing 'unknown' bucket compatibility is supported."
        )


def _apply_legacy_unknown_bucket_compatibility(
    *,
    state_dict: Mapping[str, Any],
    model: RelationGraphCritic,
    vocabs: RelationGraphVocabularies,
) -> dict[str, Any]:
    node_type_unknown_id = _require_trailing_unknown_bucket(
        vocabs.node_type_to_id,
        vocab_name="node_type",
    )
    role_unknown_id = _require_trailing_unknown_bucket(
        vocabs.role_to_id,
        vocab_name="role",
    )
    edge_type_unknown_id = _require_trailing_unknown_bucket(
        vocabs.edge_type_to_id,
        vocab_name="edge_type",
    )
    candidate_kind_unknown_id = _require_trailing_unknown_bucket(
        vocabs.candidate_kind_to_id,
        vocab_name="candidate_kind",
    )

    adapted_state_dict = dict(state_dict)
    model_state_dict = model.state_dict()
    _maybe_pad_legacy_unknown_embedding_row(
        state_dict=adapted_state_dict,
        model_state_dict=model_state_dict,
        key="node_type_embed.weight",
        vocab_name="node_type",
        unknown_id=node_type_unknown_id,
    )
    _maybe_pad_legacy_unknown_embedding_row(
        state_dict=adapted_state_dict,
        model_state_dict=model_state_dict,
        key="role_embed.weight",
        vocab_name="role",
        unknown_id=role_unknown_id,
    )
    _maybe_pad_legacy_unknown_embedding_row(
        state_dict=adapted_state_dict,
        model_state_dict=model_state_dict,
        key="candidate_kind_embed.weight",
        vocab_name="candidate_kind",
        unknown_id=candidate_kind_unknown_id,
    )
    _maybe_pad_legacy_unknown_edge_linears(
        state_dict=adapted_state_dict,
        model_state_dict=model_state_dict,
        layer_count=len(model.layers),
        edge_type_count=len(vocabs.edge_type_to_id),
        unknown_edge_type_id=edge_type_unknown_id,
    )
    return adapted_state_dict


def load_relation_graph_runtime_bundle(model_dir: Path | str) -> LoadedRelationGraphRuntimeCritic:
    resolved = Path(model_dir).resolve()
    training_config = _require_json_object(resolved / "training_config.json")
    metadata = _require_json_object(resolved / "metadata.json")

    vocabs = _load_relation_graph_vocabularies(resolved, training_config)
    text_backend = _load_text_backend(training_config)
    text_dim = int(training_config.get("embedding_dim", 0) or 0)
    if text_dim <= 0:
        raise ValueError("training_config.embedding_dim must be a positive integer.")
    hidden_dim = int(metadata.get("hidden_dim", training_config.get("hidden_dim", 0) or 0))
    if hidden_dim <= 0:
        raise ValueError("metadata.hidden_dim (or training_config.hidden_dim) must be positive.")

    model = RelationGraphCritic(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        node_type_count=len(vocabs.node_type_to_id),
        role_count=len(vocabs.role_to_id),
        edge_type_count=len(vocabs.edge_type_to_id),
        candidate_kind_count=len(vocabs.candidate_kind_to_id),
    )
    state_payload = _load_torch_checkpoint(resolved / "model.pt")
    if not isinstance(state_payload, Mapping):
        raise ValueError(f"{resolved / 'model.pt'} must contain a model state dict.")
    state_dict = state_payload.get("model_state_dict", state_payload)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"{resolved / 'model.pt'} does not contain a valid model state dict.")
    compatible_state_dict = _apply_legacy_unknown_bucket_compatibility(
        state_dict=state_dict,
        model=model,
        vocabs=vocabs,
    )
    try:
        model.load_state_dict(compatible_state_dict)
    except RuntimeError as exc:
        raise ValueError(
            f"{resolved / 'model.pt'} does not match runtime vocab shape or architecture: {exc}"
        ) from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return LoadedRelationGraphRuntimeCritic(
        model=model,
        vocabs=vocabs,
        text_backend=text_backend,
        device=device,
    )


def _fallback_scored_row(spec: Mapping[str, Any], *, reason: str, candidate_ids: tuple[str, ...]) -> dict[str, object]:
    kind = str(spec.get("kind", "")).strip()
    return {
        **dict(spec),
        "critic_score": float("-inf"),
        "critic_base_score": float("-inf"),
        "critic_calibration_bias": 0.0,
        "critic_score_calibrated": float("-inf"),
        "critic_action_family": _action_family(kind),
        "critic_aligned_signal": _aligned_signal(kind),
        "critic_calibration_enabled": False,
        "critic_calibration_feedback": {},
        "critic_calibration_deficits": {},
        "predicted_gain": _candidate_float(spec, "predicted_gain"),
        "support_gain": _candidate_float(spec, "support_gain"),
        "contradiction_gain": _candidate_float(spec, "contradiction_gain"),
        "maturity_gain": _candidate_float(spec, "maturity_gain"),
        "after_is_mature": _candidate_after_is_mature(spec),
        "controller_fallback_reason": reason,
        "controller_fallback_candidate_ids": candidate_ids,
    }


def _scored_row_from_score(spec: Mapping[str, Any], score: float) -> dict[str, object]:
    kind = str(spec.get("kind", "")).strip()
    return {
        **dict(spec),
        "critic_score": float(score),
        "critic_base_score": float(score),
        "critic_calibration_bias": 0.0,
        "critic_score_calibrated": float(score),
        "critic_action_family": _action_family(kind),
        "critic_aligned_signal": _aligned_signal(kind),
        "critic_calibration_enabled": False,
        "critic_calibration_feedback": {},
        "critic_calibration_deficits": {},
        "predicted_gain": _candidate_float(spec, "predicted_gain"),
        "support_gain": _candidate_float(spec, "support_gain"),
        "contradiction_gain": _candidate_float(spec, "contradiction_gain"),
        "maturity_gain": _candidate_float(spec, "maturity_gain"),
        "after_is_mature": _candidate_after_is_mature(spec),
    }


def _maybe_apply_low_signal_kind_swap_guard(
    *,
    policy_decision: CriticPolicyDecision,
    heuristic_row: Mapping[str, Any],
    selected_row_lookup: Mapping[str, Mapping[str, Any]],
    low_signal_threshold: float,
) -> CriticPolicyDecision:
    if policy_decision.selected_source != "critic":
        return policy_decision
    if policy_decision.selected_candidate_id == str(heuristic_row["candidate_id"]):
        return policy_decision

    selected_row = selected_row_lookup.get(policy_decision.selected_candidate_id)
    if selected_row is None:
        return policy_decision

    heuristic_kind = str(heuristic_row.get("kind", "")).strip()
    selected_kind = str(selected_row.get("kind", "")).strip()
    if not heuristic_kind or not selected_kind or heuristic_kind == selected_kind:
        return policy_decision
    if selected_kind == "skip":
        return policy_decision

    heuristic_gain = _candidate_float(heuristic_row, "predicted_gain")
    selected_gain = _candidate_float(selected_row, "predicted_gain")
    if heuristic_gain > low_signal_threshold or selected_gain > low_signal_threshold:
        return policy_decision

    return CriticPolicyDecision(
        selected_candidate_id=str(heuristic_row["candidate_id"]),
        selected_source="heuristic",
        used_heuristic_fallback=True,
        commit_allowed=policy_decision.commit_allowed,
        commit_requested=policy_decision.commit_requested,
        override_margin=policy_decision.override_margin,
        commit_margin=policy_decision.commit_margin,
        fallback_reason="low_signal_kind_swap_guard",
    )


def select_relation_graph_critic_candidate(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    state_features: Mapping[str, Any] | None = None,
    candidate_specs: Sequence[Mapping[str, Any]],
    heuristic_candidate_id: str,
    runtime_bundle: Any,
    config: RelationGraphRuntimeConfig,
) -> RelationGraphRuntimeDecision:
    del role
    normalized_specs = _normalized_candidate_specs(
        candidate_specs,
        use_commit=bool(config.use_commit),
    )
    if not normalized_specs:
        raise ValueError("candidate_specs must not be empty after runtime-critic filtering.")

    scored_lookup = {
        str(spec["candidate_id"]): dict(spec)
        for spec in normalized_specs
    }
    heuristic_row = scored_lookup.get(str(heuristic_candidate_id).strip())
    if heuristic_row is None:
        raise ValueError(f"heuristic_candidate_id '{heuristic_candidate_id}' is not present in candidate_specs.")
    runtime_batch = _build_runtime_batch(
        runtime_bundle,
        graph=graph,
        candidate_specs=normalized_specs,
        use_commit=bool(config.use_commit),
    )
    all_candidate_ids = tuple(str(spec["candidate_id"]) for spec in normalized_specs)
    unsafe_reason_by_id = _unsafe_runtime_candidates_from_batch(runtime_batch)
    bundle_unsafe_by_id, global_fallback_reason = _unsafe_runtime_candidates_from_bundle_status(
        runtime_bundle,
        runtime_batch,
    )
    for candidate_id, reason in bundle_unsafe_by_id.items():
        if candidate_id in scored_lookup:
            unsafe_reason_by_id[candidate_id] = reason
    unsafe_candidate_ids = tuple(sorted(unsafe_reason_by_id))
    heuristic_candidate_id = str(heuristic_row["candidate_id"])

    round_index = int((state_features or {}).get("round_index", _parse_round_index(round_name)) or 0)
    commit_requested = any(str(spec.get("kind", "")).strip() == "commit" for spec in normalized_specs)
    commit_allowed = bool(commit_requested and round_index >= int(config.min_commit_round))

    def _heuristic_fallback_result(
        *,
        selected_reason: str,
        default_reason_for_rows: str,
    ) -> RelationGraphRuntimeDecision:
        fallback_ids = unsafe_candidate_ids or all_candidate_ids
        scored_candidates = tuple(
            _fallback_scored_row(
                spec,
                reason=unsafe_reason_by_id.get(str(spec["candidate_id"]), default_reason_for_rows),
                candidate_ids=fallback_ids,
            )
            for spec in normalized_specs
        )
        selected_spec = _fallback_scored_row(
            heuristic_row,
            reason=selected_reason,
            candidate_ids=fallback_ids,
        )
        return RelationGraphRuntimeDecision(
            selected_spec=selected_spec,
            policy_decision=CriticPolicyDecision(
                selected_candidate_id=heuristic_candidate_id,
                selected_source="heuristic",
                used_heuristic_fallback=True,
                commit_allowed=commit_allowed,
                commit_requested=commit_requested,
                override_margin=float("-inf"),
                commit_margin=None,
            ),
            scored_candidates=scored_candidates,
        )

    if global_fallback_reason is not None:
        return _heuristic_fallback_result(
            selected_reason=global_fallback_reason,
            default_reason_for_rows=global_fallback_reason,
        )

    heuristic_unsafe_reason = unsafe_reason_by_id.get(heuristic_candidate_id)
    if heuristic_unsafe_reason is not None:
        return _heuristic_fallback_result(
            selected_reason=heuristic_unsafe_reason,
            default_reason_for_rows="heuristic_candidate_unsafe",
        )

    safe_specs = [
        spec
        for spec in normalized_specs
        if str(spec["candidate_id"]) not in unsafe_reason_by_id
    ]
    if not safe_specs:
        return _heuristic_fallback_result(
            selected_reason="no_safe_runtime_candidates",
            default_reason_for_rows="no_safe_runtime_candidates",
        )

    safe_runtime_batch = _build_runtime_batch(
        runtime_bundle,
        graph=graph,
        candidate_specs=safe_specs,
        use_commit=bool(config.use_commit),
    )
    score_fn = getattr(runtime_bundle, "score_runtime_batch", None)
    if not callable(score_fn):
        raise ValueError("runtime_bundle must provide score_runtime_batch(batch).")
    safe_scores = score_fn(safe_runtime_batch.batch)
    if len(safe_scores) != len(safe_specs):
        raise ValueError(
            f"Runtime relation-graph critic returned {len(safe_scores)} scores for {len(safe_specs)} safe candidates."
        )

    safe_scored_lookup = {
        str(spec["candidate_id"]): _calibrated_scored_row(
            graph,
            _scored_row_from_score(spec, float(score)),
            state_features=state_features,
            config=config,
        )
        for spec, score in zip(safe_specs, safe_scores, strict=True)
    }
    scored_candidates: list[dict[str, object]] = []
    policy_candidates: list[ScoredCandidate] = []
    for spec in normalized_specs:
        candidate_id = str(spec["candidate_id"])
        if candidate_id in unsafe_reason_by_id:
            scored_row = _fallback_scored_row(
                spec,
                reason=unsafe_reason_by_id[candidate_id],
                candidate_ids=(candidate_id,),
            )
        else:
            scored_row = dict(safe_scored_lookup[candidate_id])
            policy_candidates.append(_policy_candidate_from_scored_spec(scored_row))
        scored_candidates.append(scored_row)

    heuristic_scored = safe_scored_lookup.get(heuristic_candidate_id)
    if heuristic_scored is None:
        raise ValueError(f"heuristic_candidate_id '{heuristic_candidate_id}' is not present in safe candidates.")

    policy_config = SafeCriticPolicyConfig(
        min_commit_round=int(config.min_commit_round),
        tau_override=float(config.tau_override),
        tau_override_by_round=dict(config.tau_override_by_round or {}),
        tau_commit=float(config.tau_commit),
        gamma_commit=float(config.gamma_commit),
        gamma_commit_by_round=dict(config.gamma_commit_by_round or {}),
        guard_support_threshold=float(config.guard_support_threshold),
        guard_support_gain_floor=float(config.guard_support_gain_floor),
        guard_requires_contradiction_progress=bool(
            config.guard_requires_contradiction_progress
        ),
        guard_commit_support_threshold=float(config.guard_commit_support_threshold),
        guard_commit_utility_floor=float(config.guard_commit_utility_floor),
    )
    policy_decision = choose_critic_action(
        state={
            "round_index": _parse_round_index(round_name),
            **dict(state_features or {}),
        },
        critic_candidates=policy_candidates,
        heuristic_candidate=_policy_candidate_from_scored_spec(heuristic_scored),
        config=policy_config,
    )

    selected_row_lookup = {str(row["candidate_id"]): row for row in scored_candidates}
    if bool(config.use_low_signal_kind_swap_guard):
        policy_decision = _maybe_apply_low_signal_kind_swap_guard(
            policy_decision=policy_decision,
            heuristic_row=heuristic_scored,
            selected_row_lookup=selected_row_lookup,
            low_signal_threshold=float(policy_config.guard_predicted_gain_min_heuristic),
        )
    selected_spec = dict(selected_row_lookup[policy_decision.selected_candidate_id])
    if policy_decision.fallback_reason and "controller_fallback_reason" not in selected_spec:
        selected_spec["controller_fallback_reason"] = policy_decision.fallback_reason
    return RelationGraphRuntimeDecision(
        selected_spec=selected_spec,
        policy_decision=policy_decision,
        scored_candidates=tuple(scored_candidates),
    )
