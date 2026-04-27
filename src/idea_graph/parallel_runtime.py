from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from .agent_backend import ActionDecision, append_agent_trace
from .action_candidates import enumerate_edit_candidate_specs
from .engine import (
    _record_runtime_controller_trace,
    action_from_decision,
    apply_action,
    choose_round_action,
    maturity_snapshot,
)
from .models import (
    GraphAction,
    ParallelCommitCheckRecord,
    ParallelEditPatchRecord,
    ParallelRoleDecisionRecord,
    ParallelRoleRoundResult,
)
from .parallel_replay import build_parallel_edit_rows, build_post_round_commit_row
from .parallel_role_executor import collect_parallel_role_decisions
from .relation_graph_runtime_critic import (
    _graph_signal_payload,
    calibrate_commit_probability,
    select_relation_graph_critic_candidate,
)
from .critic_policy import commit_threshold_for_round
from .runtime_critic import select_text_critic_candidate
from .role_activation import active_roles_for_round
from .signal_heuristic_control import graph_signal_payload as _heuristic_graph_signal_payload


def _decision_from_graph_action(action) -> ActionDecision:
    return ActionDecision(
        kind=action.kind,
        target_ids=list(action.target_ids),
        payload=dict(action.payload),
        rationale=action.rationale,
    )


def _decision_record(role: str, decision: ActionDecision) -> ParallelRoleDecisionRecord:
    return ParallelRoleDecisionRecord(
        role=role,
        kind=str(decision.kind).strip(),
        target_ids=tuple(str(item).strip() for item in decision.target_ids if str(item).strip()),
        payload=dict(decision.payload),
        rationale=str(decision.rationale).strip(),
    )


def _patch_record(role: str, decision: ActionDecision, *, is_empty: bool) -> ParallelEditPatchRecord:
    return ParallelEditPatchRecord(
        role=role,
        kind=str(decision.kind).strip(),
        target_ids=tuple(str(item).strip() for item in decision.target_ids if str(item).strip()),
        payload=dict(decision.payload),
        is_empty=is_empty,
    )


def _synthetic_action_from_decision(graph, *, round_name: str, role: str, decision: ActionDecision) -> GraphAction:
    return GraphAction(
        id=f"parallel::{round_name}::{role}::controller-baseline",
        round_name=round_name,
        role=role,
        kind=str(decision.kind).strip(),
        target_ids=[str(item).strip() for item in decision.target_ids if str(item).strip()],
        payload=dict(decision.payload),
        rationale=str(decision.rationale).strip(),
        source="parallel_controller_baseline",
    )


def _controller_candidate_specs(graph, *, round_name: str, role: str, decision: ActionDecision) -> list[dict[str, object]]:
    baseline_action = _synthetic_action_from_decision(
        graph,
        round_name=round_name,
        role=role,
        decision=decision,
    )
    return [
        {
            **candidate,
            "candidate_id": f"parallel::{round_name}::{role}::candidate:{index:04d}",
        }
        for index, candidate in enumerate(
            enumerate_edit_candidate_specs(
                graph,
                round_name=round_name,
                role=role,
                baseline_action=baseline_action,
            )
        )
    ]


def _round_index(round_name: str) -> int:
    text = str(round_name).strip()
    if not text.startswith("Round"):
        return 0
    try:
        return int(text[5:])
    except ValueError:
        return 0


def _controller_graph_signal_payload(
    graph,
    *,
    round_name: str,
    controller_kind: str,
    controller_state: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, float]]:
    if controller_kind == "signal_heuristic_control":
        return _heuristic_graph_signal_payload(graph, round_name=round_name)
    return _graph_signal_payload(graph, controller_state)


def _candidate_diversity_signature(candidate: Mapping[str, object]) -> tuple[object, ...]:
    kind = str(candidate.get("kind", "")).strip()
    target_ids = tuple(str(item).strip() for item in candidate.get("target_ids", []) if str(item).strip())
    if len(target_ids) == 1 and kind in {"propose_repair", "attach_evidence"}:
        return (kind, target_ids[0])
    return (kind, target_ids)


def _select_nonconflicting_candidate(
    scored_candidates: list[dict[str, object]],
    *,
    preferred_candidate: dict[str, object],
    used_signatures: set[tuple[object, ...]],
) -> tuple[dict[str, object], bool]:
    if str(preferred_candidate.get("kind", "")).strip() == "skip":
        return preferred_candidate, False
    preferred_signature = _candidate_diversity_signature(preferred_candidate)
    if preferred_signature not in used_signatures:
        return preferred_candidate, False

    ranked_candidates = sorted(
        scored_candidates,
        key=lambda row: (
            float(row.get("critic_score", float("-inf"))),
            str(row.get("candidate_id", "")),
        ),
        reverse=True,
    )
    for candidate in ranked_candidates:
        signature = _candidate_diversity_signature(candidate)
        if signature in used_signatures:
            continue
        return dict(candidate), True
    return preferred_candidate, False


def _record_graph_signal_trace(
    graph,
    *,
    round_name: str,
    stage: str,
    controller_kind: str,
    signals: Mapping[str, object],
    deficits: Mapping[str, float],
) -> None:
    signal_log = graph.metadata.setdefault("runtime_graph_signal_log", [])
    if not isinstance(signal_log, list):
        return
    signal_log.append(
        {
            "round": round_name,
            "stage": stage,
            "controller_kind": controller_kind,
            "signals": dict(signals),
            "deficits": dict(deficits),
        }
    )


def _mapping_payload(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    if is_dataclass(value):
        return dict(asdict(value))
    payload: dict[str, object] = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            payload[key] = getattr(value, key)
        except AttributeError:
            continue
    return payload


def _action_decision_from_candidate(candidate: dict[str, object]) -> ActionDecision:
    raw_target_ids = candidate.get("target_ids", [])
    target_ids = (
        [str(item).strip() for item in raw_target_ids if str(item).strip()]
        if isinstance(raw_target_ids, list)
        else []
    )
    payload = candidate.get("payload", {})
    return ActionDecision(
        kind=str(candidate.get("kind", "")).strip(),
        target_ids=target_ids,
        payload=dict(payload) if isinstance(payload, dict) else {},
        rationale=str(candidate.get("rationale", "")).strip(),
    )


def _maybe_apply_runtime_controller(
    graph,
    *,
    snapshot,
    round_name: str,
    raw_decisions: list[tuple[str, ActionDecision]],
    runtime_controller: Any | None,
    runtime_controller_metadata: dict[str, Any] | None,
) -> tuple[list[tuple[str, ActionDecision]], bool]:
    if runtime_controller is None or runtime_controller_metadata is None:
        return raw_decisions, False
    controller_config = runtime_controller_metadata.get("config")
    if controller_config is None:
        return raw_decisions, False
    if not bool(getattr(controller_config, "use_edit", True)):
        return raw_decisions, False

    controller_kind = str(runtime_controller_metadata.get("kind", "")).strip() or "text_critic_rerank"
    controlled_decisions: list[tuple[str, ActionDecision]] = []
    used_controller = False
    used_signatures: set[tuple[object, ...]] = set()
    snapshot_maturity = maturity_snapshot(snapshot)
    controller_state = {
        "round_index": _round_index(round_name),
        "support_coverage": snapshot_maturity.support_coverage,
        "unresolved_contradiction_ratio": snapshot_maturity.unresolved_contradiction_ratio,
        "utility": snapshot_maturity.utility,
        "utility_stable": snapshot_maturity.utility_stable,
        "completeness": snapshot_maturity.completeness,
        "is_mature": snapshot_maturity.is_mature,
        "utility_breakdown": asdict(snapshot_maturity.utility_breakdown),
    }
    graph_signals, graph_signal_deficits = _controller_graph_signal_payload(
        snapshot,
        round_name=round_name,
        controller_kind=controller_kind,
        controller_state=controller_state,
    )
    _record_graph_signal_trace(
        graph,
        round_name=round_name,
        stage="pre_edit",
        controller_kind=controller_kind,
        signals=graph_signals,
        deficits=graph_signal_deficits,
    )

    for role, decision in raw_decisions:
        candidate_specs = _controller_candidate_specs(
            snapshot,
            round_name=round_name,
            role=role,
            decision=decision,
        )
        valid_candidates = [candidate for candidate in candidate_specs if str(candidate.get("kind", "")).strip()]
        if not valid_candidates:
            controlled_decisions.append((role, decision))
            continue
        heuristic_candidate_id = str(valid_candidates[0].get("candidate_id", "")).strip()
        controller_decision = None
        if controller_kind in {"fixed_control", "random_control"}:
            choose_candidate = getattr(runtime_controller, "choose", None)
            if not callable(choose_candidate):
                controlled_decisions.append((role, decision))
                continue
            try:
                selected_raw = choose_candidate(
                    round_name=round_name,
                    role=role,
                    candidate_specs=valid_candidates,
                )
            except TypeError:
                selected_raw = choose_candidate(
                    round_name=round_name,
                    role=role,
                    candidates=valid_candidates,
                )
            selected_mapping = dict(selected_raw) if isinstance(selected_raw, Mapping) else {}
            selected_candidate_id = str(selected_mapping.get("candidate_id", "")).strip()
            selected_candidate = next(
                (
                    dict(candidate)
                    for candidate in valid_candidates
                    if str(candidate.get("candidate_id", "")).strip() == selected_candidate_id
                ),
                None,
            )
            if selected_candidate is None:
                selected_kind = str(selected_mapping.get("kind", "")).strip()
                selected_candidate = next(
                    (
                        dict(candidate)
                        for candidate in valid_candidates
                        if str(candidate.get("kind", "")).strip() == selected_kind
                    ),
                    dict(valid_candidates[0]),
                )
            selected_candidate_id = str(selected_candidate.get("candidate_id", "")).strip()
            selected_source = f"{controller_kind}_policy"
            scored_candidates = [
                {
                    **candidate,
                    "critic_score": 1.0
                    if str(candidate.get("candidate_id", "")).strip() == selected_candidate_id
                    else 0.0,
                    "controller_kind": controller_kind,
                }
                for candidate in valid_candidates
            ]
            selected_candidate = {
                **selected_candidate,
                "critic_score": 1.0,
                "controller_kind": controller_kind,
                "controller_selected_source": selected_source,
                "controller_override_margin": None,
                "controller_used_heuristic_fallback": False,
                "controller_fallback_reason": "",
                "controller_fallback_candidate_ids": (),
            }
            _record_runtime_controller_trace(
                graph,
                round_name=round_name,
                role=role,
                controller_kind=controller_kind,
                heuristic_candidate=dict(valid_candidates[0]),
                selected_candidate=selected_candidate,
                controller_decision={
                    "selected_source": selected_source,
                    "override_margin": None,
                    "used_heuristic_fallback": False,
                },
                scored_candidates=scored_candidates,
            )
            controlled_decisions.append((role, _action_decision_from_candidate(selected_candidate)))
            used_controller = True
            continue
        if controller_kind == "signal_heuristic_control":
            choose_candidate = getattr(runtime_controller, "choose", None)
            if not callable(choose_candidate):
                controlled_decisions.append((role, decision))
                continue
            selected_raw = choose_candidate(
                round_name=round_name,
                role=role,
                candidate_specs=valid_candidates,
                graph=snapshot,
            )
            selected_mapping = dict(selected_raw) if isinstance(selected_raw, Mapping) else {}
            selected_candidate_id = str(selected_mapping.get("candidate_id", "")).strip()
            scored_candidates = [
                dict(candidate)
                for candidate in selected_mapping.get("scored_candidates", ())
                if isinstance(candidate, Mapping)
            ]
            selected_candidate = next(
                (
                    dict(candidate)
                    for candidate in scored_candidates
                    if str(candidate.get("candidate_id", "")).strip() == selected_candidate_id
                ),
                None,
            )
            if selected_candidate is None:
                selected_candidate = next(
                    (
                        {
                            **candidate,
                            "critic_score": 0.0,
                            "critic_base_score": 0.0,
                            "critic_score_calibrated": 0.0,
                            "critic_calibration_bias": 0.0,
                            "critic_action_family": "heuristic",
                            "critic_aligned_signal": "",
                            "critic_calibration_enabled": False,
                            "critic_calibration_feedback": graph_signals,
                            "critic_calibration_deficits": graph_signal_deficits,
                        }
                        for candidate in valid_candidates
                        if str(candidate.get("candidate_id", "")).strip() == selected_candidate_id
                    ),
                    dict(valid_candidates[0]),
                )
            selected_candidate = {
                **selected_candidate,
                "controller_kind": controller_kind,
                "controller_selected_source": str(
                    selected_mapping.get("selected_source", "signal_heuristic_control")
                ).strip()
                or "signal_heuristic_control",
                "controller_override_margin": None,
                "controller_used_heuristic_fallback": False,
                "controller_fallback_reason": "",
                "controller_fallback_candidate_ids": (),
            }
            scored_candidates = scored_candidates or [selected_candidate]
            selected_candidate, diversity_override = _select_nonconflicting_candidate(
                scored_candidates,
                preferred_candidate=selected_candidate,
                used_signatures=used_signatures,
            )
            fallback_ids = ()
            if diversity_override:
                fallback_ids = (selected_candidate_id,) if selected_candidate_id else ()
            selected_candidate = {
                **selected_candidate,
                "controller_selected_source": (
                    "signal_heuristic_control_diverse"
                    if diversity_override
                    else selected_candidate["controller_selected_source"]
                ),
                "controller_fallback_reason": (
                    "diversity_avoid_duplicate" if diversity_override else ""
                ),
                "controller_fallback_candidate_ids": fallback_ids,
            }
            _record_runtime_controller_trace(
                graph,
                round_name=round_name,
                role=role,
                controller_kind=controller_kind,
                heuristic_candidate=dict(valid_candidates[0]),
                selected_candidate=selected_candidate,
                controller_decision={
                    "selected_source": selected_candidate["controller_selected_source"],
                    "override_margin": None,
                    "used_heuristic_fallback": False,
                },
                scored_candidates=scored_candidates,
            )
            if str(selected_candidate.get("kind", "")).strip() != "skip":
                used_signatures.add(_candidate_diversity_signature(selected_candidate))
            controlled_decisions.append((role, _action_decision_from_candidate(selected_candidate)))
            used_controller = True
            continue
        if controller_kind == "text_critic_rerank":
            controller_decision = select_text_critic_candidate(
                snapshot,
                round_name=round_name,
                role=role,
                state_features=controller_state,
                candidate_specs=valid_candidates,
                heuristic_candidate_id=heuristic_candidate_id,
                model=runtime_controller,
                config=controller_config,
            )
        elif controller_kind in {"relation_graph_critic_rerank", "relation_graph_two_head_critic"}:
            controller_decision = select_relation_graph_critic_candidate(
                snapshot,
                round_name=round_name,
                role=role,
                state_features=controller_state,
                candidate_specs=valid_candidates,
                heuristic_candidate_id=heuristic_candidate_id,
                runtime_bundle=runtime_controller,
                config=controller_config,
            )

        if controller_decision is None:
            controlled_decisions.append((role, decision))
            continue

        selected_candidate_id = str(controller_decision.policy_decision.selected_candidate_id)
        selected_candidate = next(
            (
                {
                    **candidate,
                    "critic_score": controller_decision.selected_spec.get("critic_score"),
                    "critic_base_score": controller_decision.selected_spec.get("critic_base_score"),
                    "critic_calibration_bias": controller_decision.selected_spec.get("critic_calibration_bias"),
                    "critic_score_calibrated": controller_decision.selected_spec.get("critic_score_calibrated"),
                    "critic_action_family": controller_decision.selected_spec.get("critic_action_family"),
                    "critic_calibration_enabled": controller_decision.selected_spec.get("critic_calibration_enabled"),
                    "critic_calibration_feedback": controller_decision.selected_spec.get("critic_calibration_feedback"),
                    "controller_kind": controller_kind,
                    "controller_selected_source": controller_decision.policy_decision.selected_source,
                    "controller_override_margin": controller_decision.policy_decision.override_margin,
                    "controller_used_heuristic_fallback": controller_decision.policy_decision.used_heuristic_fallback,
                    "controller_fallback_reason": controller_decision.selected_spec.get("controller_fallback_reason"),
                    "controller_fallback_candidate_ids": controller_decision.selected_spec.get("controller_fallback_candidate_ids"),
                }
                for candidate in valid_candidates
                if str(candidate.get("candidate_id", "")).strip() == selected_candidate_id
            ),
            dict(valid_candidates[0]),
        )
        _record_runtime_controller_trace(
            graph,
            round_name=round_name,
            role=role,
            controller_kind=controller_kind,
            heuristic_candidate=dict(valid_candidates[0]),
            selected_candidate=selected_candidate,
            controller_decision={
                "selected_source": controller_decision.policy_decision.selected_source,
                "override_margin": controller_decision.policy_decision.override_margin,
                "used_heuristic_fallback": controller_decision.policy_decision.used_heuristic_fallback,
            },
            scored_candidates=controller_decision.scored_candidates,
        )
        controlled_decisions.append((role, _action_decision_from_candidate(selected_candidate)))
        used_controller = True

    return controlled_decisions, used_controller


def _runtime_commit_check(
    graph,
    *,
    round_name: str,
    post_round_snapshot,
    runtime_controller: Any | None,
    runtime_controller_metadata: dict[str, Any] | None,
) -> ParallelCommitCheckRecord:
    controller_state = {
        "round_index": _round_index(round_name),
        "support_coverage": float(getattr(post_round_snapshot, "support_coverage", 0.0)),
        "unresolved_contradiction_ratio": float(
            getattr(post_round_snapshot, "unresolved_contradiction_ratio", 0.0)
        ),
        "utility": float(getattr(post_round_snapshot, "utility", 0.0)),
        "utility_stable": bool(getattr(post_round_snapshot, "utility_stable", False)),
        "completeness": bool(getattr(post_round_snapshot, "completeness", False)),
        "is_mature": bool(getattr(post_round_snapshot, "is_mature", False)),
        "utility_breakdown": _mapping_payload(
            getattr(post_round_snapshot, "utility_breakdown", {})
        ),
    }
    controller_config = runtime_controller_metadata.get("config") if runtime_controller_metadata else None
    controller_kind = str((runtime_controller_metadata or {}).get("kind", "")).strip()
    graph_signals, graph_signal_deficits = _controller_graph_signal_payload(
        graph,
        round_name=round_name,
        controller_kind=controller_kind,
        controller_state=controller_state,
    )
    _record_graph_signal_trace(
        graph,
        round_name=round_name,
        stage="post_round",
        controller_kind=controller_kind or "maturity_snapshot",
        signals=graph_signals,
        deficits=graph_signal_deficits,
    )
    score_commit_graph = getattr(runtime_controller, "score_commit_graph", None)
    if (
        runtime_controller is not None
        and controller_config is not None
        and callable(score_commit_graph)
        and bool(getattr(controller_config, "use_commit", False))
        and _round_index(round_name) >= int(getattr(controller_config, "min_commit_round", 0))
    ):
        round_index = _round_index(round_name)
        probability = float(score_commit_graph(graph, snapshot=post_round_snapshot))
        if controller_kind == "signal_heuristic_control":
            calibrated_probability = probability
            calibration_bias = 0.0
            calibration_enabled = False
            calibration_feedback = dict(graph_signals)
        else:
            (
                calibrated_probability,
                calibration_bias,
                calibration_enabled,
                calibration_feedback,
            ) = calibrate_commit_probability(
                graph,
                raw_probability=probability,
                state_features=controller_state,
                config=controller_config,
                feedback=graph_signals,
                deficits=graph_signal_deficits,
            )
        threshold = commit_threshold_for_round(
            round_index=round_index,
            default_threshold=float(getattr(controller_config, "gamma_commit", 0.50)),
            thresholds_by_round=getattr(controller_config, "gamma_commit_by_round", None),
        )
        guard_reason = ""
        support_threshold = float(getattr(controller_config, "guard_commit_support_threshold", 0.0))
        utility_floor = float(getattr(controller_config, "guard_commit_utility_floor", 0.0))
        if (
            calibrated_probability >= threshold
            and support_threshold > 0.0
            and float(post_round_snapshot.support_coverage) < support_threshold
        ):
            guard_reason = "support_below_commit_guard"
        elif (
            calibrated_probability >= threshold
            and utility_floor > 0.0
            and float(post_round_snapshot.utility) < utility_floor
        ):
            guard_reason = "utility_below_commit_guard"
        should_commit = calibrated_probability >= threshold and not guard_reason
        return ParallelCommitCheckRecord(
            round_name=round_name,
            state_kind="parallel_post_round",
            should_commit=should_commit,
            source="runtime_controller_commit" if should_commit else "runtime_controller_continue",
            support_coverage=float(post_round_snapshot.support_coverage),
            unresolved_contradiction_ratio=float(post_round_snapshot.unresolved_contradiction_ratio),
            utility=float(post_round_snapshot.utility),
            controller_kind=controller_kind,
            commit_probability=probability,
            commit_probability_calibrated=calibrated_probability,
            commit_threshold=threshold,
            commit_calibration_bias=calibration_bias,
            commit_calibration_enabled=calibration_enabled,
            commit_calibration_feedback=calibration_feedback,
            graph_signals=graph_signals,
            graph_signal_deficits=graph_signal_deficits,
            commit_guard_reason=guard_reason,
        )

    return ParallelCommitCheckRecord(
        round_name=round_name,
        state_kind="parallel_post_round",
        should_commit=bool(post_round_snapshot.is_mature),
        source="maturity_snapshot",
        support_coverage=float(post_round_snapshot.support_coverage),
        unresolved_contradiction_ratio=float(post_round_snapshot.unresolved_contradiction_ratio),
        utility=float(post_round_snapshot.utility),
        graph_signals=graph_signals,
        graph_signal_deficits=graph_signal_deficits,
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
    del progress_callback
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
        for role, decision in raw_decisions:
            append_agent_trace(
                graph,
                stage=f"{round_name}_action",
                role=role,
                trace=decision.trace,
            )
        action_source = "parallel_llm"
        label_source = "parallel_runtime_logged_v1"
    raw_decisions, used_controller = _maybe_apply_runtime_controller(
        graph,
        snapshot=snapshot,
        round_name=round_name,
        raw_decisions=raw_decisions,
        runtime_controller=runtime_controller,
        runtime_controller_metadata=runtime_controller_metadata,
    )
    if used_controller:
        action_source = "parallel_controller"
        label_source = "parallel_runtime_controller_v1"
    edit_rows = build_parallel_edit_rows(
        snapshot,
        round_name=round_name,
        role_decisions=raw_decisions,
        runtime_protocol="parallel_graph_v2",
        label_source=label_source,
    )
    selected_role_decisions = []
    edit_patches = []
    materialized_graph_actions = []
    skipped_roles = []
    for role, decision in sorted(raw_decisions, key=lambda item: item[0]):
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
    )
    post_round_commit_row = build_post_round_commit_row(
        graph,
        round_name=round_name,
        commit_check=post_round_commit,
        runtime_protocol="parallel_graph_v2",
        label_source=post_round_commit.source,
    )
    return ParallelRoleRoundResult(
        round_name=round_name,
        active_roles=tuple(role for role, _ in raw_decisions),
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


__all__ = ["ParallelRoleRoundResult", "execute_parallel_role_round"]
