from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ScoredCandidate:
    candidate_id: str
    score: float
    is_commit: bool = False
    confidence: float | None = None
    predicted_gain: float = 0.0
    support_gain: float = 0.0
    contradiction_gain: float = 0.0
    maturity_gain: float = 0.0
    after_is_mature: bool = False


@dataclass(frozen=True)
class SafeCriticPolicyConfig:
    min_commit_round: int = 2
    tau_override: float = 0.05
    tau_override_by_round: Mapping[int | str, float] = field(default_factory=dict)
    tau_commit: float = 0.08
    gamma_commit: float = 0.60
    gamma_commit_by_round: Mapping[int | str, float] = field(default_factory=dict)
    guard_support_threshold: float = 0.66
    guard_support_gain_floor: float = 0.10
    guard_requires_contradiction_progress: bool = False
    guard_commit_support_threshold: float = 0.0
    guard_commit_utility_floor: float = 0.0
    guard_predicted_gain_min_heuristic: float = 0.05
    guard_predicted_gain_ratio: float = 0.70
    guard_predicted_gain_slack: float = 0.10


@dataclass(frozen=True)
class CriticPolicyDecision:
    selected_candidate_id: str
    selected_source: str
    used_heuristic_fallback: bool
    commit_allowed: bool
    commit_requested: bool
    override_margin: float
    commit_margin: float | None
    fallback_reason: str | None = None


def _state_round_index(state: Mapping[str, object]) -> int:
    try:
        return int(state.get("round_index", 0))
    except (TypeError, ValueError):
        return 0


def _best_candidate(candidates: Sequence[ScoredCandidate]) -> ScoredCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.score, candidate.candidate_id))


def _state_float(state: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(state.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def commit_threshold_for_round(
    *,
    round_index: int,
    default_threshold: float,
    thresholds_by_round: Mapping[int | str, float] | None = None,
) -> float:
    if not thresholds_by_round:
        return float(default_threshold)
    string_key = str(int(round_index))
    for key, value in thresholds_by_round.items():
        if str(key).strip() == string_key:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default_threshold)
    return float(default_threshold)


def commit_guard_reason(
    *,
    state: Mapping[str, object],
    config: SafeCriticPolicyConfig,
) -> str | None:
    support_threshold = float(config.guard_commit_support_threshold)
    if support_threshold > 0.0 and _state_float(state, "support_coverage") < support_threshold:
        return "support_below_commit_guard"
    utility_floor = float(config.guard_commit_utility_floor)
    if utility_floor > 0.0 and _state_float(state, "utility") < utility_floor:
        return "utility_below_commit_guard"
    return None


def _is_fragile_maturity_override(
    *,
    state: Mapping[str, object],
    candidate: ScoredCandidate,
    config: SafeCriticPolicyConfig,
) -> bool:
    if not candidate.after_is_mature:
        return False
    if candidate.maturity_gain <= 0.0:
        return False
    support_coverage = _state_float(state, "support_coverage")
    if support_coverage < float(config.guard_support_threshold):
        return False
    if candidate.support_gain >= float(config.guard_support_gain_floor):
        return False
    if bool(config.guard_requires_contradiction_progress) and candidate.contradiction_gain > 0.0:
        return False
    return True


def _is_predicted_gain_dominated(
    *,
    candidate: ScoredCandidate,
    heuristic_candidate: ScoredCandidate,
    config: SafeCriticPolicyConfig,
) -> bool:
    heuristic_gain = float(heuristic_candidate.predicted_gain)
    if heuristic_gain < float(config.guard_predicted_gain_min_heuristic):
        return False
    candidate_gain = float(candidate.predicted_gain)
    if candidate_gain + float(config.guard_predicted_gain_slack) >= heuristic_gain:
        return False
    if candidate_gain >= heuristic_gain * float(config.guard_predicted_gain_ratio):
        return False
    return True


def choose_critic_action(
    *,
    state: Mapping[str, object],
    critic_candidates: Sequence[ScoredCandidate],
    heuristic_candidate: ScoredCandidate,
    config: SafeCriticPolicyConfig,
) -> CriticPolicyDecision:
    commit_candidates = [candidate for candidate in critic_candidates if candidate.is_commit]
    edit_candidates = [candidate for candidate in critic_candidates if not candidate.is_commit]
    best_commit = _best_candidate(commit_candidates)
    best_edit = _best_candidate(edit_candidates)

    round_index = _state_round_index(state)
    commit_requested = best_commit is not None
    commit_allowed = commit_requested and round_index >= int(config.min_commit_round)
    commit_margin: float | None = None
    commit_rejection_reason: str | None = None

    if best_commit is not None and commit_allowed:
        best_edit_score = best_edit.score if best_edit is not None else heuristic_candidate.score
        commit_margin = float(best_commit.score - best_edit_score)
        commit_confidence = (
            float(best_commit.confidence)
            if best_commit.confidence is not None
            else float(best_commit.score)
        )
        commit_threshold = commit_threshold_for_round(
            round_index=round_index,
            default_threshold=float(config.gamma_commit),
            thresholds_by_round=config.gamma_commit_by_round,
        )
        guard_reason = commit_guard_reason(state=state, config=config)
        if guard_reason is not None:
            commit_rejection_reason = guard_reason
        elif commit_margin < float(config.tau_commit):
            commit_rejection_reason = "commit_margin_below_threshold"
        elif commit_confidence < commit_threshold:
            commit_rejection_reason = "commit_confidence_below_round_threshold"
        else:
            return CriticPolicyDecision(
                selected_candidate_id=best_commit.candidate_id,
                selected_source="critic",
                used_heuristic_fallback=False,
                commit_allowed=True,
                commit_requested=True,
                override_margin=float(best_commit.score - heuristic_candidate.score),
                commit_margin=commit_margin,
                fallback_reason=None,
            )

    if best_edit is not None:
        override_margin = float(best_edit.score - heuristic_candidate.score)
        override_threshold = commit_threshold_for_round(
            round_index=round_index,
            default_threshold=float(config.tau_override),
            thresholds_by_round=config.tau_override_by_round,
        )
        if override_margin >= override_threshold:
            if _is_predicted_gain_dominated(
                candidate=best_edit,
                heuristic_candidate=heuristic_candidate,
                config=config,
            ):
                return CriticPolicyDecision(
                    selected_candidate_id=heuristic_candidate.candidate_id,
                    selected_source="heuristic",
                    used_heuristic_fallback=True,
                    commit_allowed=commit_allowed,
                    commit_requested=commit_requested,
                    override_margin=override_margin,
                    commit_margin=commit_margin,
                    fallback_reason="predicted_gain_guard",
                )
            if _is_fragile_maturity_override(
                state=state,
                candidate=best_edit,
                config=config,
            ):
                return CriticPolicyDecision(
                    selected_candidate_id=heuristic_candidate.candidate_id,
                    selected_source="heuristic",
                    used_heuristic_fallback=True,
                    commit_allowed=commit_allowed,
                    commit_requested=commit_requested,
                    override_margin=override_margin,
                    commit_margin=commit_margin,
                    fallback_reason="fragile_maturity_guard",
                )
            return CriticPolicyDecision(
                selected_candidate_id=best_edit.candidate_id,
                selected_source="critic",
                used_heuristic_fallback=False,
                commit_allowed=commit_allowed,
                commit_requested=commit_requested,
                override_margin=override_margin,
                commit_margin=commit_margin,
                fallback_reason=commit_rejection_reason,
            )
    else:
        override_margin = float("-inf")

    return CriticPolicyDecision(
        selected_candidate_id=heuristic_candidate.candidate_id,
        selected_source="heuristic",
        used_heuristic_fallback=True,
        commit_allowed=commit_allowed,
        commit_requested=commit_requested,
        override_margin=override_margin,
        commit_margin=commit_margin,
        fallback_reason="override_margin_below_threshold" if best_edit is not None else "no_edit_candidates",
    )
