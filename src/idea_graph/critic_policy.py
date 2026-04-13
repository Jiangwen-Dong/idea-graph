from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ScoredCandidate:
    candidate_id: str
    score: float
    is_commit: bool = False
    confidence: float | None = None


@dataclass(frozen=True)
class SafeCriticPolicyConfig:
    min_commit_round: int = 2
    tau_override: float = 0.05
    tau_commit: float = 0.08
    gamma_commit: float = 0.60


@dataclass(frozen=True)
class CriticPolicyDecision:
    selected_candidate_id: str
    selected_source: str
    used_heuristic_fallback: bool
    commit_allowed: bool
    commit_requested: bool
    override_margin: float
    commit_margin: float | None


def _state_round_index(state: Mapping[str, object]) -> int:
    try:
        return int(state.get("round_index", 0))
    except (TypeError, ValueError):
        return 0


def _best_candidate(candidates: Sequence[ScoredCandidate]) -> ScoredCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.score, candidate.candidate_id))


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

    if best_commit is not None and commit_allowed:
        best_edit_score = best_edit.score if best_edit is not None else heuristic_candidate.score
        commit_margin = float(best_commit.score - best_edit_score)
        commit_confidence = (
            float(best_commit.confidence)
            if best_commit.confidence is not None
            else float(best_commit.score)
        )
        if commit_margin >= float(config.tau_commit) and commit_confidence >= float(
            config.gamma_commit
        ):
            return CriticPolicyDecision(
                selected_candidate_id=best_commit.candidate_id,
                selected_source="critic",
                used_heuristic_fallback=False,
                commit_allowed=True,
                commit_requested=True,
                override_margin=float(best_commit.score - heuristic_candidate.score),
                commit_margin=commit_margin,
            )

    if best_edit is not None:
        override_margin = float(best_edit.score - heuristic_candidate.score)
        if override_margin >= float(config.tau_override):
            return CriticPolicyDecision(
                selected_candidate_id=best_edit.candidate_id,
                selected_source="critic",
                used_heuristic_fallback=False,
                commit_allowed=commit_allowed,
                commit_requested=commit_requested,
                override_margin=override_margin,
                commit_margin=commit_margin,
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
    )
