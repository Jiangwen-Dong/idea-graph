from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from .agent_backend import ActionDecision


def _skip_decision_from_role_error(role: str, exc: Exception) -> ActionDecision:
    fallback_reason = "invalid_role_action" if isinstance(exc, ValueError) else "role_action_runtime_error"
    return ActionDecision(
        kind="skip",
        target_ids=[],
        payload={},
        rationale=(
            f"Skipped {role}'s role-local edit because the proposed action could not be used: {exc}"
        ),
        trace={
            "fallback_action": "skip",
            "fallback_reason": fallback_reason,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )


def collect_parallel_role_decisions(graph, round_name, collaboration_backend, roles):
    snapshot = deepcopy(graph)

    def _run(role: str):
        try:
            return role, collaboration_backend.choose_action(snapshot, round_name, role)
        except (RuntimeError, ValueError) as exc:
            return role, _skip_decision_from_role_error(role, exc)

    with ThreadPoolExecutor(max_workers=max(1, len(roles))) as pool:
        return list(pool.map(_run, roles))
