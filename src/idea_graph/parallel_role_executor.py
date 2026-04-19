from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from .agent_backend import ActionDecision


def _skip_decision_from_invalid_action(role: str, exc: ValueError) -> ActionDecision:
    return ActionDecision(
        kind="skip",
        target_ids=[],
        payload={},
        rationale=(
            f"Skipped {role}'s role-local edit because the proposed action was invalid: {exc}"
        ),
        trace={
            "fallback_action": "skip",
            "fallback_reason": "invalid_role_action",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )


def collect_parallel_role_decisions(graph, round_name, collaboration_backend, roles):
    snapshot = deepcopy(graph)

    def _run(role: str):
        try:
            return role, collaboration_backend.choose_action(snapshot, round_name, role)
        except ValueError as exc:
            return role, _skip_decision_from_invalid_action(role, exc)

    with ThreadPoolExecutor(max_workers=max(1, len(roles))) as pool:
        return list(pool.map(_run, roles))
