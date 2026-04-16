from __future__ import annotations

from .schema import ROLE_NAMES


def active_roles_for_round(graph, round_name: str) -> tuple[str, ...]:
    del graph, round_name
    return tuple(ROLE_NAMES)
