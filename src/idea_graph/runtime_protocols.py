from __future__ import annotations

from typing import Sequence


PARALLEL_GRAPH_V2 = "parallel_graph_v2"
SEQUENTIAL_GRAPH_V2 = "sequential_graph_v2"

ROLE_ORDER_PRESETS: dict[str, tuple[str, ...]] = {
    "order_a_canonical": (
        "MechanismProposer",
        "FeasibilityCritic",
        "NoveltyExaminer",
        "EvaluationDesigner",
        "ImpactReframer",
    ),
    "order_b_reverse": (
        "ImpactReframer",
        "EvaluationDesigner",
        "NoveltyExaminer",
        "FeasibilityCritic",
        "MechanismProposer",
    ),
    "order_c_cyclic": (
        "NoveltyExaminer",
        "EvaluationDesigner",
        "ImpactReframer",
        "MechanismProposer",
        "FeasibilityCritic",
    ),
}


def resolve_role_order(order_id: str, active_roles: Sequence[str]) -> tuple[str, ...]:
    try:
        preset = ROLE_ORDER_PRESETS[order_id]
    except KeyError as exc:
        options = ", ".join(sorted(ROLE_ORDER_PRESETS))
        raise ValueError(f"Unknown role order '{order_id}'. Expected one of: {options}.") from exc

    active_role_set = {str(role).strip() for role in active_roles if str(role).strip()}
    unknown_roles = sorted(active_role_set.difference(preset))
    if unknown_roles:
        raise ValueError(f"Unknown active roles for '{order_id}': {', '.join(unknown_roles)}.")

    resolved = tuple(role for role in preset if role in active_role_set)
    if not resolved:
        raise ValueError(f"Role order '{order_id}' resolved to an empty active-role subset.")
    return resolved
