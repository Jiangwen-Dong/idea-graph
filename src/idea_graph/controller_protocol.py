from __future__ import annotations

from .schema import ROLE_NAMES


ACTIVE_EDIT_ACTIONS = (
    "add_support_edge",
    "attach_evidence",
    "add_dependency_edge",
    "add_contradiction_edge",
    "propose_repair",
    "skip",
)

CONTROLLER_ACTIONS = ("commit",)

SIGNAL_NAMES = (
    "grounding",
    "contradiction_load",
    "completeness",
    "maturity",
)

LEGACY_INACTIVE_ACTIONS = (
    "freeze_branch",
    "request_evidence",
    "mark_overlap",
)

ACTION_PRIMARY_SIGNAL = {
    "add_support_edge": "grounding",
    "attach_evidence": "grounding",
    "add_dependency_edge": "completeness",
    "add_contradiction_edge": "contradiction_load",
    "propose_repair": "contradiction_load",
    "skip": "maturity",
    "commit": "maturity",
}

ROLE_TO_ALLOWED_ACTIONS = {
    "ImpactReframer": (
        "add_support_edge",
        "attach_evidence",
        "propose_repair",
        "skip",
    ),
    "MechanismProposer": (
        "add_support_edge",
        "add_dependency_edge",
        "attach_evidence",
        "propose_repair",
        "skip",
    ),
    "FeasibilityCritic": (
        "add_contradiction_edge",
        "attach_evidence",
        "propose_repair",
        "skip",
    ),
    "NoveltyExaminer": (
        "add_contradiction_edge",
        "attach_evidence",
        "propose_repair",
        "skip",
    ),
    "EvaluationDesigner": (
        "add_dependency_edge",
        "add_support_edge",
        "attach_evidence",
        "propose_repair",
        "skip",
    ),
}


def is_active_edit_action(kind: str) -> bool:
    return str(kind).strip() in ACTIVE_EDIT_ACTIONS


def is_controller_action(kind: str) -> bool:
    return str(kind).strip() in CONTROLLER_ACTIONS


def primary_signal_for_action(kind: str) -> str:
    return ACTION_PRIMARY_SIGNAL.get(str(kind).strip(), "")


def allowed_actions_for_role(role: str) -> tuple[str, ...]:
    normalized_role = str(role).strip()
    if normalized_role in ROLE_TO_ALLOWED_ACTIONS:
        return ROLE_TO_ALLOWED_ACTIONS[normalized_role]
    if normalized_role in ROLE_NAMES:
        return ("skip",)
    return tuple()


def is_action_allowed_for_role(role: str, kind: str) -> bool:
    return str(kind).strip() in set(allowed_actions_for_role(role))
