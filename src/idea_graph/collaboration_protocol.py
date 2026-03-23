from __future__ import annotations

from dataclasses import dataclass
import re


ACTION_TARGET_COUNTS = {
    "add_support_edge": 2,
    "add_contradiction_edge": 2,
    "add_dependency_edge": 2,
    "request_evidence": 1,
    "attach_evidence": 1,
    "mark_overlap": 1,
    "propose_repair": 1,
    "freeze_branch": 0,
}

ACTION_REQUIRED_PAYLOAD_FIELDS = {
    "attach_evidence": ("evidence",),
    "mark_overlap": ("paper_id",),
    "propose_repair": ("repair_text",),
}

ACTION_PROMPT_HINTS = {
    "add_support_edge": {
        "target_shape": ["source_node_id", "target_node_id"],
        "payload_fields": ["branch_id"],
        "when_to_use": "Use when one existing node clearly strengthens or justifies another existing node.",
    },
    "add_contradiction_edge": {
        "target_shape": ["source_node_id", "target_node_id"],
        "payload_fields": ["branch_id"],
        "when_to_use": "Use when one existing node directly challenges feasibility, validity, or scope of another.",
    },
    "add_dependency_edge": {
        "target_shape": ["dependent_node_id", "prerequisite_node_id"],
        "payload_fields": ["branch_id"],
        "when_to_use": "Use when one claim or plan only makes sense if another claim holds first.",
    },
    "request_evidence": {
        "target_shape": ["node_id"],
        "payload_fields": ["branch_id", "query"],
        "when_to_use": "Use when a claim is plausible but needs explicit grounding before it should influence synthesis.",
    },
    "attach_evidence": {
        "target_shape": ["node_id"],
        "payload_fields": ["branch_id", "evidence"],
        "when_to_use": "Use when a node lacks grounding and one literature item can materially strengthen it.",
    },
    "mark_overlap": {
        "target_shape": ["node_id"],
        "payload_fields": ["branch_id", "paper_id", "evidence"],
        "when_to_use": "Use when a novelty claim or idea fragment appears close to prior work and should be marked as overlap.",
    },
    "propose_repair": {
        "target_shape": ["node_id"],
        "payload_fields": ["branch_id", "repair_text"],
        "when_to_use": "Use when a contradiction or weakness can be addressed by revising a target node with a concrete repair.",
    },
    "freeze_branch": {
        "target_shape": [],
        "payload_fields": ["branch_id"],
        "when_to_use": "Use when a branch is coherent enough to preserve as an alternative and should stop expanding for now.",
    },
}


@dataclass(frozen=True)
class RoundPhase:
    key: str
    title: str
    objective: str
    allowed_actions: tuple[str, ...]
    decision_focus: tuple[str, ...]


ROUND_PHASES = {
    "structure": RoundPhase(
        key="structure",
        title="Structure Expansion",
        objective="Expose graph structure by adding support, contradiction, dependency, overlap, or evidence-request signals.",
        allowed_actions=(
            "add_support_edge",
            "add_contradiction_edge",
            "add_dependency_edge",
            "mark_overlap",
            "request_evidence",
        ),
        decision_focus=(
            "Prefer actions that reveal latent structure rather than repeating edges that already exist.",
            "Use contradiction or dependency edges only when the relation is concrete and locally justified by the snapshot.",
            "If novelty is uncertain, requesting evidence or marking overlap is often better than asserting support.",
        ),
    ),
    "stress_test": RoundPhase(
        key="stress_test",
        title="Stress Test",
        objective="Stress-test branches created by other roles using evidence attachment, overlap marking, or critique.",
        allowed_actions=(
            "attach_evidence",
            "mark_overlap",
            "request_evidence",
            "add_support_edge",
            "add_contradiction_edge",
        ),
        decision_focus=(
            "Prefer actions that interrogate unsupported claims, unresolved novelty, or cross-branch weaknesses.",
            "Attach evidence when one literature item can clearly improve a weak node.",
            "Use contradiction edges sparingly and only for substantive failure modes rather than mild disagreement.",
        ),
    ),
    "repair": RoundPhase(
        key="repair",
        title="Repair And Consolidation",
        objective="Repair or recombine the graph into stronger branches without collapsing into early whole-idea voting.",
        allowed_actions=(
            "propose_repair",
            "add_support_edge",
            "freeze_branch",
            "attach_evidence",
        ),
        decision_focus=(
            "When unresolved contradictions block maturity, consider whether a repair can directly address them.",
            "Do not default to repair or support automatically; choose the single highest-leverage action for the current graph state.",
            "Freeze a branch only when preserving it as an alternative is more valuable than editing it further right now.",
        ),
    ),
}


def build_round_name(round_index: int) -> str:
    return f"Round{max(1, round_index)}"


def round_index_from_name(round_name: str) -> int:
    match = re.search(r"(\d+)$", round_name)
    if match is None:
        return 1
    return max(1, int(match.group(1)))


def resolve_round_phase(round_name: str) -> RoundPhase:
    round_index = round_index_from_name(round_name)
    if round_index == 1:
        return ROUND_PHASES["structure"]
    if round_index % 2 == 0:
        return ROUND_PHASES["stress_test"]
    return ROUND_PHASES["repair"]
