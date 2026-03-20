from __future__ import annotations

from dataclasses import dataclass

ROLE_NAMES = (
    "MechanismProposer",
    "FeasibilityCritic",
    "NoveltyExaminer",
    "EvaluationDesigner",
    "ImpactReframer",
)

NODE_TYPES = (
    "Problem",
    "Hypothesis",
    "Method",
    "Assumption",
    "Risk",
    "EvidenceNeed",
    "EvalPlan",
    "NoveltyClaim",
    "Repair",
)

EDGE_TYPES = (
    "supports",
    "contradicts",
    "refines",
    "depends_on",
    "requires_evidence",
    "overlaps_prior",
    "repairs",
)


@dataclass(frozen=True)
class SeedTemplate:
    anchor_type: str
    anchor_text: str
    support_nodes: tuple[tuple[str, str], ...]


def build_seed_template(role: str, topic: str) -> SeedTemplate:
    if role == "MechanismProposer":
        return SeedTemplate(
            anchor_type="Hypothesis",
            anchor_text=(
                f"A typed graph with delayed consensus can improve ideation for: {topic}"
            ),
            support_nodes=(
                ("Method", "Represent partial claims as typed nodes instead of full drafts."),
                ("Assumption", "Role diversity increases perspective coverage."),
            ),
        )

    if role == "FeasibilityCritic":
        return SeedTemplate(
            anchor_type="Risk",
            anchor_text="The graph may become noisy unless edit actions are tightly constrained.",
            support_nodes=(
                ("Assumption", "A compact schema is easier to maintain than a rich ontology."),
                ("EvalPlan", "Monitor support coverage and contradiction repair after each round."),
            ),
        )

    if role == "NoveltyExaminer":
        return SeedTemplate(
            anchor_type="NoveltyClaim",
            anchor_text=(
                "The key novelty is delayed consensus over claim-level graph units rather than "
                "voting over whole ideas."
            ),
            support_nodes=(
                ("EvidenceNeed", "Compare against discussion-plus-voting and single-agent baselines."),
            ),
        )

    if role == "EvaluationDesigner":
        return SeedTemplate(
            anchor_type="EvalPlan",
            anchor_text=(
                "Evaluate novelty, feasibility, coherence, significance, and process dynamics."
            ),
            support_nodes=(
                ("Method", "Run three collaboration rounds after merging seed graphs."),
                ("Risk", "Without budget matching, gains may reflect extra compute rather than collaboration."),
            ),
        )

    if role == "ImpactReframer":
        return SeedTemplate(
            anchor_type="Problem",
            anchor_text=(
                "Scientific ideation systems often collapse to persuasive full drafts too early."
            ),
            support_nodes=(
                ("Hypothesis", "Preserving disagreement can improve final proposal quality."),
                ("NoveltyClaim", "Rejected and frozen branches can remain interpretable traces."),
            ),
        )

    raise ValueError(f"Unknown role: {role}")
