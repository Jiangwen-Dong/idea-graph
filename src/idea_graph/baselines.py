from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .agent_backend import OpenAICompatibleCollaborationBackend
from .benchmark_mode import apply_io_mode
from .engine import emit_progress, run_experiment
from .fixed_control_policy import load_fixed_control_policy
from .random_control_policy import RandomControlPolicy
from .joint_controller_calibration import (
    apply_joint_controller_calibration,
    load_joint_controller_calibration,
)
from .literature_grounding import build_literature_grounding
from .models import FinalProposal, IdeaGraph
from .relation_graph_runtime_critic import (
    RelationGraphRuntimeConfig,
    load_relation_graph_runtime_bundle,
)
from .relation_graph_two_head_runtime_critic import load_relation_graph_two_head_runtime_bundle
from .runtime_critic import TextCriticRuntimeConfig, load_pickled_text_critic_model
from .signal_heuristic_control import SignalHeuristicController


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    display_name: str
    strategy: str
    description: str
    is_local_variant: bool = False
    reference_target: str = ""
    prompt_style: str = ""
    candidate_count: int = 1
    runtime_controller: str = ""


ROOT = Path(__file__).resolve().parents[2]
TEXT_CRITIC_MODEL_RELATIVE_PATH = (
    Path("outputs")
    / "graph_critic_models"
    / "current_benchmarked_ours_eig_full_g46_text_online_real_train_v1"
    / "model.pkl"
)
RELATION_GRAPH_CRITIC_MODEL_RELATIVE_DIR = (
    Path("outputs")
    / "graph_critic_models"
    / "development_pool_v3_relation_graph_sanitized_v1"
)
RELATION_GRAPH_TWO_HEAD_MODEL_RELATIVE_DIR = (
    Path("outputs")
    / "critic_models"
    / "parallel_v2_twohead_gold256_st_e8_20260423"
)
TRACKED_JOINT_CONTROLLER_CALIBRATION_RELATIVE_PATH = (
    Path("configs")
    / "joint_controller_calibration.json"
)
FIXED_CONTROL_POLICY_RELATIVE_PATH = (
    Path("configs")
    / "fixed_control_policy.example.json"
)


RUNTIME_CONTROLLER_METADATA_KEYS = (
    "runtime_controller_enabled",
    "runtime_controller_kind",
    "runtime_controller_use_edit",
    "runtime_controller_use_commit",
    "runtime_controller_tau_override",
    "runtime_controller_tau_override_by_round",
    "runtime_controller_tau_commit",
    "runtime_controller_gamma_commit",
    "runtime_controller_gamma_commit_by_round",
    "runtime_controller_min_commit_round",
    "runtime_controller_use_low_signal_kind_swap_guard",
    "runtime_controller_guard_support_threshold",
    "runtime_controller_guard_support_gain_floor",
    "runtime_controller_guard_requires_contradiction_progress",
    "runtime_controller_guard_commit_support_threshold",
    "runtime_controller_guard_commit_utility_floor",
    "runtime_controller_use_action_score_calibration",
    "runtime_controller_action_score_calibration_strength",
    "runtime_controller_action_score_calibration_max_bias",
    "runtime_controller_model_path",
    "runtime_controller_model_dir",
    "runtime_controller_policy_path",
    "runtime_controller_random_seed",
    "runtime_controller_calibration_path",
    "runtime_controller_calibration_source",
    "runtime_controller_calibration_version",
    "runtime_controller_calibration_missing",
    "runtime_controller_calibration_missing_path",
    "runtime_controller_use_joint_threshold_calibration",
    "runtime_controller_disable_calibration",
    "runtime_controller_error",
    "runtime_controller_loaded",
    "max_rounds_hint",
)

GRAPH_OF_THOUGHT_ROUND_COUNT = 3


def _shared_repo_root_from_worktree(root: Path) -> Path | None:
    if root.parent.name != ".worktrees":
        return None
    return root.parent.parent


def _default_graph_critic_models_root() -> Path:
    shared_root = _shared_repo_root_from_worktree(ROOT)
    if shared_root is not None:
        return shared_root
    return ROOT


def _default_text_critic_model_path() -> Path:
    return (_default_graph_critic_models_root() / TEXT_CRITIC_MODEL_RELATIVE_PATH).resolve()


def _default_relation_graph_runtime_model_dir() -> Path:
    return (_default_graph_critic_models_root() / RELATION_GRAPH_CRITIC_MODEL_RELATIVE_DIR).resolve()


def _default_relation_graph_two_head_runtime_model_dir() -> Path:
    return (_default_graph_critic_models_root() / RELATION_GRAPH_TWO_HEAD_MODEL_RELATIVE_DIR).resolve()


def _default_joint_controller_calibration_path() -> Path:
    return (ROOT / TRACKED_JOINT_CONTROLLER_CALIBRATION_RELATIVE_PATH).resolve()


def _default_fixed_control_policy_path() -> Path:
    return (ROOT / FIXED_CONTROL_POLICY_RELATIVE_PATH).resolve()


DEFAULT_TEXT_CRITIC_MODEL_PATH = _default_text_critic_model_path()
TWO_HEAD_RUNTIME_CONTROLLER_DEFAULTS: dict[str, Any] = {
    "runtime_controller_use_edit": True,
    "runtime_controller_tau_override": 0.05,
    "runtime_controller_tau_override_by_round": {},
    "runtime_controller_tau_commit": 0.08,
    "runtime_controller_gamma_commit": 0.50,
    "runtime_controller_gamma_commit_by_round": {},
    "runtime_controller_min_commit_round": 3,
    "runtime_controller_use_low_signal_kind_swap_guard": False,
    "runtime_controller_guard_support_threshold": 0.66,
    "runtime_controller_guard_support_gain_floor": 0.10,
    "runtime_controller_guard_requires_contradiction_progress": False,
    "runtime_controller_guard_commit_support_threshold": 0.0,
    "runtime_controller_guard_commit_utility_floor": 0.0,
    "runtime_controller_use_action_score_calibration": True,
    "runtime_controller_action_score_calibration_strength": 0.35,
    "runtime_controller_action_score_calibration_max_bias": 0.35,
}


def reset_two_head_runtime_controller_defaults(metadata: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(metadata)
    updated.update(TWO_HEAD_RUNTIME_CONTROLLER_DEFAULTS)
    updated.pop("runtime_controller_calibration_path", None)
    updated.pop("runtime_controller_calibration_source", None)
    updated.pop("runtime_controller_calibration_version", None)
    updated.pop("runtime_controller_calibration_missing", None)
    updated.pop("runtime_controller_calibration_missing_path", None)
    updated.pop("runtime_controller_use_joint_threshold_calibration", None)
    return updated


def _coerce_round_float_map(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, float] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        try:
            normalized[str(int(key_text))] = float(item)
        except (TypeError, ValueError):
            continue
    return normalized


def _apply_joint_runtime_calibration_from_model_dir(
    metadata: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    calibration_path = model_dir / "joint_controller_calibration.json"
    if not calibration_path.exists():
        return metadata
    calibration = load_joint_controller_calibration(calibration_path)
    calibrated = apply_joint_controller_calibration(metadata, calibration)
    calibrated["runtime_controller_calibration_path"] = str(calibration_path.resolve())
    return calibrated


def _apply_joint_runtime_calibration(
    metadata: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    if not bool(metadata.get("runtime_controller_use_joint_threshold_calibration", False)):
        calibrated = dict(metadata)
        calibrated.pop("runtime_controller_calibration_path", None)
        calibrated.pop("runtime_controller_calibration_source", None)
        calibrated.pop("runtime_controller_calibration_version", None)
        calibrated.pop("runtime_controller_calibration_missing", None)
        calibrated.pop("runtime_controller_calibration_missing_path", None)
        calibrated.pop("runtime_controller_disable_calibration", None)
        return calibrated
    if bool(metadata.get("runtime_controller_disable_calibration", False)):
        return reset_two_head_runtime_controller_defaults(metadata)
    explicit_path = str(metadata.get("runtime_controller_calibration_path", "")).strip()
    if explicit_path:
        calibration_path = Path(explicit_path)
        if not calibration_path.exists():
            calibrated = dict(metadata)
            calibrated.pop("runtime_controller_calibration_path", None)
            calibrated["runtime_controller_calibration_missing"] = True
            calibrated["runtime_controller_calibration_missing_path"] = str(
                calibration_path.resolve()
            )
            return calibrated
        calibration = load_joint_controller_calibration(calibration_path)
        calibrated = apply_joint_controller_calibration(metadata, calibration)
        calibrated["runtime_controller_calibration_path"] = str(calibration_path.resolve())
        return calibrated
    return _apply_joint_runtime_calibration_from_model_dir(metadata, model_dir)


BASELINE_ALIASES: dict[str, str] = {
    "ours-delayed-consensus": "ours-eig",
}


def canonical_baseline_name(name: str) -> str:
    cleaned = str(name).strip()
    return BASELINE_ALIASES.get(cleaned, cleaned)


def is_eig_baseline_name(name: str) -> bool:
    return canonical_baseline_name(name) == "ours-eig"


BASELINE_SPECS: dict[str, BaselineSpec] = {
    "ours-eig": BaselineSpec(
        name="ours-eig",
        display_name="Ours (EIG)",
        strategy="evolving_graph",
        description="Evolving Idea Graph multi-agent collaboration with maturity-based commitment.",
        prompt_style="ours",
    ),
    "ours-eig-critic-text": BaselineSpec(
        name="ours-eig-critic-text",
        display_name="Ours (EIG + Text Critic)",
        strategy="evolving_graph",
        description="Evolving Idea Graph multi-agent collaboration with the G4.8 adapted text critic as a conservative edit reranker.",
        prompt_style="ours",
        runtime_controller="text_critic_rerank",
    ),
    "ours-eig-critic-graph": BaselineSpec(
        name="ours-eig-critic-graph",
        display_name="Ours (EIG + Relation-Graph Critic)",
        strategy="evolving_graph",
        description="Evolving Idea Graph multi-agent collaboration with a relation-graph runtime critic as a conservative edit reranker.",
        prompt_style="ours",
        runtime_controller="relation_graph_critic_rerank",
    ),
    "ours-eig-critic-graph-twohead": BaselineSpec(
        name="ours-eig-critic-graph-twohead",
        display_name="Ours (EIG + Two-Head Graph Critic)",
        strategy="evolving_graph",
        description="Parallel EIG with a shared-encoder two-head graph critic for edit selection and post-round commit control.",
        prompt_style="ours",
        runtime_controller="relation_graph_two_head_critic",
    ),
    "ours-eig-critic-calibrated": BaselineSpec(
        name="ours-eig-critic-calibrated",
        display_name="Ours (EIG + Calibrated Two-Head Graph Critic)",
        strategy="evolving_graph",
        description="Parallel EIG with a shared-encoder two-head graph critic and graph-health action-score calibration.",
        prompt_style="ours",
        runtime_controller="relation_graph_two_head_critic",
    ),
    "ours-eig-critic-no-commit": BaselineSpec(
        name="ours-eig-critic-no-commit",
        display_name="Ours (EIG + Two-Head Graph Critic, No Commit)",
        strategy="evolving_graph",
        description="Parallel EIG with learned edit selection but fixed-horizon stopping.",
        prompt_style="ours",
        runtime_controller="relation_graph_two_head_critic",
    ),
    "ours-eig-critic-no-edit": BaselineSpec(
        name="ours-eig-critic-no-edit",
        display_name="Ours (EIG + Two-Head Graph Critic, No Edit)",
        strategy="evolving_graph",
        description="Parallel EIG with heuristic role-local edits and learned post-round commit control.",
        prompt_style="ours",
        runtime_controller="relation_graph_two_head_critic",
    ),
    "ours-eig-fixed-control": BaselineSpec(
        name="ours-eig-fixed-control",
        display_name="Ours (Fixed Control)",
        strategy="evolving_graph",
        description="Parallel EIG with a frozen role-and-round edit policy and fixed five-round horizon.",
        prompt_style="ours",
        runtime_controller="fixed_control",
    ),
    "ours-eig-random-control": BaselineSpec(
        name="ours-eig-random-control",
        display_name="Ours (Random Control)",
        strategy="evolving_graph",
        description="Parallel EIG with seeded random action selection from each legal role-local slate and fixed five-round horizon.",
        prompt_style="ours",
        runtime_controller="random_control",
    ),
    "ours-eig-signal-heuristic": BaselineSpec(
        name="ours-eig-signal-heuristic",
        display_name="Ours (Signal Heuristic Control)",
        strategy="evolving_graph",
        description="Parallel EIG with the streamlined graph-signal controller and heuristic edit/commit decisions.",
        prompt_style="ours",
        runtime_controller="signal_heuristic_control",
    ),
    "direct": BaselineSpec(
        name="direct",
        display_name="Direct",
        strategy="direct",
        description="One-pass single-agent structured idea generation.",
        prompt_style="direct",
    ),
    "self-refine": BaselineSpec(
        name="self-refine",
        display_name="Self Refine",
        strategy="self_refine",
        description="Single-agent draft, critique, and revision baseline.",
        prompt_style="self_refine",
    ),
    "graph-of-thought": BaselineSpec(
        name="graph-of-thought",
        display_name="Graph-of-Thought",
        strategy="graph_of_thought",
        description="Single-model Graph-of-Thought baseline with three graph-generation/scoring rounds followed by final synthesis.",
        prompt_style="graph_of_thought",
        candidate_count=4,
    ),
    "ai-researcher": BaselineSpec(
        name="ai-researcher",
        display_name="AI-Researcher",
        strategy="external",
        description="External wrapper that runs the official AI-Researcher ideation pipeline from its upstream repository.",
    ),
    "scipip": BaselineSpec(
        name="scipip",
        display_name="SciPIP",
        strategy="external",
        description="External wrapper that runs the official SciPIP pipeline from its upstream repository.",
    ),
    "virsci": BaselineSpec(
        name="virsci",
        display_name="VirSci",
        strategy="external",
        description="External wrapper that runs the official Virtual-Scientists pipeline from its upstream repository when the task setting is supported.",
    ),
    "ai-researcher-guided": BaselineSpec(
        name="ai-researcher-guided",
        display_name="AI-Researcher Guided",
        strategy="candidate_rank",
        description="Local AI-Researcher-style workflow with literature-grounded candidate generation and selection.",
        is_local_variant=True,
        reference_target="AI-Researcher",
        prompt_style="ai_researcher_guided",
        candidate_count=4,
    ),
    "scipip-structured": BaselineSpec(
        name="scipip-structured",
        display_name="SciPIP Structured",
        strategy="self_refine",
        description="Local SciPIP-style workflow emphasizing structured motivation and experiment decomposition.",
        is_local_variant=True,
        reference_target="SciPIP",
        prompt_style="scipip_structured",
    ),
    "virsci-discussion": BaselineSpec(
        name="virsci-discussion",
        display_name="VirSci Discussion",
        strategy="evolving_graph",
        description="Local VirSci-style discussion workflow for multi-agent ideation.",
        is_local_variant=True,
        reference_target="VirSci",
        prompt_style="virsci_discussion",
    ),
}


PROMPT_STYLE_GUIDANCE = {
    "ours": (
        "Preserve typed intermediate claims, disagreement tracking, maturity-based commitment, and section-level rigor."
    ),
    "direct": (
        "Produce one concise, strong idea directly from the provided packet without extra self-critique."
    ),
    "self_refine": (
        "Produce a strong first draft, then use explicit critique to revise weak sections."
    ),
    "graph_of_thought": (
        "Represent intermediate reasoning as a graph of candidate thoughts, score/prune the graph, then synthesize the best connected path into one proposal."
    ),
    "ai_researcher_guided": (
        "Emphasize literature-grounded candidate generation, proposal elaboration, diversity across ideas, and selective ranking."
    ),
    "scipip_structured": (
        "Emphasize structured decomposition from topic and inspiration context into motivation and experiment plan."
    ),
    "virsci_discussion": (
        "Emphasize diverse agent perspectives, discussion-style synthesis, and explicit tradeoffs across alternatives."
    ),
}


def get_baseline_spec(name: str) -> BaselineSpec:
    name = canonical_baseline_name(name)
    try:
        return BASELINE_SPECS[name]
    except KeyError as exc:
        options = ", ".join(sorted(BASELINE_SPECS))
        raise KeyError(f"Unknown baseline '{name}'. Available baselines: {options}") from exc


def baseline_choices(*, include_aliases: bool = True) -> list[str]:
    names = list(BASELINE_SPECS)
    if include_aliases:
        names.extend(BASELINE_ALIASES)
    return sorted(set(names))


def attach_baseline_metadata(
    instance,
    *,
    baseline_name: str,
    io_mode: str = "auto",
):
    requested_baseline_name = str(baseline_name).strip()
    baseline = get_baseline_spec(requested_baseline_name)
    instance = apply_io_mode(instance, io_mode=io_mode)
    metadata = dict(instance.metadata)
    if requested_baseline_name != baseline.name:
        metadata["baseline_requested_name"] = requested_baseline_name
    metadata["baseline_name"] = baseline.name
    metadata["baseline_display_name"] = baseline.display_name
    metadata["baseline_strategy"] = baseline.strategy
    metadata["baseline_prompt_style"] = baseline.prompt_style
    metadata["baseline_description"] = baseline.description
    metadata["baseline_local_variant"] = baseline.is_local_variant
    metadata["baseline_reference_target"] = baseline.reference_target
    metadata["baseline_runtime_controller"] = baseline.runtime_controller
    if baseline.strategy == "evolving_graph":
        metadata["runtime_protocol"] = "parallel_graph_v2"
    for key in RUNTIME_CONTROLLER_METADATA_KEYS:
        metadata.pop(key, None)

    if baseline.runtime_controller == "text_critic_rerank":
        metadata["runtime_controller_enabled"] = True
        metadata["runtime_controller_kind"] = "text_critic_rerank"
        metadata["runtime_controller_use_edit"] = True
        metadata["runtime_controller_use_commit"] = False
        metadata["runtime_controller_tau_override"] = 0.05
        metadata["runtime_controller_model_path"] = str(_default_text_critic_model_path())
    elif baseline.runtime_controller == "relation_graph_critic_rerank":
        metadata["runtime_controller_enabled"] = True
        metadata["runtime_controller_kind"] = "relation_graph_critic_rerank"
        metadata["runtime_controller_use_edit"] = True
        metadata["runtime_controller_use_commit"] = False
        metadata["runtime_controller_tau_override"] = 0.05
        metadata["runtime_controller_model_dir"] = str(_default_relation_graph_runtime_model_dir())
    elif baseline.runtime_controller == "relation_graph_two_head_critic":
        metadata["runtime_controller_enabled"] = True
        metadata["runtime_controller_kind"] = "relation_graph_two_head_critic"
        metadata["runtime_controller_use_commit"] = True
        metadata.update(TWO_HEAD_RUNTIME_CONTROLLER_DEFAULTS)
        metadata["runtime_controller_model_dir"] = str(_default_relation_graph_two_head_runtime_model_dir())
        if baseline.name == "ours-eig-critic-no-commit":
            metadata["runtime_controller_use_commit"] = False
        if baseline.name == "ours-eig-critic-no-edit":
            metadata["runtime_controller_use_edit"] = False
            metadata["runtime_controller_use_action_score_calibration"] = False
        metadata = _apply_joint_runtime_calibration(
            metadata,
            Path(str(metadata["runtime_controller_model_dir"])),
        )
    elif baseline.runtime_controller == "fixed_control":
        metadata["runtime_controller_enabled"] = True
        metadata["runtime_controller_kind"] = "fixed_control"
        metadata["runtime_controller_use_edit"] = True
        metadata["runtime_controller_use_commit"] = False
        metadata["runtime_controller_policy_path"] = str(_default_fixed_control_policy_path())
        metadata["max_rounds_hint"] = 5
    elif baseline.runtime_controller == "random_control":
        metadata["runtime_controller_enabled"] = True
        metadata["runtime_controller_kind"] = "random_control"
        metadata["runtime_controller_use_edit"] = True
        metadata["runtime_controller_use_commit"] = False
        metadata["runtime_controller_random_seed"] = 0
        metadata["max_rounds_hint"] = 5
    elif baseline.runtime_controller == "signal_heuristic_control":
        metadata["runtime_controller_enabled"] = True
        metadata["runtime_controller_kind"] = "signal_heuristic_control"
        metadata["runtime_controller_use_edit"] = True
        metadata["runtime_controller_use_commit"] = True
        metadata["runtime_controller_tau_override"] = 0.0
        metadata["runtime_controller_gamma_commit"] = 0.58
        metadata["runtime_controller_gamma_commit_by_round"] = {
            1: 0.95,
            2: 0.74,
            3: 0.62,
            4: 0.58,
            5: 0.55,
            6: 0.53,
        }
        metadata["runtime_controller_min_commit_round"] = 2
        metadata["max_rounds_hint"] = 6
    return instance.__class__(
        name=instance.name,
        topic=instance.topic,
        literature=list(instance.literature),
        source_path=instance.source_path,
        metadata=metadata,
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object.")
    return payload


def _coerce_string(value: Any) -> str:
    return _clean_text(value)


def _first_sentence(text: Any) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for separator in (". ", "! ", "? "):
        if separator in cleaned:
            return cleaned.split(separator, 1)[0].strip().rstrip(".!?") + "."
    return cleaned.rstrip(".!?") + "."


def _proposal_from_payload(payload: dict[str, Any]) -> FinalProposal:
    return FinalProposal(
        title=_coerce_string(payload.get("title")),
        abstract="",
        problem=_coerce_string(payload.get("problem")),
        existing_methods=_coerce_string(payload.get("existing_methods") or payload.get("existing_method")),
        motivation=_coerce_string(payload.get("motivation")),
        hypothesis=_coerce_string(payload.get("hypothesis") or payload.get("core_idea")),
        method=_coerce_string(payload.get("method") or payload.get("method_sketch")),
        evaluation=_coerce_string(payload.get("evaluation") or payload.get("experiment_plan")),
        significance=_coerce_string(payload.get("significance") or payload.get("expected_contribution")),
        caveats=_coerce_string(payload.get("caveats") or payload.get("risk") or payload.get("risks")),
    )


def _proposal_as_prompt_payload(proposal: FinalProposal) -> dict[str, str]:
    return {
        "title": proposal.title,
        "problem": proposal.problem,
        "existing_methods": proposal.existing_methods,
        "motivation": proposal.motivation,
        "hypothesis": proposal.hypothesis,
        "method": proposal.method,
        "evaluation": proposal.evaluation,
        "significance": proposal.significance,
        "caveats": proposal.caveats,
    }


def _topic_text(graph: IdeaGraph) -> str:
    cleaned = _clean_text(graph.topic).rstrip(".")
    prefixes = (
        "The topic of this paper is ",
        "Ideation topic keyword: ",
    )
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned or _clean_text(graph.topic)


def _reference_packet(graph: IdeaGraph) -> list[dict[str, str]]:
    packet = graph.metadata.get("benchmark_input_packet", {})
    if not isinstance(packet, dict):
        return []
    references = packet.get("reference_packet", [])
    if not isinstance(references, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in references:
        if not isinstance(item, dict):
            continue
        title = _coerce_string(item.get("title"))
        snippet = _coerce_string(item.get("snippet"))
        if title or snippet:
            normalized.append({"title": title, "snippet": snippet})
    return normalized


def _generation_metadata(graph: IdeaGraph) -> dict[str, Any]:
    payload = graph.metadata.get("generation_safe_metadata", graph.metadata)
    return payload if isinstance(payload, dict) else graph.metadata


def _baseline_anchor_terms(graph: IdeaGraph) -> list[str]:
    topic = _topic_text(graph)
    corpus = " ".join(
        [
            topic,
            *[
                f"{item.get('title', '')} {item.get('snippet', '')}"
                for item in _reference_packet(graph)
            ],
        ]
    ).lower()
    anchors = [topic] if topic else []
    candidate_phrases = [
        "language field",
        "radiance field",
        "gaussian splatting",
        "open-vocabulary",
        "clip embedding",
        "hierarchical semantics",
        "segmentation",
        "query",
        "localization",
        "panoramic",
        "relative pose",
        "geometric",
        "alignment",
        "uncertainty",
        "compression",
        "retrieval",
    ]
    for phrase in candidate_phrases:
        if phrase in corpus and phrase not in anchors:
            anchors.append(phrase)
    return anchors[:8]


def _baseline_focus_constraints(graph: IdeaGraph, baseline: BaselineSpec) -> list[str]:
    topic = _topic_text(graph)
    anchors = _baseline_anchor_terms(graph)
    anchor_hint = ", ".join(anchors[1:6]) if len(anchors) > 1 else topic
    constraints = [
        f"Keep the main task tightly centered on '{topic}', not an adjacent task family.",
        "Use only the benchmark topic and visible reference packet; do not rely on hidden target-paper fields.",
        "Do not treat cited method papers as datasets. Use method papers as baselines or inspiration; mention datasets only when the packet clearly signals benchmark datasets or evaluation assets.",
        "Avoid generic high-level proposals that could fit many topics; tie the mechanism and evaluation to the given benchmark packet.",
        f"Use packet anchors when helpful: {anchor_hint}.",
    ]
    if baseline.prompt_style == "direct":
        constraints.extend(
            [
                "Choose one strong central mechanism instead of listing several possible directions.",
                "Prefer the simplest benchmark-faithful idea that is still non-trivial and testable.",
            ]
        )
    elif baseline.prompt_style == "self_refine":
        constraints.extend(
            [
                "Use critique to remove generic wording, unsupported claims, and vague evaluation language.",
                "The final revision should improve specificity rather than simply making the text longer.",
            ]
        )
    elif baseline.prompt_style == "graph_of_thought":
        constraints.extend(
            [
                "Build a compact thought graph whose nodes cover problem gap, mechanism, evaluation, risk, and expected impact.",
                "Score connected paths by benchmark fit, novelty, feasibility, and evaluation clarity before synthesis.",
                "Synthesize one final idea from the strongest connected path rather than averaging unrelated thoughts.",
            ]
        )
    elif baseline.prompt_style == "scipip_structured":
        constraints.extend(
            [
                "Emphasize structured decomposition: identify the bottleneck, explain why current methods fail, then propose one coherent pipeline.",
                "Make the experiment plan concrete with datasets, metrics, baselines, and ablations whenever the packet supports them.",
            ]
        )
    elif baseline.prompt_style == "ai_researcher_guided":
        constraints.extend(_ai_researcher_focus_constraints(graph))
    elif baseline.prompt_style == "virsci_discussion":
        constraints.extend(
            [
                "Preserve multiple viewpoints and tradeoffs before settling on one final idea.",
                "Explicitly surface why the chosen idea wins over nearby alternatives.",
            ]
        )
    return _unique_strings(constraints)


def _grounding_brief(graph: IdeaGraph) -> dict[str, str]:
    grounding = build_literature_grounding(
        literature=graph.literature,
        metadata=_generation_metadata(graph),
    )
    return {
        "existing_methods_summary": grounding.existing_methods_summary,
        "experiment_plan_summary": grounding.experiment_plan_summary,
    }


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = _coerce_string(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def _ai_researcher_anchor_terms(graph: IdeaGraph) -> list[str]:
    return _baseline_anchor_terms(graph)[:6]


def _ai_researcher_context_corpus(graph: IdeaGraph) -> str:
    parts: list[str] = [_topic_text(graph)]
    keyword = _coerce_string(
        _generation_metadata(graph).get("keyword")
        or graph.metadata.get("keyword")
    )
    if keyword:
        parts.append(keyword)
    parts.extend(
        [
            f"{item.get('title', '')} {item.get('snippet', '')}"
            for item in _reference_packet(graph)
        ]
    )
    parts.extend(_reference_support_texts(graph))
    return " ".join(parts).lower()


def _is_ai_researcher_language_field_context(graph: IdeaGraph) -> bool:
    corpus = _ai_researcher_context_corpus(graph)
    return any(
        term in corpus
        for term in (
            "language field",
            "radiance field",
            "language embedded radiance",
            "lerf",
            "open-vocabulary scene understanding",
        )
    )


def _ai_researcher_focus_constraints(graph: IdeaGraph) -> list[str]:
    topic = _topic_text(graph)
    anchors = _ai_researcher_anchor_terms(graph)
    anchor_hint = ", ".join(anchors[1:]) if len(anchors) > 1 else topic
    if _is_ai_researcher_language_field_context(graph):
        contribution_hint = (
            "Prefer contributions about language or radiance field representation, "
            "querying, grounding, efficiency, or semantic structure."
        )
        drift_hint = (
            "Penalize drift into generic text-to-3D generation, generic 3D reconstruction, "
            "generic scene synthesis, or temporal/video extensions unless the packet directly "
            "supports that shift."
        )
    else:
        contribution_hint = (
            "Prefer one concrete technical idea whose mechanism and evaluation stay close to the "
            "benchmark topic and visible references."
        )
        drift_hint = (
            "Penalize drift into adjacent task families, unsupported modality shifts, or broad "
            "generic proposals that are not justified by the packet."
        )
    return [
        f"Keep the main task tightly centered on '{topic}', not an adjacent task family.",
        contribution_hint,
        drift_hint,
        f"Use the packet anchors when helpful: {anchor_hint}.",
        "Do not treat cited method papers as datasets. Use method papers as baselines or inspiration; mention datasets only when the packet clearly indicates a benchmark dataset.",
        "Favor ideas whose evaluation can be justified directly from the benchmark topic and visible references.",
    ]


def _ai_researcher_topic_fidelity_score(graph: IdeaGraph, proposal: FinalProposal) -> float:
    title_text = proposal.title.lower()
    problem_text = proposal.problem.lower()
    hypothesis_text = proposal.hypothesis.lower()
    method_text = proposal.method.lower()
    text = " ".join(
        [
            proposal.title,
            proposal.problem,
            proposal.existing_methods,
            proposal.motivation,
            proposal.hypothesis,
            proposal.method,
            proposal.evaluation,
            proposal.significance,
        ]
    ).lower()
    anchors = _ai_researcher_anchor_terms(graph)
    if not anchors:
        return 0.0

    anchor_hits = sum(1 for term in anchors if term and term.lower() in text)
    score = 0.45 * (anchor_hits / len(anchors))

    topic = _topic_text(graph).lower()
    context_corpus = _ai_researcher_context_corpus(graph)
    title_problem_method = " ".join([proposal.title, proposal.problem, proposal.method]).lower()
    evaluation_text = proposal.evaluation.lower()
    if topic and topic in text:
        score += 0.25
    keyword = _coerce_string(
        _generation_metadata(graph).get("keyword")
        or graph.metadata.get("keyword")
    ).lower()
    if keyword and keyword in text:
        score += 0.25

    topical_terms = [topic, *anchors[1:]]
    topical_hits = sum(1 for term in topical_terms if term and term.lower() in title_problem_method)
    score += min(0.18, 0.06 * topical_hits)

    if any(
        signal in evaluation_text
        for signal in ("benchmark", "dataset", "metric", "accuracy", "calibration", "ablation", "precision", "recall")
    ):
        score += 0.06

    unsupported_drift_terms = {
        "language field": ("language field", "3d language field"),
        "radiance field": ("radiance field", "nerf"),
        "gaussian splatting": ("gaussian splatting",),
        "text-to-3d": ("text-to-3d",),
        "scene generation": ("scene generation", "scene synthesis"),
        "3d reconstruction": ("3d reconstruction",),
        "temporal": ("temporal", "video-based"),
    }
    for context_term, proposal_terms in unsupported_drift_terms.items():
        if context_term not in context_corpus and any(term in text for term in proposal_terms):
            score -= 0.12

    if _is_ai_researcher_language_field_context(graph):
        if "3d language field" in title_text:
            score += 0.25
        elif "language field" in title_text:
            score += 0.18
        else:
            score -= 0.18
        if "3d language field" in problem_text or "language field modeling" in problem_text:
            score += 0.18
        if "language field" in hypothesis_text or "language field" in method_text:
            score += 0.12
        if "modeling" in title_text or "modeling" in problem_text or "modeling" in method_text:
            score += 0.08

        if any(term in text for term in ("language field", "radiance field", "language-embedded radiance")):
            score += 0.25
        else:
            score -= 0.35

        for positive_term in ("open-vocabulary", "gaussian splatting", "localization"):
            if positive_term in text:
                score += 0.08

        for drift_term in ("text-to-3d", "scene generation", "3d reconstruction", "temporal", "video-based"):
            if drift_term in text:
                score -= 0.12
    else:
        topic_tokens = [
            token
            for token in re.findall(r"[a-z0-9][a-z0-9-]+", topic)
            if len(token) >= 5
        ]
        distinct_hits = sum(1 for token in topic_tokens if token in title_problem_method)
        score += min(0.16, 0.08 * distinct_hits)
        if keyword and keyword not in title_problem_method and keyword not in evaluation_text:
            score -= 0.10

    return max(0.0, min(1.0, score))


def _reference_support_texts(graph: IdeaGraph) -> list[str]:
    texts: list[str] = []
    raw_grounding = graph.metadata.get("paper_grounding", {})
    if isinstance(raw_grounding, dict):
        raw_references = raw_grounding.get("reference_paper_snippets", [])
        if isinstance(raw_references, list):
            for item in raw_references:
                if not isinstance(item, dict):
                    continue
                for field_name in ("snippet", "abstract", "introduction", "method", "evaluation", "text_excerpt"):
                    value = _coerce_string(item.get(field_name))
                    if value:
                        texts.append(value)
    for item in _reference_packet(graph):
        for field_name in ("title", "snippet"):
            value = _coerce_string(item.get(field_name))
            if value:
                texts.append(value)
    return texts


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _ai_researcher_guided_postprocess_proposal(graph: IdeaGraph, proposal: FinalProposal) -> FinalProposal:
    references = _reference_packet(graph)
    reference_titles = [item.get("title", "") for item in references if item.get("title")]
    support_text = " ".join(_reference_support_texts(graph))

    title = proposal.title
    if "language field" not in title.lower():
        title = f"{title.rstrip()} for 3D Language Field Modeling".strip()
    if "3d language field" not in title.lower() and "language field" in title.lower():
        title = title.replace("Language Field", "3D Language Field", 1)
    if len(title.split()) > 18:
        title = "Efficient Open-Vocabulary Querying in 3D Language Fields"

    problem = proposal.problem
    if "3d language field" not in problem.lower():
        problem = problem.rstrip(".") + " In particular, scalable 3D language field modeling remains difficult when queries must stay open-vocabulary and mask-free."

    existing_methods = proposal.existing_methods
    if "lerf" not in existing_methods.lower() and any("lerf" in title_item.lower() for title_item in reference_titles):
        existing_methods = existing_methods.rstrip(".") + " LERF demonstrates language-embedded radiance fields for open-vocabulary querying."
    if "gaussian splatting" not in existing_methods.lower() and any("gaussian splatting" in title_item.lower() for title_item in reference_titles):
        existing_methods = existing_methods.rstrip(".") + " 3D Gaussian Splatting offers efficient rendering but does not by itself solve language-field querying."

    motivation = proposal.motivation
    if "3d language field modeling" not in motivation.lower():
        motivation = motivation.rstrip(".") + " This matters because 3D language field modeling should support efficient open-vocabulary interaction in real scenes."

    hypothesis = proposal.hypothesis
    if "language field" not in hypothesis.lower():
        hypothesis = hypothesis.rstrip(".") + " within a 3D language field."

    method = proposal.method
    if "3d language field" not in method.lower():
        method = method.rstrip(".") + " The resulting representation is optimized specifically as a 3D language field rather than a generic text-guided scene generator."

    evaluation = proposal.evaluation
    if "lerf dataset" not in evaluation.lower() and "lerf dataset" in support_text.lower():
        evaluation = evaluation.rstrip(".") + " Use the LERF dataset for open-vocabulary querying analysis."
    if "3d-ovs" not in evaluation.lower() and "3d-ovs" in support_text.lower():
        evaluation = evaluation.rstrip(".") + " Also test on the 3D-OVS dataset for open-vocabulary semantic behavior."
    if "localization accuracy" not in evaluation.lower() and "localization accuracy" in support_text.lower():
        evaluation = evaluation.rstrip(".") + " Report localization accuracy in addition to query precision and recall."

    significance = proposal.significance
    if "3d language field modeling" not in significance.lower():
        significance = significance.rstrip(".") + " This would strengthen efficient 3D language field modeling for open-vocabulary interaction."

    return FinalProposal(
        title=title,
        abstract="",
        problem=problem,
        existing_methods=existing_methods,
        motivation=motivation,
        hypothesis=hypothesis,
        method=method,
        evaluation=evaluation,
        significance=significance,
        caveats=proposal.caveats,
    )


def _split_sentences(text: Any) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]


def _is_noisy_sentence(text: str) -> bool:
    lowered = _clean_text(text).casefold()
    if not lowered:
        return True
    noisy_markers = (
        "experiments were conducted on paper introduces",
        "paper introduces",
        "captured in diverse real scenarios",
        "project website",
        "see the project website",
        "figure",
        "table",
        "et al",
    )
    return any(marker in lowered for marker in noisy_markers)


def _clean_section_text(text: Any) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return _clean_text(text)
    kept: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = _clean_text(sentence).casefold()
        if not normalized or normalized in seen:
            continue
        if _is_noisy_sentence(sentence):
            continue
        kept.append(sentence)
        seen.add(normalized)
    return " ".join(kept) if kept else _clean_text(text)


def _baseline_postprocess_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    proposal: FinalProposal,
) -> FinalProposal:
    if baseline.prompt_style == "ai_researcher_guided" and _is_ai_researcher_language_field_context(graph):
        return _ai_researcher_guided_postprocess_proposal(graph, proposal)

    topic = _topic_text(graph)
    anchors = _baseline_anchor_terms(graph)
    anchor_phrase = anchors[1] if len(anchors) > 1 else topic
    grounding = _grounding_brief(graph)
    benchmark_packet = graph.metadata.get("benchmark_input_packet", {})
    reference_packet = benchmark_packet.get("reference_packet", []) if isinstance(benchmark_packet, dict) else []
    keyword = _coerce_string(
        graph.metadata.get("keyword")
        or (benchmark_packet.get("keyword") if isinstance(benchmark_packet, dict) else "")
        or topic
    )
    keyword_only_mode = (
        _coerce_string(graph.metadata.get("benchmark")) == "liveideabench"
        or (not reference_packet and "benchmark keyword:" in grounding["existing_methods_summary"].lower())
    )

    title = proposal.title or topic.title()
    if baseline.prompt_style == "scipip_structured" and topic and topic.lower() not in title.lower():
        title = f"{title.rstrip()} for {topic}".strip()
    if len(title.split()) > 18:
        title = title[:96].rsplit(" ", 1)[0].strip()

    problem = proposal.problem or f"{topic} remains insufficiently solved under the current benchmark setting."
    if topic and topic.lower() not in problem.lower():
        problem = problem.rstrip(".") + f" This matters directly for {topic}."

    existing_methods = proposal.existing_methods or grounding["existing_methods_summary"]
    if baseline.prompt_style == "scipip_structured" and grounding["existing_methods_summary"]:
        summary = grounding["existing_methods_summary"]
        should_append_summary = bool(summary) and summary not in existing_methods
        if should_append_summary and len(existing_methods.split()) >= 38:
            should_append_summary = False
        if should_append_summary:
            existing_methods = existing_methods.rstrip(".") + " " + summary
    if keyword_only_mode and (
        "benchmark keyword:" in existing_methods.lower()
        or "held-out metadata" in existing_methods.lower()
        or "row provides a keyword prompt" in existing_methods.lower()
    ):
        existing_methods = (
            f"For {keyword}, common directions include spatiotemporal forecasting models, "
            "physics-aware simulation, and multi-source data fusion."
        )

    motivation = proposal.motivation or f"A more precise and testable idea for {topic} is needed."
    if baseline.prompt_style == "scipip_structured" and "why" not in motivation.lower():
        motivation = motivation.rstrip(".") + " The key motivation is that current methods do not adequately address the core bottleneck exposed by the benchmark packet."

    hypothesis = proposal.hypothesis or f"A more explicit mechanism around {anchor_phrase} can improve results for {topic}."
    method = proposal.method
    if baseline.prompt_style == "scipip_structured" and method:
        if "first" not in method.lower():
            method = (
                "First, identify the core bottleneck from the benchmark packet. "
                + method.rstrip(".")
                + ". Then connect that bottleneck to one coherent pipeline and targeted ablations."
            )
    if not method:
        method = f"Design one concrete mechanism around {anchor_phrase} for {topic}, with explicit implementation steps and ablations."

    evaluation = proposal.evaluation or grounding["experiment_plan_summary"]
    experiment_summary = grounding["experiment_plan_summary"]
    should_append_experiment_summary = bool(experiment_summary) and experiment_summary not in evaluation
    if should_append_experiment_summary and baseline.prompt_style in {"self_refine", "scipip_structured"}:
        if (
            experiment_summary.casefold().startswith("compare against strong baselines")
            and _contains_any(evaluation, ["compare against", "ablation", "quantitative metric"])
        ):
            should_append_experiment_summary = False
        if baseline.prompt_style == "scipip_structured" and len(evaluation.split()) >= 28:
            should_append_experiment_summary = False
    if should_append_experiment_summary:
        if baseline.prompt_style in {"self_refine", "scipip_structured"}:
            evaluation = evaluation.rstrip(".") + " " + grounding["experiment_plan_summary"]
    if "metric" not in evaluation.lower() and "accuracy" not in evaluation.lower():
        evaluation = evaluation.rstrip(".") + " Report quantitative metrics and compare against strong baselines."
    if keyword_only_mode and any(
        marker in evaluation.lower()
        for marker in ("synthetic urban dataset", "lerf dataset", "3d-ovs", "polycam", "scannet")
    ):
        evaluation = (
            f"Evaluate on realistic benchmark tasks for {keyword}, compare against strong data-driven and hybrid baselines, "
            "report task-specific quantitative metrics, and include ablations over the main components."
        )

    significance = proposal.significance or f"If successful, the idea would make {topic} more accurate, robust, or testable."
    if topic and topic.lower() not in significance.lower():
        significance = significance.rstrip(".") + f" This could make progress on {topic} more reliable and testable."

    caveats = proposal.caveats or "The idea may still depend on incomplete literature context and should be validated with targeted ablations."

    existing_methods = _clean_section_text(existing_methods)
    method = _clean_section_text(method)
    evaluation = _clean_section_text(evaluation)
    significance = _clean_section_text(significance)

    return FinalProposal(
        title=title,
        abstract="",
        problem=problem,
        existing_methods=existing_methods,
        motivation=motivation,
        hypothesis=hypothesis,
        method=method,
        evaluation=evaluation,
        significance=significance,
        caveats=caveats,
    )


def _deterministic_direct_proposal(graph: IdeaGraph, baseline: BaselineSpec) -> FinalProposal:
    generation_metadata = _generation_metadata(graph)
    grounding = build_literature_grounding(literature=graph.literature, metadata=generation_metadata)
    topic_text = _topic_text(graph)
    benchmark_packet = graph.metadata.get("benchmark_input_packet", {})
    task_instruction = _coerce_string(
        benchmark_packet.get("task_instruction") if isinstance(benchmark_packet, dict) else ""
    )
    references = []
    if isinstance(benchmark_packet, dict):
        references = benchmark_packet.get("reference_packet", [])
    reference_titles = []
    if isinstance(references, list):
        for item in references:
            if isinstance(item, dict):
                title = _coerce_string(item.get("title"))
                if title:
                    reference_titles.append(title)

    existing_methods = grounding.existing_methods_summary
    if not existing_methods and reference_titles:
        existing_methods = (
            "Relevant nearby work includes "
            + ", ".join(reference_titles[:3])
            + ", but the current benchmark packet does not expose their full target-paper labels."
        )
    if not existing_methods:
        existing_methods = (
            f"Existing methods for {topic_text} remain only partially grounded in the current context packet."
        )

    motivation = _coerce_string(generation_metadata.get("motivation"))
    if not motivation:
        motivation = (
            f"The benchmark task asks for a concrete and testable idea for {topic_text}. "
            f"{task_instruction or 'The output should remain grounded in the provided context.'}"
        )

    method = (
        f"Design a concise method for {topic_text} that combines a clear mechanism, explicit evaluation hooks, "
        "and literature-aware comparison points instead of only high-level brainstorming."
    )
    if baseline.prompt_style == "scipip_structured":
        method = (
            f"Decompose {topic_text} into a structured motivation, method sketch, and experiment plan derived "
            "from the benchmark topic and reference packet."
        )

    evaluation = grounding.experiment_plan_summary or (
        f"Evaluate the idea for {topic_text} with strong baselines, benchmark-relevant datasets or tasks, "
        "and targeted ablations that isolate the main proposed mechanism."
    )

    proposal = FinalProposal(
        title=topic_text[:1].upper() + topic_text[1:] if topic_text else baseline.display_name,
        abstract="",
        problem=f"{topic_text} still lacks a compact, testable research formulation in the current benchmark setting.",
        existing_methods=existing_methods,
        motivation=motivation,
        hypothesis=f"A more explicitly structured idea for {topic_text} can improve testability and scientific usefulness.",
        method=method,
        evaluation=evaluation,
        significance=(
            f"If successful, the idea would provide a clearer and more benchmark-faithful research direction for {topic_text}."
        ),
        caveats="The current baseline uses only the benchmark packet and may miss broader external literature context.",
    )
    return _baseline_postprocess_proposal(graph, baseline, proposal)


def _deterministic_refine_proposal(graph: IdeaGraph, draft: FinalProposal, baseline: BaselineSpec) -> FinalProposal:
    grounding = build_literature_grounding(
        literature=graph.literature,
        metadata=_generation_metadata(graph),
    )
    existing_methods = draft.existing_methods
    if grounding.existing_methods_summary and grounding.existing_methods_summary not in existing_methods:
        existing_methods = grounding.existing_methods_summary

    evaluation = draft.evaluation
    if grounding.experiment_plan_summary and grounding.experiment_plan_summary not in evaluation:
        evaluation = grounding.experiment_plan_summary

    caveats = draft.caveats
    if "test" not in caveats.casefold():
        caveats = (
            caveats.rstrip(".")
            + ". Further stress tests should check whether the idea still holds under stronger baselines and ablations."
        )

    significance = draft.significance
    if baseline.prompt_style == "ai_researcher_guided" and "literature" not in significance.casefold():
        significance = significance.rstrip(".") + ". It should also improve literature-grounded ideation quality."

    proposal = FinalProposal(
        title=draft.title,
        abstract="",
        problem=draft.problem,
        existing_methods=existing_methods,
        motivation=draft.motivation,
        hypothesis=draft.hypothesis,
        method=draft.method,
        evaluation=evaluation,
        significance=significance,
        caveats=caveats,
    )
    return _baseline_postprocess_proposal(graph, baseline, proposal)


def _direct_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are a scientific idea generation baseline. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Generate exactly one structured research idea using the provided benchmark packet and output schema. "
        "Do not assume access to hidden target-paper labels. "
        "Benchmark fidelity matters more than writing style. "
        "Each section must add distinct information instead of repeating the same sentence in different fields. "
        "Keep the idea specific, testable, and grounded in the visible references. "
        "Do not copy raw extraction fragments from literature snippets; if a dataset or method fragment looks noisy or truncated, omit it. "
        "Prefer one coherent mechanism and one coherent evaluation story rather than several loosely connected ideas. "
        'JSON schema: {"title":"...","problem":"...","existing_methods":"...","motivation":"...",'
        '"hypothesis":"...","method":"...","evaluation":"...","significance":"...","caveats":"..."}'
    )


def _direct_user_prompt(graph: IdeaGraph, baseline: BaselineSpec) -> str:
    packet = graph.metadata.get("benchmark_input_packet", {})
    payload = {
        "baseline": asdict(baseline),
        "topic": graph.topic,
        "task_instruction": graph.metadata.get("task_instruction", ""),
        "input_packet": packet,
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "literature_grounding": _grounding_brief(graph),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _critique_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are a scientific idea critic. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Critique the current draft using only the benchmark packet and return concise revision guidance. "
        "Focus on benchmark fidelity, literature grounding, unsupported claims, vague evaluation design, repetition across sections, noisy copied snippet fragments, and overcomplicated mechanism drift. "
        'JSON schema: {"strengths":["..."],"weaknesses":["..."],"revision_focus":["..."]}'
    )


def _critique_user_prompt(graph: IdeaGraph, baseline: BaselineSpec, draft: FinalProposal) -> str:
    payload = {
        "baseline": asdict(baseline),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "literature_grounding": _grounding_brief(graph),
        "draft": _proposal_as_prompt_payload(draft),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _refine_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are revising a scientific research idea after critique. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Revise the draft to improve grounding, coherence, benchmark fidelity, and testability while keeping the output concise. "
        "Do not add generic filler; prefer sharper, more benchmark-faithful content. "
        "If the draft contains noisy copied snippet fragments, remove them rather than elaborating on them. "
        "If the draft contains multiple loosely connected mechanisms, simplify to one coherent method story. "
        'JSON schema: {"title":"...","problem":"...","existing_methods":"...","motivation":"...",'
        '"hypothesis":"...","method":"...","evaluation":"...","significance":"...","caveats":"..."}'
    )


def _refine_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    draft: FinalProposal,
    critique_payload: dict[str, Any],
) -> str:
    payload = {
        "baseline": asdict(baseline),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "literature_grounding": _grounding_brief(graph),
        "draft": _proposal_as_prompt_payload(draft),
        "critique": critique_payload,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _candidate_generation_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    candidate_count = max(2, baseline.candidate_count)
    return (
        "You are generating diverse scientific research ideas for a baseline wrapper. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        f"Generate exactly {candidate_count} different structured idea candidates using only the provided benchmark packet. "
        "Make the candidates meaningfully different in mechanism, framing, or evaluation strategy rather than rephrasing the same idea. "
        "Do not assume access to hidden target-paper labels. "
        "All candidates must remain benchmark-faithful and avoid drifting into adjacent tasks. "
        'JSON schema: {"candidates":[{"title":"...","problem":"...","existing_methods":"...","motivation":"...",'
        '"hypothesis":"...","method":"...","evaluation":"...","significance":"...","caveats":"..."}]}'
    )


def _candidate_generation_user_prompt(graph: IdeaGraph, baseline: BaselineSpec) -> str:
    payload = {
        "baseline": asdict(baseline),
        "topic": graph.topic,
        "task_instruction": graph.metadata.get("task_instruction", ""),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "literature_grounding": _grounding_brief(graph),
        "candidate_count": max(2, baseline.candidate_count),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _graph_of_thought_generation_system_prompt(
    baseline: BaselineSpec,
    *,
    round_index: int,
    total_rounds: int,
) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    candidate_count = max(3, baseline.candidate_count)
    round_instruction = (
        "This is the first round, so create the initial graph from scratch."
        if round_index <= 1
        else (
            "This is a later round, so expand or repair the previous graph. "
            "Preserve useful node ids, add only the nodes or edges needed to address the previous scoring feedback, "
            "and return the full updated graph."
        )
    )
    return (
        "You are implementing a Graph-of-Thought scientific ideation baseline. "
        f"{guidance} "
        f"Round {round_index}/{total_rounds}: generate or update a compact directed thought graph, not a final proposal. "
        f"{round_instruction} "
        f"Create {candidate_count} to {candidate_count + 2} thought nodes and enough edges to show which ideas support, test, repair, or contradict each other. "
        "Nodes should cover problem gap, mechanism, evaluation, risk, and impact when possible. "
        "Use only the visible benchmark packet and references. Do not assume hidden target-paper labels. "
        "Return strict JSON only, with no markdown or commentary. "
        'JSON schema: {"thought_nodes":[{"id":"n1","type":"problem_gap|mechanism|evaluation|risk|impact|assumption","text":"..."}],'
        '"thought_edges":[{"source":"n1","relation":"supports|tests|repairs|contradicts|motivates","target":"n2","rationale":"..."}]}'
    )


def _graph_of_thought_generation_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    *,
    round_index: int,
    total_rounds: int,
    previous_graph: dict[str, list[dict[str, str]]] | None = None,
    previous_scoring: dict[str, Any] | None = None,
) -> str:
    payload = {
        "baseline": asdict(baseline),
        "round_index": round_index,
        "total_rounds": total_rounds,
        "topic": graph.topic,
        "task_instruction": graph.metadata.get("task_instruction", ""),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "literature_grounding": _grounding_brief(graph),
    }
    if previous_graph is not None:
        payload["previous_thought_graph"] = previous_graph
    if previous_scoring is not None:
        payload["previous_scoring"] = previous_scoring
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_thought_nodes(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_nodes = payload.get("thought_nodes")
    if raw_nodes is None:
        raw_nodes = payload.get("nodes", [])
    if not isinstance(raw_nodes, list):
        return []

    nodes: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            continue
        node_id = _coerce_string(item.get("id") or item.get("node_id") or f"n{index + 1}")
        node_id = re.sub(r"[^A-Za-z0-9_.:-]+", "_", node_id).strip("_") or f"n{index + 1}"
        if node_id in seen_ids:
            node_id = f"{node_id}_{index + 1}"
        thought_type = _coerce_string(item.get("type") or item.get("kind") or "thought")
        text = _coerce_string(item.get("text") or item.get("content") or item.get("thought"))
        if not text:
            continue
        nodes.append({"id": node_id, "type": thought_type or "thought", "text": text})
        seen_ids.add(node_id)
    return nodes


def _normalize_thought_edges(payload: dict[str, Any], node_ids: set[str]) -> list[dict[str, str]]:
    raw_edges = payload.get("thought_edges")
    if raw_edges is None:
        raw_edges = payload.get("edges", [])
    if not isinstance(raw_edges, list):
        return []

    edges: list[dict[str, str]] = []
    for item in raw_edges:
        if not isinstance(item, dict):
            continue
        source = _coerce_string(item.get("source") or item.get("source_id"))
        target = _coerce_string(item.get("target") or item.get("target_id"))
        if source not in node_ids or target not in node_ids:
            continue
        relation = _coerce_string(item.get("relation") or item.get("type") or "related_to")
        rationale = _coerce_string(item.get("rationale") or item.get("reason"))
        edges.append(
            {
                "source": source,
                "relation": relation or "related_to",
                "target": target,
                "rationale": rationale,
            }
        )
    return edges


def _normalize_thought_graph_payload(payload: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    nodes = _normalize_thought_nodes(payload)
    node_ids = {node["id"] for node in nodes}
    edges = _normalize_thought_edges(payload, node_ids)
    return {"thought_nodes": nodes, "thought_edges": edges}


def _graph_of_thought_scoring_system_prompt(
    baseline: BaselineSpec,
    *,
    round_index: int,
    total_rounds: int,
) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    next_step = (
        "Return feedback that will guide the next graph-generation round."
        if round_index < total_rounds
        else "Return the final selected subgraph/path for proposal synthesis."
    )
    return (
        "You are scoring a Graph-of-Thought ideation graph. "
        f"{guidance} "
        f"Round {round_index}/{total_rounds}: select the strongest connected path or subgraph and critique what the graph still needs. "
        f"{next_step} "
        "Score thoughts and paths by benchmark fit, novelty, feasibility, literature grounding, and evaluation clarity. "
        "Prefer a coherent path with one central mechanism over disconnected high-level thoughts. "
        "Return strict JSON only, with no markdown or commentary. "
        'JSON schema: {"selected_node_ids":["n1"],"critique":"...","repair_instructions":["..."],'
        '"scores":[{"node_id":"n1","novelty":1,"feasibility":1,"benchmark_fit":1,"overall":1}]}'
    )


def _graph_of_thought_scoring_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    thought_graph: dict[str, list[dict[str, str]]],
    *,
    round_index: int,
    total_rounds: int,
) -> str:
    payload = {
        "baseline": asdict(baseline),
        "round_index": round_index,
        "total_rounds": total_rounds,
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "thought_graph": thought_graph,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_selected_node_ids(
    scoring_payload: dict[str, Any],
    thought_graph: dict[str, list[dict[str, str]]],
) -> list[str]:
    valid_ids = [node["id"] for node in thought_graph.get("thought_nodes", []) if node.get("id")]
    valid_id_set = set(valid_ids)
    raw_ids = scoring_payload.get("selected_node_ids")
    if raw_ids is None:
        raw_ids = scoring_payload.get("selected_ids", [])
    selected: list[str] = []
    if isinstance(raw_ids, list):
        for item in raw_ids:
            node_id = _coerce_string(item)
            if node_id in valid_id_set and node_id not in selected:
                selected.append(node_id)
    if not selected:
        selected = valid_ids[: min(3, len(valid_ids))]
    return selected


def _graph_of_thought_synthesis_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are synthesizing the final proposal for a Graph-of-Thought scientific ideation baseline. "
        f"{guidance} "
        "Stage 3 converts the selected thought subgraph into one concise structured research idea. "
        "Follow the selected path and repair instructions, but do not mention the internal graph in the final proposal. "
        "Do not add unsupported datasets, hidden target-paper labels, or generic filler. "
        "Return strict JSON only, with no markdown or commentary. "
        'JSON schema: {"title":"...","problem":"...","existing_methods":"...","motivation":"...",'
        '"hypothesis":"...","method":"...","evaluation":"...","significance":"...","caveats":"..."}'
    )


def _graph_of_thought_synthesis_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    thought_graph: dict[str, list[dict[str, str]]],
    scoring_payload: dict[str, Any],
    selected_node_ids: list[str],
) -> str:
    selected_id_set = set(selected_node_ids)
    selected_nodes = [
        node for node in thought_graph.get("thought_nodes", [])
        if node.get("id") in selected_id_set
    ]
    selected_edges = [
        edge for edge in thought_graph.get("thought_edges", [])
        if edge.get("source") in selected_id_set and edge.get("target") in selected_id_set
    ]
    payload = {
        "baseline": asdict(baseline),
        "topic": graph.topic,
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "literature_grounding": _grounding_brief(graph),
        "selected_subgraph": {
            "thought_nodes": selected_nodes,
            "thought_edges": selected_edges,
        },
        "graph_critique": _coerce_string(scoring_payload.get("critique")),
        "repair_instructions": scoring_payload.get("repair_instructions", []),
        "scores": scoring_payload.get("scores", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _ai_researcher_seed_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    candidate_count = max(2, baseline.candidate_count)
    return (
        "You are implementing a lightweight AI-Researcher-style baseline for scientific idea generation. "
        f"{guidance} "
        "Benchmark faithfulness is the first requirement. "
        "Stage 1 is seed ideation only. Use the topic and reference packet as inspiration, but do not copy or restate the references. "
        f"Generate exactly {candidate_count} diverse seed ideas with clearly different mechanisms or problem framings. "
        "The ideas must stay inside the benchmark task family rather than drifting to a nearby but different task. "
        "Return strict JSON only, with no markdown or commentary. "
        'JSON schema: {"seed_ideas":[{"idea_name":"...","problem_focus":"...","existing_gap":"...","core_mechanism":"...","evaluation_hint":"..."}]}'
    )


def _ai_researcher_seed_user_prompt(graph: IdeaGraph, baseline: BaselineSpec) -> str:
    payload = {
        "baseline": asdict(baseline),
        "topic": graph.topic,
        "task_instruction": graph.metadata.get("task_instruction", ""),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _ai_researcher_focus_constraints(graph),
        "anchor_terms": _ai_researcher_anchor_terms(graph),
        "candidate_count": max(2, baseline.candidate_count),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_seed_idea_payload(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "idea_name": _coerce_string(payload.get("idea_name") or payload.get("title") or payload.get("name")),
        "problem_focus": _coerce_string(payload.get("problem_focus") or payload.get("problem")),
        "existing_gap": _coerce_string(payload.get("existing_gap") or payload.get("gap") or payload.get("existing_methods")),
        "core_mechanism": _coerce_string(payload.get("core_mechanism") or payload.get("mechanism") or payload.get("hypothesis")),
        "evaluation_hint": _coerce_string(payload.get("evaluation_hint") or payload.get("evaluation")),
    }


def _ai_researcher_expansion_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are implementing a lightweight AI-Researcher-style baseline for scientific idea generation. "
        f"{guidance} "
        "Stage 2 expands one seed idea into one full structured proposal. "
        "Preserve benchmark faithfulness: if the seed drifts away from the benchmark task, pull it back toward the benchmark topic instead of amplifying the drift. "
        "Keep one coherent proposal with one main mechanism, grounded in the provided packet, and non-repetitive across sections. "
        "Do not copy raw extraction fragments from literature snippets; if a detail looks noisy or truncated, omit it. "
        "Return strict JSON only, with no markdown or commentary. "
        'JSON schema: {"title":"...","problem":"...","existing_methods":"...","motivation":"...",'
        '"hypothesis":"...","method":"...","evaluation":"...","significance":"...","caveats":"..."}'
    )


def _ai_researcher_expansion_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    seed_idea: dict[str, str],
) -> str:
    payload = {
        "baseline": asdict(baseline),
        "topic": graph.topic,
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _ai_researcher_focus_constraints(graph),
        "anchor_terms": _ai_researcher_anchor_terms(graph),
        "seed_idea": seed_idea,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _ai_researcher_ranking_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are implementing a lightweight AI-Researcher-style ranking stage. "
        f"{guidance} "
        "Rank the expanded candidates and choose the single best one. "
        "Benchmark fidelity is the first gate: a candidate that sounds exciting but drifts away from the benchmark task should lose. "
        "Use topic fidelity, literature grounding, novelty, significance, feasibility, clarity, and experiment quality. "
        "Penalize noisy copied snippet fragments, unsupported dataset mentions, and multi-mechanism proposals that lack one clear core idea. "
        "Do not reward longer text unless it improves substance. "
        "Return strict JSON only, with no markdown or commentary. "
        'JSON schema: {"selected_index":0,"reason":"...","scores":[{"index":0,"topic_fidelity":1,"novelty":1,"significance":1,"feasibility":1,"clarity":1,"literature_grounding":1,"experiment_quality":1,"overall":1}]}'
    )


def _ai_researcher_ranking_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    candidates: list[FinalProposal],
) -> str:
    payload = {
        "baseline": asdict(baseline),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _ai_researcher_focus_constraints(graph),
        "anchor_terms": _ai_researcher_anchor_terms(graph),
        "candidates": [
            {"index": index, **_proposal_as_prompt_payload(candidate)}
            for index, candidate in enumerate(candidates)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _llm_json_object(
    backend: OpenAICompatibleCollaborationBackend,
    *,
    role: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    attempt_messages = list(messages)
    last_error = "Unknown JSON decoding failure."
    for attempt in range(backend.settings.max_retries + 1):
        result = backend.client.create_chat_completion(
            messages=attempt_messages,
            model=backend.settings.model_for_role(role),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        trace: dict[str, object] = {
            "role": role,
            "attempt": attempt + 1,
            "messages": attempt_messages,
            "raw_response": result.raw_response,
        }
        try:
            payload = _extract_json_object(result.content)
            return payload, trace
        except Exception as exc:
            last_error = str(exc)
            trace["error"] = last_error
            if attempt >= backend.settings.max_retries:
                raise
            attempt_messages = attempt_messages + [
                {"role": "assistant", "content": result.content},
                {
                    "role": "user",
                    "content": (
                        "Your last message was not valid strict JSON for the required schema. "
                        "Return one JSON object only, with no markdown, no code fences, and no extra commentary."
                    ),
                },
            ]
    raise ValueError(last_error)


def _candidate_selection_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are ranking candidate scientific research ideas for a baseline wrapper. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Select the single best candidate using novelty, significance, feasibility, clarity, and topic fit. "
        "Benchmark fidelity and literature grounding should dominate tie-breaking. "
        "Prefer the candidate with the strongest overall research promise rather than the longest text. "
        'JSON schema: {"selected_index":0,"reason":"...","scores":[{"index":0,"novelty":1,"significance":1,"feasibility":1,"clarity":1,"topic_fit":1,"overall":1}]}'
    )


def _candidate_selection_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    candidates: list[FinalProposal],
) -> str:
    payload = {
        "baseline": asdict(baseline),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "candidates": [
            {"index": index, **_proposal_as_prompt_payload(candidate)}
            for index, candidate in enumerate(candidates)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _llm_direct_proposal(graph: IdeaGraph, baseline: BaselineSpec, backend: OpenAICompatibleCollaborationBackend) -> FinalProposal:
    payload, trace = _llm_json_object(
        backend,
        role="BaselineDirect",
        messages=[
            {"role": "system", "content": _direct_system_prompt(baseline)},
            {"role": "user", "content": _direct_user_prompt(graph, baseline)},
        ],
        temperature=0.35,
        max_tokens=1400,
    )
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "direct_generation", "baseline": baseline.name, **trace}
    )
    proposal = _proposal_from_payload(payload)
    return _baseline_postprocess_proposal(graph, baseline, proposal)


def _llm_candidate_rank_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    backend: OpenAICompatibleCollaborationBackend,
    progress_callback: Callable[[str], None] | None = None,
) -> FinalProposal:
    if baseline.prompt_style == "ai_researcher_guided":
        return _llm_ai_researcher_guided_proposal(
            graph,
            baseline,
            backend,
            progress_callback=progress_callback,
        )

    generation_payload, generation_trace = _llm_json_object(
        backend,
        role="BaselineCandidateGeneration",
        messages=[
            {"role": "system", "content": _candidate_generation_system_prompt(baseline)},
            {"role": "user", "content": _candidate_generation_user_prompt(graph, baseline)},
        ],
        temperature=0.7,
        max_tokens=2200,
    )
    raw_candidates = generation_payload.get("candidates", [])
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("Candidate-generation response did not contain a non-empty 'candidates' list.")

    candidates = [
        _baseline_postprocess_proposal(graph, baseline, _proposal_from_payload(item))
        for item in raw_candidates
        if isinstance(item, dict)
    ]
    if not candidates:
        raise ValueError("Candidate-generation response did not contain any valid proposal objects.")

    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "candidate_generation", "baseline": baseline.name, **generation_trace}
    )

    selection_payload, selection_trace = _llm_json_object(
        backend,
        role="BaselineCandidateSelection",
        messages=[
            {"role": "system", "content": _candidate_selection_system_prompt(baseline)},
            {"role": "user", "content": _candidate_selection_user_prompt(graph, baseline, candidates)},
        ],
        temperature=0.0,
        max_tokens=1400,
    )
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "candidate_selection", "baseline": baseline.name, **selection_trace}
    )
    try:
        selected_index = int(selection_payload.get("selected_index", 0))
    except (TypeError, ValueError):
        selected_index = 0
    if not (0 <= selected_index < len(candidates)):
        selected_index = 0

    selected = _baseline_postprocess_proposal(graph, baseline, candidates[selected_index])
    selection_reason = _coerce_string(selection_payload.get("reason"))
    if selection_reason:
        graph.metadata["baseline_selection_reason"] = selection_reason
    return selected


def _llm_ai_researcher_guided_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    backend: OpenAICompatibleCollaborationBackend,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> FinalProposal:
    emit_progress(
        graph,
        progress_callback,
        stage="baseline_seed_generation",
        message=f"Baseline '{baseline.name}': generating literature-grounded seed ideas.",
    )
    seed_payload, seed_trace = _llm_json_object(
        backend,
        role="BaselineSeedGeneration",
        messages=[
            {"role": "system", "content": _ai_researcher_seed_system_prompt(baseline)},
            {"role": "user", "content": _ai_researcher_seed_user_prompt(graph, baseline)},
        ],
        temperature=0.8,
        max_tokens=1800,
    )
    raw_seed_ideas = seed_payload.get("seed_ideas", [])
    if not isinstance(raw_seed_ideas, list) or not raw_seed_ideas:
        raise ValueError("AI-Researcher-style seed generation did not return a non-empty 'seed_ideas' list.")
    seed_ideas = [
        _normalize_seed_idea_payload(item)
        for item in raw_seed_ideas
        if isinstance(item, dict)
    ]
    seed_ideas = [item for item in seed_ideas if any(item.values())]
    if not seed_ideas:
        raise ValueError("AI-Researcher-style seed generation did not return any usable seed ideas.")
    graph.metadata["ai_researcher_guided_seed_ideas"] = seed_ideas
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "seed_idea_generation", "baseline": baseline.name, **seed_trace}
    )

    candidates: list[FinalProposal] = []
    expansion_errors: list[dict[str, str]] = []
    for index, seed_idea in enumerate(seed_ideas):
        emit_progress(
            graph,
            progress_callback,
            stage="baseline_candidate_expansion",
            message=(
                f"Baseline '{baseline.name}': expanding seed candidate {index + 1}/{len(seed_ideas)}"
            ),
        )
        try:
            expansion_payload, expansion_trace = _llm_json_object(
                backend,
                role="BaselineCandidateExpansion",
                messages=[
                    {"role": "system", "content": _ai_researcher_expansion_system_prompt(baseline)},
                    {"role": "user", "content": _ai_researcher_expansion_user_prompt(graph, baseline, seed_idea)},
                ],
                temperature=0.35,
                max_tokens=1800,
            )
            proposal = _proposal_from_payload(expansion_payload)
            if not proposal.title:
                proposal.title = seed_idea.get("idea_name", "")
            if not proposal.problem:
                proposal.problem = seed_idea.get("problem_focus", "")
            if not proposal.hypothesis:
                proposal.hypothesis = seed_idea.get("core_mechanism", "")
            if not proposal.evaluation:
                proposal.evaluation = seed_idea.get("evaluation_hint", "")
            proposal = _baseline_postprocess_proposal(graph, baseline, proposal)
            candidates.append(proposal)
            graph.metadata.setdefault("baseline_traces", []).append(
                {
                    "stage": "candidate_expansion",
                    "baseline": baseline.name,
                    "seed_index": index,
                    "seed_idea": seed_idea,
                    **expansion_trace,
                }
            )
        except Exception as exc:
            expansion_errors.append(
                {
                    "seed_index": str(index),
                    "idea_name": seed_idea.get("idea_name", ""),
                    "error": str(exc),
                }
            )
    if expansion_errors:
        graph.metadata["ai_researcher_guided_expansion_errors"] = expansion_errors
    if not candidates:
        raise ValueError("AI-Researcher-style expansion did not produce any valid candidate proposals.")

    graph.metadata["ai_researcher_guided_candidate_count"] = len(candidates)
    graph.metadata["ai_researcher_guided_candidates"] = [
        _proposal_as_prompt_payload(candidate)
        for candidate in candidates
    ]

    emit_progress(
        graph,
        progress_callback,
        stage="baseline_candidate_selection",
        message=f"Baseline '{baseline.name}': ranking expanded candidates.",
    )
    ranking_error = ""
    try:
        selection_payload, selection_trace = _llm_json_object(
            backend,
            role="BaselineCandidateSelection",
            messages=[
                {"role": "system", "content": _ai_researcher_ranking_system_prompt(baseline)},
                {"role": "user", "content": _ai_researcher_ranking_user_prompt(graph, baseline, candidates)},
            ],
            temperature=0.0,
            max_tokens=1600,
        )
        graph.metadata.setdefault("baseline_traces", []).append(
            {"stage": "candidate_selection", "baseline": baseline.name, **selection_trace}
        )
        selection_reason = _coerce_string(selection_payload.get("reason"))
        scores = selection_payload.get("scores", [])
        score_rows = scores if isinstance(scores, list) else []
        if isinstance(scores, list):
            graph.metadata["ai_researcher_guided_scores"] = scores
        llm_overall_by_index: dict[int, float] = {}
        for row in score_rows:
            if not isinstance(row, dict):
                continue
            try:
                row_index = int(row.get("index", -1))
            except (TypeError, ValueError):
                continue
            try:
                llm_overall_by_index[row_index] = float(row.get("overall", 0.0))
            except (TypeError, ValueError):
                llm_overall_by_index[row_index] = 0.0

        combined_rows: list[dict[str, float | int]] = []
        for index, candidate in enumerate(candidates):
            candidate_for_scoring = _baseline_postprocess_proposal(graph, baseline, candidate)
            topic_fidelity = _ai_researcher_topic_fidelity_score(graph, candidate_for_scoring)
            llm_overall = llm_overall_by_index.get(index, 0.0)
            combined_score = llm_overall + 3.0 * topic_fidelity
            combined_rows.append(
                {
                    "index": index,
                    "llm_overall": round(llm_overall, 3),
                    "topic_fidelity": round(topic_fidelity, 3),
                    "combined_score": round(combined_score, 3),
                }
            )
        graph.metadata["ai_researcher_guided_combined_scores"] = combined_rows
        best_row = max(combined_rows, key=lambda item: float(item["combined_score"]))
        selected_index = int(best_row["index"])
        selected = _baseline_postprocess_proposal(graph, baseline, candidates[selected_index])
        if selection_reason:
            try:
                llm_selected_index = int(selection_payload.get("selected_index", selected_index))
            except (TypeError, ValueError):
                llm_selected_index = selected_index
            if selected_index != llm_selected_index:
                graph.metadata["baseline_selection_reason"] = (
                    selection_reason
                    + " Final selection was adjusted by a deterministic topic-fidelity reranker to reduce benchmark drift."
                )
            else:
                graph.metadata["baseline_selection_reason"] = selection_reason
        return selected
    except Exception as exc:
        ranking_error = str(exc)

    graph.metadata["baseline_selection_error"] = ranking_error
    return _baseline_postprocess_proposal(graph, baseline, candidates[0])


def _llm_self_refine_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    backend: OpenAICompatibleCollaborationBackend,
) -> FinalProposal:
    draft = _llm_direct_proposal(graph, baseline, backend)
    critique_payload, critique_trace = _llm_json_object(
        backend,
        role="BaselineCritique",
        messages=[
            {"role": "system", "content": _critique_system_prompt(baseline)},
            {"role": "user", "content": _critique_user_prompt(graph, baseline, draft)},
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "self_refine_critique", "baseline": baseline.name, **critique_trace}
    )
    payload, refine_trace = _llm_json_object(
        backend,
        role="BaselineRevision",
        messages=[
            {"role": "system", "content": _refine_system_prompt(baseline)},
            {"role": "user", "content": _refine_user_prompt(graph, baseline, draft, critique_payload)},
        ],
        temperature=0.25,
        max_tokens=1500,
    )
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "self_refine_revision", "baseline": baseline.name, **refine_trace}
    )
    proposal = _proposal_from_payload(payload)
    return _baseline_postprocess_proposal(graph, baseline, proposal)


def _llm_graph_of_thought_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    backend: OpenAICompatibleCollaborationBackend,
) -> FinalProposal:
    thought_graph: dict[str, list[dict[str, str]]] | None = None
    scoring_payload: dict[str, Any] | None = None
    selected_node_ids: list[str] = []
    round_summaries: list[dict[str, Any]] = []

    for round_index in range(1, GRAPH_OF_THOUGHT_ROUND_COUNT + 1):
        thought_graph_payload, generation_trace = _llm_json_object(
            backend,
            role="BaselineThoughtGraphGeneration",
            messages=[
                {
                    "role": "system",
                    "content": _graph_of_thought_generation_system_prompt(
                        baseline,
                        round_index=round_index,
                        total_rounds=GRAPH_OF_THOUGHT_ROUND_COUNT,
                    ),
                },
                {
                    "role": "user",
                    "content": _graph_of_thought_generation_user_prompt(
                        graph,
                        baseline,
                        round_index=round_index,
                        total_rounds=GRAPH_OF_THOUGHT_ROUND_COUNT,
                        previous_graph=thought_graph,
                        previous_scoring=scoring_payload,
                    ),
                },
            ],
            temperature=0.55,
            max_tokens=2000,
        )
        thought_graph = _normalize_thought_graph_payload(thought_graph_payload)
        if not thought_graph["thought_nodes"]:
            raise ValueError(
                f"Graph-of-Thought generation round {round_index} did not return any usable thought nodes."
            )
        graph.metadata["graph_of_thought_graph"] = thought_graph
        graph.metadata.setdefault("baseline_traces", []).append(
            {
                "stage": f"graph_of_thought_round_{round_index}_generation",
                "baseline": baseline.name,
                "round_index": round_index,
                **generation_trace,
            }
        )

        scoring_payload, scoring_trace = _llm_json_object(
            backend,
            role="BaselineThoughtGraphScoring",
            messages=[
                {
                    "role": "system",
                    "content": _graph_of_thought_scoring_system_prompt(
                        baseline,
                        round_index=round_index,
                        total_rounds=GRAPH_OF_THOUGHT_ROUND_COUNT,
                    ),
                },
                {
                    "role": "user",
                    "content": _graph_of_thought_scoring_user_prompt(
                        graph,
                        baseline,
                        thought_graph,
                        round_index=round_index,
                        total_rounds=GRAPH_OF_THOUGHT_ROUND_COUNT,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=1400,
        )
        selected_node_ids = _normalize_selected_node_ids(scoring_payload, thought_graph)
        repair_instructions = scoring_payload.get("repair_instructions", [])
        normalized_repair_instructions = (
            [_coerce_string(item) for item in repair_instructions if _coerce_string(item)]
            if isinstance(repair_instructions, list)
            else []
        )
        round_summaries.append(
            {
                "round": round_index,
                "node_count": len(thought_graph["thought_nodes"]),
                "edge_count": len(thought_graph["thought_edges"]),
                "selected_node_ids": selected_node_ids,
                "critique": _coerce_string(scoring_payload.get("critique")),
                "repair_instructions": normalized_repair_instructions,
                "scores": scoring_payload.get("scores", []),
            }
        )
        graph.metadata["graph_of_thought_selected_node_ids"] = selected_node_ids
        graph.metadata["graph_of_thought_critique"] = _coerce_string(scoring_payload.get("critique"))
        graph.metadata["graph_of_thought_repair_instructions"] = normalized_repair_instructions
        graph.metadata.setdefault("baseline_traces", []).append(
            {
                "stage": f"graph_of_thought_round_{round_index}_scoring",
                "baseline": baseline.name,
                "round_index": round_index,
                **scoring_trace,
            }
        )

    if thought_graph is None or scoring_payload is None:
        raise ValueError("Graph-of-Thought did not complete any generation/scoring rounds.")

    graph.metadata["graph_of_thought_round_count"] = GRAPH_OF_THOUGHT_ROUND_COUNT
    graph.metadata["graph_of_thought_rounds"] = round_summaries

    proposal_payload, synthesis_trace = _llm_json_object(
        backend,
        role="BaselineThoughtGraphSynthesis",
        messages=[
            {"role": "system", "content": _graph_of_thought_synthesis_system_prompt(baseline)},
            {
                "role": "user",
                "content": _graph_of_thought_synthesis_user_prompt(
                    graph,
                    baseline,
                    thought_graph,
                    scoring_payload,
                    selected_node_ids,
                ),
            },
        ],
        temperature=0.25,
        max_tokens=1600,
    )
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "graph_of_thought_synthesis", "baseline": baseline.name, **synthesis_trace}
    )
    proposal = _proposal_from_payload(proposal_payload)
    return _baseline_postprocess_proposal(graph, baseline, proposal)


def _deterministic_graph_of_thought_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
) -> FinalProposal:
    draft = _deterministic_direct_proposal(graph, baseline)
    thought_graph = {
        "thought_nodes": [
            {"id": "n1", "type": "problem_gap", "text": draft.problem},
            {"id": "n2", "type": "mechanism", "text": draft.method},
            {"id": "n3", "type": "evaluation", "text": draft.evaluation},
            {"id": "n4", "type": "impact", "text": draft.significance},
        ],
        "thought_edges": [
            {"source": "n1", "relation": "motivates", "target": "n2", "rationale": "The mechanism addresses the problem gap."},
            {"source": "n2", "relation": "tested_by", "target": "n3", "rationale": "The evaluation checks the proposed mechanism."},
            {"source": "n2", "relation": "enables", "target": "n4", "rationale": "The mechanism drives the expected impact."},
        ],
    }
    graph.metadata["graph_of_thought_graph"] = thought_graph
    graph.metadata["graph_of_thought_selected_node_ids"] = ["n1", "n2", "n3"]
    graph.metadata["graph_of_thought_round_count"] = GRAPH_OF_THOUGHT_ROUND_COUNT
    graph.metadata["graph_of_thought_rounds"] = [
        {
            "round": round_index,
            "node_count": len(thought_graph["thought_nodes"]),
            "edge_count": len(thought_graph["thought_edges"]),
            "selected_node_ids": ["n1", "n2", "n3"],
            "critique": "Deterministic Graph-of-Thought fallback preserves the strongest problem-method-evaluation path.",
            "repair_instructions": [],
            "scores": [],
        }
        for round_index in range(1, GRAPH_OF_THOUGHT_ROUND_COUNT + 1)
    ]
    return _deterministic_refine_proposal(graph, draft, baseline)


def _build_baseline_graph(
    instance,
    *,
    baseline: BaselineSpec,
) -> IdeaGraph:
    graph = IdeaGraph(topic=instance.topic, literature=list(instance.literature), metadata=dict(instance.metadata))
    graph.metadata.setdefault(
        "literature_grounding",
        build_literature_grounding(
            literature=graph.literature,
            metadata=_generation_metadata(graph),
        ).as_dict(),
    )
    graph.metadata["baseline_name"] = baseline.name
    graph.metadata["baseline_display_name"] = baseline.display_name
    graph.metadata["baseline_strategy"] = baseline.strategy
    graph.metadata["baseline_prompt_style"] = baseline.prompt_style
    graph.metadata["baseline_local_variant"] = baseline.is_local_variant
    graph.metadata["baseline_reference_target"] = baseline.reference_target
    graph.metadata["baseline_description"] = baseline.description
    graph.metadata.setdefault("instance_name", instance.name)
    return graph


def _maybe_build_runtime_controller(graph: IdeaGraph, baseline: BaselineSpec) -> tuple[Any | None, dict[str, Any] | None]:
    runtime_kind = str(graph.metadata.get("runtime_controller_kind", "")).strip() or baseline.runtime_controller
    if not runtime_kind:
        return None, None

    if runtime_kind == "text_critic_rerank":
        model_path = Path(
            str(
                graph.metadata.get("runtime_controller_model_path")
                or _default_text_critic_model_path()
            )
        )
        if not model_path.exists():
            graph.metadata["runtime_controller_error"] = f"Missing runtime controller model at {model_path}."
            return None, None

        model = load_pickled_text_critic_model(str(model_path))
        config = TextCriticRuntimeConfig(
            tau_override=float(graph.metadata.get("runtime_controller_tau_override", 0.05)),
            tau_override_by_round=_coerce_round_float_map(
                graph.metadata.get("runtime_controller_tau_override_by_round", {})
            ),
            tau_commit=float(graph.metadata.get("runtime_controller_tau_commit", 0.08)),
            gamma_commit=float(graph.metadata.get("runtime_controller_gamma_commit", 0.50)),
            gamma_commit_by_round=_coerce_round_float_map(
                graph.metadata.get("runtime_controller_gamma_commit_by_round", {})
            ),
            min_commit_round=int(graph.metadata.get("runtime_controller_min_commit_round", 3)),
            use_commit=bool(graph.metadata.get("runtime_controller_use_commit", False)),
            guard_support_threshold=float(
                graph.metadata.get("runtime_controller_guard_support_threshold", 0.66)
            ),
            guard_support_gain_floor=float(
                graph.metadata.get("runtime_controller_guard_support_gain_floor", 0.10)
            ),
            guard_requires_contradiction_progress=bool(
                graph.metadata.get("runtime_controller_guard_requires_contradiction_progress", False)
            ),
            guard_commit_support_threshold=float(
                graph.metadata.get("runtime_controller_guard_commit_support_threshold", 0.0)
            ),
            guard_commit_utility_floor=float(
                graph.metadata.get("runtime_controller_guard_commit_utility_floor", 0.0)
            ),
        )
        controller_metadata = {
            "kind": "text_critic_rerank",
            "model_path": str(model_path.resolve()),
            "use_commit": bool(config.use_commit),
            "tau_override": float(config.tau_override),
        }
        graph.metadata["runtime_controller_loaded"] = controller_metadata
        return model, {"config": config, **controller_metadata}

    if runtime_kind == "relation_graph_critic_rerank":
        model_dir = Path(
            str(
                graph.metadata.get("runtime_controller_model_dir")
                or _default_relation_graph_runtime_model_dir()
            )
        )
        if not model_dir.exists():
            graph.metadata["runtime_controller_error"] = f"Missing runtime controller model directory at {model_dir}."
            return None, None

        runtime_bundle = load_relation_graph_runtime_bundle(model_dir)
        config = RelationGraphRuntimeConfig(
            tau_override=float(graph.metadata.get("runtime_controller_tau_override", 0.05)),
            tau_override_by_round=_coerce_round_float_map(
                graph.metadata.get("runtime_controller_tau_override_by_round", {})
            ),
            tau_commit=float(graph.metadata.get("runtime_controller_tau_commit", 0.08)),
            gamma_commit=float(graph.metadata.get("runtime_controller_gamma_commit", 0.50)),
            gamma_commit_by_round=_coerce_round_float_map(
                graph.metadata.get("runtime_controller_gamma_commit_by_round", {})
            ),
            min_commit_round=int(graph.metadata.get("runtime_controller_min_commit_round", 3)),
            use_edit=bool(graph.metadata.get("runtime_controller_use_edit", True)),
            use_commit=bool(graph.metadata.get("runtime_controller_use_commit", False)),
            use_low_signal_kind_swap_guard=bool(
                graph.metadata.get("runtime_controller_use_low_signal_kind_swap_guard", False)
            ),
            guard_support_threshold=float(
                graph.metadata.get("runtime_controller_guard_support_threshold", 0.66)
            ),
            guard_support_gain_floor=float(
                graph.metadata.get("runtime_controller_guard_support_gain_floor", 0.10)
            ),
            guard_requires_contradiction_progress=bool(
                graph.metadata.get("runtime_controller_guard_requires_contradiction_progress", False)
            ),
            guard_commit_support_threshold=float(
                graph.metadata.get("runtime_controller_guard_commit_support_threshold", 0.0)
            ),
            guard_commit_utility_floor=float(
                graph.metadata.get("runtime_controller_guard_commit_utility_floor", 0.0)
            ),
            use_action_score_calibration=bool(
                graph.metadata.get("runtime_controller_use_action_score_calibration", False)
            ),
            action_score_calibration_strength=float(
                graph.metadata.get("runtime_controller_action_score_calibration_strength", 0.35)
            ),
            action_score_calibration_max_bias=float(
                graph.metadata.get("runtime_controller_action_score_calibration_max_bias", 0.35)
            ),
        )
        controller_metadata = {
            "kind": "relation_graph_critic_rerank",
            "model_dir": str(model_dir.resolve()),
            "model_path": str(model_dir.resolve()),
            "use_commit": bool(config.use_commit),
            "tau_override": float(config.tau_override),
            "use_action_score_calibration": bool(config.use_action_score_calibration),
            "action_score_calibration_strength": float(config.action_score_calibration_strength),
            "action_score_calibration_max_bias": float(config.action_score_calibration_max_bias),
        }
        graph.metadata["runtime_controller_loaded"] = controller_metadata
        return runtime_bundle, {"config": config, **controller_metadata}

    if runtime_kind == "relation_graph_two_head_critic":
        model_dir = Path(
            str(
                graph.metadata.get("runtime_controller_model_dir")
                or _default_relation_graph_two_head_runtime_model_dir()
            )
        )
        if not model_dir.exists():
            graph.metadata["runtime_controller_error"] = f"Missing runtime controller model directory at {model_dir}."
            return None, None

        calibrated_metadata = _apply_joint_runtime_calibration(dict(graph.metadata), model_dir)
        for key in (
            "runtime_controller_calibration_path",
            "runtime_controller_calibration_source",
            "runtime_controller_calibration_version",
        ):
            if key not in calibrated_metadata:
                graph.metadata.pop(key, None)
        graph.metadata.update(calibrated_metadata)
        runtime_bundle = load_relation_graph_two_head_runtime_bundle(model_dir)
        config = RelationGraphRuntimeConfig(
            tau_override=float(graph.metadata.get("runtime_controller_tau_override", 0.05)),
            tau_override_by_round=_coerce_round_float_map(
                graph.metadata.get("runtime_controller_tau_override_by_round", {})
            ),
            tau_commit=float(graph.metadata.get("runtime_controller_tau_commit", 0.08)),
            gamma_commit=float(graph.metadata.get("runtime_controller_gamma_commit", 0.50)),
            gamma_commit_by_round=_coerce_round_float_map(
                graph.metadata.get("runtime_controller_gamma_commit_by_round", {})
            ),
            min_commit_round=int(graph.metadata.get("runtime_controller_min_commit_round", 3)),
            use_edit=bool(graph.metadata.get("runtime_controller_use_edit", True)),
            use_commit=bool(graph.metadata.get("runtime_controller_use_commit", True)),
            use_low_signal_kind_swap_guard=bool(
                graph.metadata.get("runtime_controller_use_low_signal_kind_swap_guard", False)
            ),
            guard_support_threshold=float(
                graph.metadata.get("runtime_controller_guard_support_threshold", 0.66)
            ),
            guard_support_gain_floor=float(
                graph.metadata.get("runtime_controller_guard_support_gain_floor", 0.10)
            ),
            guard_requires_contradiction_progress=bool(
                graph.metadata.get("runtime_controller_guard_requires_contradiction_progress", False)
            ),
            guard_commit_support_threshold=float(
                graph.metadata.get("runtime_controller_guard_commit_support_threshold", 0.0)
            ),
            guard_commit_utility_floor=float(
                graph.metadata.get("runtime_controller_guard_commit_utility_floor", 0.0)
            ),
            use_action_score_calibration=bool(
                graph.metadata.get("runtime_controller_use_action_score_calibration", True)
            ),
            action_score_calibration_strength=float(
                graph.metadata.get("runtime_controller_action_score_calibration_strength", 0.35)
            ),
            action_score_calibration_max_bias=float(
                graph.metadata.get("runtime_controller_action_score_calibration_max_bias", 0.35)
            ),
        )
        controller_metadata = {
            "kind": "relation_graph_two_head_critic",
            "model_dir": str(model_dir.resolve()),
            "model_path": str(model_dir.resolve()),
            "use_commit": bool(config.use_commit),
            "tau_override": float(config.tau_override),
            "use_action_score_calibration": bool(config.use_action_score_calibration),
            "action_score_calibration_strength": float(config.action_score_calibration_strength),
            "action_score_calibration_max_bias": float(config.action_score_calibration_max_bias),
        }
        graph.metadata["runtime_controller_loaded"] = controller_metadata
        return runtime_bundle, {"config": config, **controller_metadata}

    if runtime_kind == "fixed_control":
        policy_path = Path(
            str(
                graph.metadata.get("runtime_controller_policy_path")
                or _default_fixed_control_policy_path()
            )
        )
        if not policy_path.exists():
            graph.metadata["runtime_controller_error"] = f"Missing fixed control policy at {policy_path}."
            return None, None

        controller = load_fixed_control_policy(policy_path)
        config = RelationGraphRuntimeConfig(use_edit=True, use_commit=False)
        controller_metadata = {
            "kind": "fixed_control",
            "policy_path": str(policy_path.resolve()),
            "use_commit": False,
        }
        graph.metadata["runtime_controller_loaded"] = controller_metadata
        return controller, {"config": config, **controller_metadata}

    if runtime_kind == "random_control":
        base_seed = int(graph.metadata.get("runtime_controller_random_seed", 0) or 0)
        restart_offset = int(graph.metadata.get("batch_restart", 0) or 0)
        seed = base_seed + restart_offset
        controller = RandomControlPolicy(seed=seed)
        config = RelationGraphRuntimeConfig(use_edit=True, use_commit=False)
        controller_metadata = {
            "kind": "random_control",
            "seed": seed,
            "use_commit": False,
        }
        graph.metadata["runtime_controller_loaded"] = controller_metadata
        return controller, {"config": config, **controller_metadata}

    if runtime_kind == "signal_heuristic_control":
        controller = SignalHeuristicController()
        config = RelationGraphRuntimeConfig(
            tau_override=float(graph.metadata.get("runtime_controller_tau_override", 0.0) or 0.0),
            gamma_commit=float(graph.metadata.get("runtime_controller_gamma_commit", 0.58) or 0.58),
            gamma_commit_by_round=dict(
                graph.metadata.get(
                    "runtime_controller_gamma_commit_by_round",
                    {
                        1: 0.95,
                        2: 0.74,
                        3: 0.62,
                        4: 0.58,
                        5: 0.55,
                        6: 0.53,
                    },
                )
            ),
            min_commit_round=int(graph.metadata.get("runtime_controller_min_commit_round", 2) or 2),
            use_edit=bool(graph.metadata.get("runtime_controller_use_edit", True)),
            use_commit=bool(graph.metadata.get("runtime_controller_use_commit", True)),
        )
        controller_metadata = {
            "kind": "signal_heuristic_control",
            "use_commit": bool(config.use_commit),
        }
        graph.metadata["runtime_controller_loaded"] = controller_metadata
        return controller, {"config": config, **controller_metadata}

    graph.metadata["runtime_controller_error"] = f"Unsupported runtime controller kind '{runtime_kind}'."
    return None, None


def run_baseline_experiment(
    instance,
    *,
    baseline_name: str,
    collaboration_backend=None,
    progress_callback: Callable[[str], None] | None = None,
    max_rounds: int = 3,
    stop_when_mature: bool = True,
    external_baseline_config: dict[str, dict[str, Any]] | None = None,
):
    baseline = get_baseline_spec(baseline_name)
    if (
        _coerce_string(instance.metadata.get("baseline_name")) != baseline.name
        or "benchmark_input_packet" not in instance.metadata
    ):
        instance = attach_baseline_metadata(instance, baseline_name=baseline_name, io_mode="auto")

    if baseline.strategy == "evolving_graph":
        baseline_graph = _build_baseline_graph(instance, baseline=baseline)
        runtime_controller = None
        runtime_controller_metadata = None
        if bool(baseline_graph.metadata.get("runtime_controller_enabled", False)):
            runtime_controller_kind = str(
                baseline_graph.metadata.get("runtime_controller_kind", baseline.runtime_controller)
            ).strip()
            try:
                runtime_controller, runtime_controller_metadata = _maybe_build_runtime_controller(
                    baseline_graph,
                    baseline,
                )
            except Exception as exc:
                baseline_graph.metadata["runtime_controller_error"] = str(exc)
                raise RuntimeError(
                    f"Runtime controller '{runtime_controller_kind or baseline.runtime_controller}' failed to load "
                    f"for baseline '{baseline.name}': {exc}"
                ) from exc
            if runtime_controller is None or runtime_controller_metadata is None:
                error_detail = str(
                    baseline_graph.metadata.get("runtime_controller_error", "unknown runtime controller load failure")
                ).strip()
                raise RuntimeError(
                    f"Runtime controller '{runtime_controller_kind or baseline.runtime_controller}' failed "
                    f"for baseline '{baseline.name}': {error_detail}"
                )
        return run_experiment(
            topic=instance.topic,
            literature=list(instance.literature),
            metadata=dict(baseline_graph.metadata),
            collaboration_backend=collaboration_backend,
            progress_callback=progress_callback,
            max_rounds=max_rounds,
            stop_when_mature=stop_when_mature,
            runtime_controller=runtime_controller,
            runtime_controller_metadata=runtime_controller_metadata,
        )

    graph = _build_baseline_graph(instance, baseline=baseline)
    emit_progress(
        graph,
        progress_callback,
        stage="start",
        message=(
            f"Initialized baseline '{baseline.name}' using strategy '{baseline.strategy}' for topic '{graph.topic}'."
        ),
        details={"baseline": baseline.name, "strategy": baseline.strategy},
    )

    if baseline.strategy == "external":
        from .external_baselines import run_external_baseline

        emit_progress(
            graph,
            progress_callback,
            stage="baseline_generation",
            message=f"Running external baseline '{baseline.name}' through its upstream repository adapter.",
        )
        proposal = run_external_baseline(
            graph,
            baseline_name=baseline.name,
            external_config=external_baseline_config,
            progress_callback=progress_callback,
        )
    elif baseline.strategy == "direct":
        emit_progress(
            graph,
            progress_callback,
            stage="baseline_generation",
            message=f"Generating a one-pass structured idea with baseline '{baseline.name}'.",
        )
        if isinstance(collaboration_backend, OpenAICompatibleCollaborationBackend):
            try:
                proposal = _llm_direct_proposal(graph, baseline, collaboration_backend)
            except Exception as exc:
                graph.metadata["baseline_generation_error"] = str(exc)
                emit_progress(
                    graph,
                    progress_callback,
                    stage="baseline_fallback",
                    message=(
                        f"Baseline '{baseline.name}' returned an invalid LLM response. "
                        "Falling back to the deterministic baseline implementation."
                    ),
                )
                proposal = _deterministic_direct_proposal(graph, baseline)
        else:
            proposal = _deterministic_direct_proposal(graph, baseline)
    elif baseline.strategy == "candidate_rank":
        emit_progress(
            graph,
            progress_callback,
            stage="baseline_generation",
            message=(
                f"Generating and selecting diverse structured idea candidates with baseline '{baseline.name}'."
            ),
        )
        if isinstance(collaboration_backend, OpenAICompatibleCollaborationBackend):
            try:
                proposal = _llm_candidate_rank_proposal(
                    graph,
                    baseline,
                    collaboration_backend,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                graph.metadata["baseline_generation_error"] = str(exc)
                emit_progress(
                    graph,
                    progress_callback,
                    stage="baseline_fallback",
                    message=(
                        f"Baseline '{baseline.name}' returned an invalid LLM response. "
                        "Falling back to the deterministic baseline implementation."
                    ),
                )
                draft = _deterministic_direct_proposal(graph, baseline)
                proposal = _deterministic_refine_proposal(graph, draft, baseline)
        else:
            draft = _deterministic_direct_proposal(graph, baseline)
            proposal = _deterministic_refine_proposal(graph, draft, baseline)
    elif baseline.strategy == "self_refine":
        emit_progress(
            graph,
            progress_callback,
            stage="baseline_generation",
            message=f"Generating and refining a structured idea with baseline '{baseline.name}'.",
        )
        if isinstance(collaboration_backend, OpenAICompatibleCollaborationBackend):
            try:
                proposal = _llm_self_refine_proposal(graph, baseline, collaboration_backend)
            except Exception as exc:
                graph.metadata["baseline_generation_error"] = str(exc)
                emit_progress(
                    graph,
                    progress_callback,
                    stage="baseline_fallback",
                    message=(
                        f"Baseline '{baseline.name}' returned an invalid LLM response. "
                        "Falling back to the deterministic baseline implementation."
                    ),
                )
                draft = _deterministic_direct_proposal(graph, baseline)
                proposal = _deterministic_refine_proposal(graph, draft, baseline)
        else:
            draft = _deterministic_direct_proposal(graph, baseline)
            proposal = _deterministic_refine_proposal(graph, draft, baseline)
    elif baseline.strategy == "graph_of_thought":
        emit_progress(
            graph,
            progress_callback,
            stage="baseline_generation",
            message=f"Generating a Graph-of-Thought proposal with baseline '{baseline.name}'.",
        )
        if isinstance(collaboration_backend, OpenAICompatibleCollaborationBackend):
            try:
                proposal = _llm_graph_of_thought_proposal(graph, baseline, collaboration_backend)
            except Exception as exc:
                graph.metadata["baseline_generation_error"] = str(exc)
                emit_progress(
                    graph,
                    progress_callback,
                    stage="baseline_fallback",
                    message=(
                        f"Baseline '{baseline.name}' returned an invalid LLM response. "
                        "Falling back to the deterministic Graph-of-Thought implementation."
                    ),
                )
                proposal = _deterministic_graph_of_thought_proposal(graph, baseline)
        else:
            proposal = _deterministic_graph_of_thought_proposal(graph, baseline)
    else:
        raise ValueError(f"Unsupported baseline strategy '{baseline.strategy}'.")

    graph.final_subgraph = {"node_ids": [], "edge_ids": [], "utility": 0.0}
    graph.final_proposal = proposal
    graph.metadata["max_rounds_requested"] = 0
    graph.metadata["stop_when_mature"] = False
    graph.metadata["executed_round_count"] = 0
    graph.metadata["stopped_early"] = False
    if baseline.strategy == "external":
        graph.metadata["stop_reason"] = f"baseline_{baseline.name}_complete"
    else:
        graph.metadata["stop_reason"] = f"baseline_{baseline.strategy}_complete"

    emit_progress(
        graph,
        progress_callback,
        stage="complete",
        message=(
            f"Baseline run complete for '{baseline.name}': {len(graph.nodes)} nodes, "
            f"{len(graph.edges)} edges, {len(graph.actions)} actions."
        ),
        details={"baseline": baseline.name, "nodes": len(graph.nodes), "edges": len(graph.edges)},
    )
    return graph
