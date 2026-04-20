from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from .baselines import attach_baseline_metadata


@dataclass(frozen=True)
class ExperimentMethodPlan:
    name: str
    baseline_name: str
    restarts: int
    max_rounds: int
    stop_when_mature: bool
    rationale: str
    runtime_protocol: str = "sequential_v1"
    metadata_overrides: dict[str, Any] = field(default_factory=dict)
    strip_reference_packet: bool = False
    strip_paper_grounding: bool = False
    strip_literature_grounding: bool = False
    strip_literature_list: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


METHOD_PLAN_ALIASES: dict[str, str] = {
    "ours-delayed-consensus": "ours-eig",
}


def canonical_method_plan_name(name: str) -> str:
    cleaned = str(name).strip()
    return METHOD_PLAN_ALIASES.get(cleaned, cleaned)


MAIN_METHOD_PLANS: dict[str, ExperimentMethodPlan] = {
    "direct": ExperimentMethodPlan(
        name="direct",
        baseline_name="direct",
        restarts=1,
        max_rounds=1,
        stop_when_mature=True,
        rationale="Single-pass lower bound under the shared benchmark-facing output contract.",
    ),
    "self-refine": ExperimentMethodPlan(
        name="self-refine",
        baseline_name="self-refine",
        restarts=1,
        max_rounds=1,
        stop_when_mature=True,
        rationale="Single-agent draft-critique-revise baseline.",
    ),
    "ai-researcher": ExperimentMethodPlan(
        name="ai-researcher",
        baseline_name="ai-researcher",
        restarts=1,
        max_rounds=1,
        stop_when_mature=True,
        rationale="Exact literature-grounded AI-Researcher baseline under the shared benchmark-facing I/O contract.",
    ),
    "scipip": ExperimentMethodPlan(
        name="scipip",
        baseline_name="scipip",
        restarts=1,
        max_rounds=1,
        stop_when_mature=True,
        rationale="Paper-faithful SciPIP bridge baseline under the shared benchmark-facing I/O contract.",
    ),
    "virsci": ExperimentMethodPlan(
        name="virsci",
        baseline_name="virsci",
        restarts=1,
        max_rounds=1,
        stop_when_mature=True,
        rationale="Paper-faithful fixed-topic VirSci bridge baseline under the shared benchmark-facing I/O contract.",
    ),
    "scipip-proxy": ExperimentMethodPlan(
        name="scipip-proxy",
        baseline_name="scipip-proxy",
        restarts=1,
        max_rounds=1,
        stop_when_mature=True,
        rationale="Structured decomposition proxy baseline.",
    ),
    "ai-researcher-proxy": ExperimentMethodPlan(
        name="ai-researcher-proxy",
        baseline_name="ai-researcher-proxy",
        restarts=1,
        max_rounds=1,
        stop_when_mature=True,
        rationale="Literature-grounded candidate-generation proxy baseline.",
    ),
    "ours-eig": ExperimentMethodPlan(
        name="ours-eig",
        baseline_name="ours-eig",
        restarts=1,
        max_rounds=5,
        stop_when_mature=True,
        runtime_protocol="parallel_graph_v2",
        rationale="Evolving Idea Graph multi-agent collaboration with maturity-based early stopping.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_heuristic",
        },
    ),
    "ours-eig-critic-graph-twohead": ExperimentMethodPlan(
        name="ours-eig-critic-graph-twohead",
        baseline_name="ours-eig-critic-graph-twohead",
        restarts=1,
        max_rounds=5,
        stop_when_mature=True,
        runtime_protocol="parallel_graph_v2",
        rationale="Parallel EIG with a shared-encoder two-head graph critic for role-local edit control and post-round commit prediction.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_twohead_critic",
        },
    ),
}


ABLATION_METHOD_PLANS: dict[str, ExperimentMethodPlan] = {
    "ours-eig": MAIN_METHOD_PLANS["ours-eig"],
    "ours-eig-critic-graph-twohead": MAIN_METHOD_PLANS["ours-eig-critic-graph-twohead"],
    "ours-eig-critic-text": ExperimentMethodPlan(
        name="ours-eig-critic-text",
        baseline_name="ours-eig-critic-text",
        restarts=1,
        max_rounds=5,
        stop_when_mature=True,
        runtime_protocol="parallel_graph_v2",
        rationale="Parallel EIG with a text-only critic for role-local edit selection.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_text_critic",
        },
    ),
    "ours-eig-critic-calibrated": ExperimentMethodPlan(
        name="ours-eig-critic-calibrated",
        baseline_name="ours-eig-critic-calibrated",
        restarts=1,
        max_rounds=5,
        stop_when_mature=True,
        runtime_protocol="parallel_graph_v2",
        rationale="Parallel EIG with the two-head graph critic and frozen-dev controller calibration.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_twohead_calibrated",
        },
    ),
    "ours-eig-critic-no-commit": ExperimentMethodPlan(
        name="ours-eig-critic-no-commit",
        baseline_name="ours-eig-critic-no-commit",
        restarts=1,
        max_rounds=5,
        stop_when_mature=False,
        runtime_protocol="parallel_graph_v2",
        rationale="Parallel EIG where the graph critic selects edits, but stopping is fixed-horizon.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_twohead_no_commit",
        },
    ),
    "ours-eig-critic-no-edit": ExperimentMethodPlan(
        name="ours-eig-critic-no-edit",
        baseline_name="ours-eig-critic-no-edit",
        restarts=1,
        max_rounds=5,
        stop_when_mature=True,
        runtime_protocol="parallel_graph_v2",
        rationale="Parallel EIG where heuristic role-local edits are retained and the learned commit head controls stopping.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_twohead_no_edit",
        },
    ),
    "ours-eig-fixed-control": ExperimentMethodPlan(
        name="ours-eig-fixed-control",
        baseline_name="ours-eig-fixed-control",
        restarts=1,
        max_rounds=5,
        stop_when_mature=False,
        runtime_protocol="parallel_graph_v2",
        rationale="Parallel EIG with a frozen role-and-round edit policy and fixed five-round stopping.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_fixed_control",
        },
    ),
    "ours-eig-random-control": ExperimentMethodPlan(
        name="ours-eig-random-control",
        baseline_name="ours-eig-random-control",
        restarts=3,
        max_rounds=5,
        stop_when_mature=False,
        runtime_protocol="parallel_graph_v2",
        rationale="Parallel EIG with seeded random legal action selection and fixed five-round stopping.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_random_control",
        },
    ),
    "ours-early-consensus": ExperimentMethodPlan(
        name="ours-early-consensus",
        baseline_name="ours-eig",
        restarts=1,
        max_rounds=2,
        stop_when_mature=False,
        runtime_protocol="parallel_graph_v2",
        rationale="EIG with early commitment: synthesize after a short fixed horizon instead of waiting for structural maturity.",
        metadata_overrides={
            "idea_graph_protocol_variant": "early_consensus",
            "idea_graph_delayed_consensus": False,
        },
    ),
    "ours-no-maturity-stop": ExperimentMethodPlan(
        name="ours-no-maturity-stop",
        baseline_name="ours-eig",
        restarts=1,
        max_rounds=5,
        stop_when_mature=False,
        runtime_protocol="parallel_graph_v2",
        rationale="Full idea graph without maturity-based early stopping.",
        metadata_overrides={
            "idea_graph_protocol_variant": "no_maturity_stop",
        },
    ),
    "ours-no-coverage-safeguard": ExperimentMethodPlan(
        name="ours-no-coverage-safeguard",
        baseline_name="ours-eig",
        restarts=1,
        max_rounds=5,
        stop_when_mature=True,
        runtime_protocol="parallel_graph_v2",
        rationale="Full idea graph without the core-node completeness safeguard added after the hard-case failure analysis.",
        metadata_overrides={
            "idea_graph_protocol_variant": "no_coverage_safeguard",
            "idea_graph_disable_core_node_coverage": True,
        },
    ),
    "ours-no-reference-grounding": ExperimentMethodPlan(
        name="ours-no-reference-grounding",
        baseline_name="ours-eig",
        restarts=1,
        max_rounds=5,
        stop_when_mature=True,
        runtime_protocol="parallel_graph_v2",
        rationale="Full idea graph with the benchmark topic preserved but reference-packet and paper-grounding evidence removed.",
        metadata_overrides={
            "idea_graph_protocol_variant": "no_reference_grounding",
        },
        strip_reference_packet=True,
        strip_paper_grounding=True,
        strip_literature_grounding=True,
        strip_literature_list=True,
    ),
    "virsci-proxy": ExperimentMethodPlan(
        name="virsci-proxy",
        baseline_name="virsci-proxy",
        restarts=1,
        max_rounds=5,
        stop_when_mature=True,
        rationale="Discussion-oriented multi-agent proxy baseline sharing the same benchmark-facing I/O contract.",
    ),
}


METHOD_PLAN_PRESETS: dict[str, dict[str, ExperimentMethodPlan]] = {
    "main": MAIN_METHOD_PLANS,
    "ablation": ABLATION_METHOD_PLANS,
}


def get_method_plan_catalog(preset: str) -> dict[str, ExperimentMethodPlan]:
    try:
        return METHOD_PLAN_PRESETS[preset]
    except KeyError as exc:
        options = ", ".join(sorted(METHOD_PLAN_PRESETS))
        raise KeyError(f"Unknown method-plan preset '{preset}'. Available presets: {options}") from exc


def method_plan_choices(preset: str) -> list[str]:
    names = list(get_method_plan_catalog(preset))
    names.extend(alias for alias, target in METHOD_PLAN_ALIASES.items() if target in names)
    return sorted(set(names))


def prepare_instance_for_method_plan(instance, *, plan: ExperimentMethodPlan):
    prepared = attach_baseline_metadata(instance, baseline_name=plan.baseline_name, io_mode="auto")
    metadata = deepcopy(prepared.metadata)
    metadata["method_name"] = plan.name
    metadata["runner_baseline_name"] = plan.baseline_name
    metadata["method_plan"] = plan.as_dict()
    metadata["runtime_protocol"] = plan.runtime_protocol
    metadata["idea_graph_protocol_variant"] = metadata.get("idea_graph_protocol_variant") or plan.name
    metadata.update(deepcopy(plan.metadata_overrides))

    if plan.strip_reference_packet:
        benchmark_packet = metadata.get("benchmark_input_packet", {})
        if isinstance(benchmark_packet, dict):
            benchmark_packet = deepcopy(benchmark_packet)
            benchmark_packet["reference_packet"] = []
            constraints = benchmark_packet.get("constraints", [])
            if not isinstance(constraints, list):
                constraints = []
            constraints = list(constraints)
            note = "Reference packet intentionally removed for this protocol ablation."
            if note not in constraints:
                constraints.append(note)
            benchmark_packet["constraints"] = constraints
            metadata["benchmark_input_packet"] = benchmark_packet

    if plan.strip_paper_grounding:
        metadata["paper_grounding"] = {"reference_paper_snippets": []}
        metadata["reference_titles"] = []

    if plan.strip_literature_grounding:
        metadata["literature_grounding"] = {
            "source": "protocol_ablation",
            "reference_titles": [],
            "design_highlights": [],
            "dataset_items": [],
            "metric_items": [],
            "existing_methods_summary": "",
            "experiment_plan_summary": "",
        }

    literature = [] if plan.strip_literature_list else list(prepared.literature)
    if not literature:
        benchmark_packet = metadata.get("benchmark_input_packet", {})
        topic = ""
        if isinstance(benchmark_packet, dict):
            topic = str(benchmark_packet.get("topic", "")).strip()
        fallback_topic = topic or str(prepared.topic).strip()
        if fallback_topic:
            literature = [fallback_topic]

    return prepared.__class__(
        name=prepared.name,
        topic=prepared.topic,
        literature=literature,
        source_path=prepared.source_path,
        metadata=metadata,
    )
