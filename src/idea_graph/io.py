from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import re

from .fs_utils import write_text_file
from .benchmark_scoring import (
    BenchmarkNativeEvaluation,
    BenchmarkNativeMetric,
    format_benchmark_native_markdown,
)
from .engine import graph_as_dict
from .evaluation import evaluate_graph, format_evaluation_markdown
from .instances import ExperimentInstance
from .models import IdeaGraph


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "run"


def _action_source_counts(graph: IdeaGraph) -> dict[str, int]:
    counts = {
        "llm": 0,
        "deterministic": 0,
        "deterministic_fallback": 0,
    }
    for action in graph.actions:
        source = str(getattr(action, "source", "deterministic") or "deterministic").strip()
        if source not in counts:
            counts[source] = 0
        counts[source] += 1
    return counts


def load_instance(path: str | Path) -> ExperimentInstance:
    return ExperimentInstance.from_json_file(path)


def build_run_summary(
    graph: IdeaGraph,
    instance_name: str,
    source_path: str,
    *,
    evaluation_payload: dict[str, object] | None = None,
    native_evaluation_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    if evaluation_payload is None:
        evaluation_payload = evaluate_graph(graph).as_dict()
    final_proposal = {
        "title": graph.final_proposal.title if graph.final_proposal else "",
        "problem": graph.final_proposal.problem if graph.final_proposal else "",
        "existing_methods": graph.final_proposal.existing_methods if graph.final_proposal else "",
        "motivation": graph.final_proposal.motivation if graph.final_proposal else "",
        "hypothesis": graph.final_proposal.hypothesis if graph.final_proposal else "",
        "method": graph.final_proposal.method if graph.final_proposal else "",
        "evaluation": graph.final_proposal.evaluation if graph.final_proposal else "",
        "significance": graph.final_proposal.significance if graph.final_proposal else "",
        "caveats": graph.final_proposal.caveats if graph.final_proposal else "",
    }
    summary = {
        "instance_name": instance_name,
        "source_path": source_path,
        "topic": graph.topic,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "branch_count": len(graph.branches),
        "action_count": len(graph.actions),
        "executed_round_count": graph.metadata.get("executed_round_count", len(graph.round_summaries)),
        "max_rounds_requested": graph.metadata.get("max_rounds_requested"),
        "matured_at_round": graph.matured_at_round,
        "stopped_early": graph.metadata.get("stopped_early", False),
        "stop_reason": graph.metadata.get("stop_reason", "unknown"),
        "action_source_counts": _action_source_counts(graph),
        "rounds": [
            {
                "round": round_name,
                "support_coverage": snapshot.support_coverage,
                "unresolved_contradiction_ratio": snapshot.unresolved_contradiction_ratio,
                "utility": snapshot.utility,
                "utility_breakdown": {
                    "promise": snapshot.utility_breakdown.promise,
                    "support": snapshot.utility_breakdown.support,
                    "coherence": snapshot.utility_breakdown.coherence,
                    "evidence": snapshot.utility_breakdown.evidence,
                    "novelty": snapshot.utility_breakdown.novelty,
                    "contradiction_penalty": snapshot.utility_breakdown.contradiction_penalty,
                    "open_risk_penalty": snapshot.utility_breakdown.open_risk_penalty,
                    "size_penalty": snapshot.utility_breakdown.size_penalty,
                    "total": snapshot.utility_breakdown.total,
                },
                "utility_stable": snapshot.utility_stable,
                "completeness": snapshot.completeness,
                "is_mature": snapshot.is_mature,
            }
            for round_name, snapshot in graph.round_summaries
        ],
        "final_proposal": final_proposal,
        "literature_grounding": graph.metadata.get("literature_grounding", {}),
        "idea_evaluation": evaluation_payload,
    }
    if native_evaluation_payload is not None:
        summary["benchmark_native_evaluation"] = native_evaluation_payload
    return summary


def write_run_artifacts(
    graph: IdeaGraph,
    *,
    output_root: str | Path,
    instance: ExperimentInstance,
    native_evaluation_payload: dict[str, object] | None = None,
) -> Path:
    output_root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{timestamp}-{slugify(instance.name)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    evaluation = evaluate_graph(graph)
    graph_payload = graph_as_dict(graph)
    summary_payload = build_run_summary(
        graph,
        instance_name=instance.name,
        source_path=instance.source_path,
        evaluation_payload=evaluation.as_dict(),
        native_evaluation_payload=native_evaluation_payload,
    )
    if instance.metadata:
        summary_payload["instance_metadata"] = instance.metadata
    evaluation_payload = summary_payload.get("idea_evaluation", {})

    write_text_file(
        run_dir / "graph.json",
        json.dumps(graph_payload, indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(
        run_dir / "summary.json",
        json.dumps(summary_payload, indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(
        run_dir / "evaluation.json",
        json.dumps(evaluation_payload, indent=2, ensure_ascii=False, default=str),
    )

    proposal = summary_payload["final_proposal"]
    title = str(proposal.get("title") or instance.name).strip() or instance.name
    final_proposal_lines = [f"# {title}", ""]

    section_map = [
        ("Problem", "problem"),
        ("Existing Methods", "existing_methods"),
        ("Motivation", "motivation"),
        ("Core Hypothesis", "hypothesis"),
        ("Proposed Method", "method"),
        ("Experiment Plan", "evaluation"),
        ("Expected Contribution", "significance"),
        ("Risks And Limitations", "caveats"),
    ]
    for heading, key in section_map:
        value = str(proposal.get(key, "")).strip()
        if not value:
            continue
        final_proposal_lines.extend([f"## {heading}", value, ""])

    write_text_file(run_dir / "final_proposal.md", "\n".join(final_proposal_lines))
    if evaluation_payload:
        write_text_file(
            run_dir / "evaluation.md",
            format_evaluation_markdown(evaluation),
        )
    if native_evaluation_payload:
        write_text_file(
            run_dir / "benchmark_native_evaluation.json",
            json.dumps(native_evaluation_payload, indent=2, ensure_ascii=False, default=str),
        )
        if isinstance(native_evaluation_payload, dict):
            native_evaluation = BenchmarkNativeEvaluation(
                protocol_name=str(native_evaluation_payload.get("protocol_name", "")).strip(),
                benchmark=str(native_evaluation_payload.get("benchmark", "")).strip(),
                metrics=[],
                summary={},
                notes=[],
            )
            metrics_payload = native_evaluation_payload.get("metrics", [])
            if isinstance(metrics_payload, list):
                native_evaluation = BenchmarkNativeEvaluation(
                    protocol_name=str(native_evaluation_payload.get("protocol_name", "")).strip(),
                    benchmark=str(native_evaluation_payload.get("benchmark", "")).strip(),
                    metrics=[
                        BenchmarkNativeMetric(
                            key=str(item.get("key", "")).strip(),
                            display_name=str(item.get("display_name", "")).strip(),
                            score=float(item.get("score", 0.0) or 0.0),
                            max_score=float(item.get("max_score", 0.0) or 0.0),
                            rationale=str(item.get("rationale", "")).strip(),
                            available=bool(item.get("available", False)),
                            details=dict(item.get("details", {}) or {}),
                        )
                        for item in metrics_payload
                        if isinstance(item, dict)
                    ],
                    summary={
                        str(key): float(value)
                        for key, value in dict(native_evaluation_payload.get("summary", {}) or {}).items()
                    },
                    notes=[
                        str(item).strip()
                        for item in native_evaluation_payload.get("notes", [])
                        if str(item).strip()
                    ],
                )
            write_text_file(
                run_dir / "benchmark_native_evaluation.md",
                format_benchmark_native_markdown(native_evaluation),
            )

    return run_dir
