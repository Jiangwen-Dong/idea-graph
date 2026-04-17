from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.evaluation import evaluate_graph
from idea_graph.models import FinalProposal, IdeaGraph


def detect_paper_dir() -> Path:
    candidates = [
        ROOT.parent / "paper" / "ideation_2026",
        ROOT.parent.parent / "paper" / "ideation_2026",
    ]
    for candidate in candidates:
        if (candidate / "neurips_2025.tex").exists():
            return candidate
    return candidates[0]


PAPER_DIR = detect_paper_dir()
ARCHIVE_ROOT = ROOT / "outputs" / "_archive" / "legacy_runs_pre_20260409"
DEFAULT_BATCH_SUMMARY = (
    ROOT
    / "outputs"
    / "quality_batches"
    / "20260401-merged-cross-benchmark-small-batch"
    / "batch_summary.json"
)
DEFAULT_HARD_CASE_OLD_SUMMARY = (
    ROOT
    / "outputs"
    / "pilots"
    / "20260401-161640-april01-quality-batch-v2"
    / "runs"
    / "20260401-162704-ai-idea-bench-2025-18"
    / "summary.json"
)
DEFAULT_HARD_CASE_NEW_SUMMARY = ARCHIVE_ROOT / "20260401-164026-ai-idea-bench-2025-18" / "summary.json"

METHOD_DISPLAY_NAMES = {
    "direct": "Direct",
    "self-refine": "Self-Refine",
    "ai-researcher": r"\textsc{AI-Researcher}",
    "scipip": r"\textsc{SciPIP}",
    "virsci": r"\textsc{VirSci}",
    "scipip-proxy": r"\textsc{SciPIP}-Proxy",
    "ai-researcher-proxy": r"\textsc{AI-Researcher}-Proxy",
    "virsci-proxy": r"\textsc{VirSci}-Proxy",
    "ours-eig": "Ours (EIG Prototype)",
    "ours-delayed-consensus": "Ours (EIG Prototype)",
    "ours-early-consensus": "Ours (Early Commit)",
    "ours-no-maturity-stop": "Ours w/o Maturity Stop",
    "ours-no-coverage-safeguard": "Ours w/o Coverage Safeguard",
    "ours-no-reference-grounding": "Ours w/o Reference Grounding",
}

BENCHMARK_DISPLAY_NAMES = {
    "AI_Idea_Bench_2025": "AI Idea Bench 2025",
    "liveideabench": "LiveIdeaBench",
}

BENCHMARK_SHORT_NAMES = {
    "AI_Idea_Bench_2025": "AI",
    "liveideabench": "Live",
}

METHOD_ORDER = [
    "direct",
    "self-refine",
    "ai-researcher",
    "scipip",
    "virsci",
    "ours-eig",
]

BENCHMARK_ORDER = ["AI_Idea_Bench_2025", "liveideabench"]


def normalize_method_name(name: str) -> str:
    cleaned = str(name).strip()
    if cleaned == "ours-delayed-consensus":
        return "ours-eig"
    return cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paper-ready tables and figures from the merged cross-benchmark quality batch."
    )
    parser.add_argument("--batch-summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--hard-case-old-summary", type=Path, default=DEFAULT_HARD_CASE_OLD_SUMMARY)
    parser.add_argument("--hard-case-new-summary", type=Path, default=DEFAULT_HARD_CASE_NEW_SUMMARY)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def resolve_run_dir(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    run_dir_name = candidate.name
    outputs_root = ROOT / "outputs"
    for match in outputs_root.rglob(run_dir_name):
        if match.is_dir():
            return match

    raise FileNotFoundError(f"Could not resolve run directory: {path_str}")


def format_stop_reason(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "--"
    if text.startswith("mature_at_"):
        return text.replace("mature_at_", "mature@")
    if text == "max_rounds_reached":
        return "max-rounds"
    return text.replace("_", "-")


def numeric_round(round_name: str) -> int:
    digits = "".join(ch for ch in str(round_name) if ch.isdigit())
    return int(digits) if digits else 0


def read_run_summary(run_dir: Path) -> dict[str, Any]:
    return load_json(run_dir / "summary.json")


def read_run_graph(run_dir: Path) -> dict[str, Any]:
    return load_json(run_dir / "graph.json")


def evaluate_run_outcomes(run_dir: Path) -> dict[str, float]:
    summary_payload = read_run_summary(run_dir)
    metadata = dict(summary_payload.get("instance_metadata", {}) or {})
    literature_grounding = summary_payload.get("literature_grounding", {})
    if isinstance(literature_grounding, dict) and literature_grounding:
        metadata.setdefault("literature_grounding", literature_grounding)
    final_proposal_payload = dict(summary_payload.get("final_proposal", {}) or {})

    graph = IdeaGraph(
        topic=str(summary_payload.get("topic", "")).strip(),
        literature=[],
        metadata=metadata,
    )
    graph.final_proposal = FinalProposal(
        title=str(final_proposal_payload.get("title", "")).strip(),
        problem=str(final_proposal_payload.get("problem", "")).strip(),
        existing_methods=str(final_proposal_payload.get("existing_methods", "")).strip(),
        motivation=str(final_proposal_payload.get("motivation", "")).strip(),
        hypothesis=str(final_proposal_payload.get("hypothesis", "")).strip(),
        method=str(final_proposal_payload.get("method", "")).strip(),
        evaluation=str(final_proposal_payload.get("evaluation", "")).strip(),
        significance=str(final_proposal_payload.get("significance", "")).strip(),
        caveats=str(final_proposal_payload.get("caveats", "")).strip(),
    )
    evaluation = evaluate_graph(graph)
    return {
        "overall_score": float(evaluation.overall_score),
        "benchmark_alignment": float(evaluation.category_scores.get("benchmark_alignment", 0.0) or 0.0),
        "expert_style_quality": float(evaluation.category_scores.get("expert_style_quality", 0.0) or 0.0),
    }


def reevaluated_selected_rows(batch_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in batch_summary.get("selected_rows", []):
        run_dir = resolve_run_dir(str(row["run_dir"]))
        refreshed = evaluate_run_outcomes(run_dir)
        merged = dict(row)
        merged.update(refreshed)
        rows.append(merged)
    return rows


def reevaluated_aggregate_rows(batch_summary: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in reevaluated_selected_rows(batch_summary):
        key = (str(row["benchmark"]), row_method_name(row))
        groups.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (benchmark, baseline_name), items in sorted(groups.items()):
        aggregates.append(
            {
                "benchmark": benchmark,
                "baseline_name": baseline_name,
                "mean_overall_score": sum(float(item["overall_score"]) for item in items) / len(items),
                "mean_benchmark_alignment": sum(float(item["benchmark_alignment"]) for item in items) / len(items),
                "mean_expert_style_quality": sum(float(item["expert_style_quality"]) for item in items) / len(items),
                "mean_graph_process": sum(float(item["graph_process"]) for item in items) / len(items),
                "mean_llm_call_count": sum(float(item["llm_call_count"]) for item in items) / len(items),
                "mean_total_tokens": sum(float(item["total_tokens"]) for item in items) / len(items),
            }
        )
    return aggregates


def reevaluated_overall_rows(batch_summary: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in reevaluated_selected_rows(batch_summary):
        groups.setdefault(row_method_name(row), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for baseline_name, items in sorted(groups.items()):
        aggregates.append(
            {
                "baseline_name": baseline_name,
                "mean_overall_score": sum(float(item["overall_score"]) for item in items) / len(items),
                "mean_benchmark_alignment": sum(float(item["benchmark_alignment"]) for item in items) / len(items),
                "mean_expert_style_quality": sum(float(item["expert_style_quality"]) for item in items) / len(items),
                "mean_graph_process": sum(float(item["graph_process"]) for item in items) / len(items),
                "mean_llm_call_count": sum(float(item["llm_call_count"]) for item in items) / len(items),
                "mean_total_tokens": sum(float(item["total_tokens"]) for item in items) / len(items),
            }
        )
    return aggregates


def final_round_snapshot(summary_payload: dict[str, Any]) -> dict[str, Any]:
    rounds = summary_payload.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        return {
            "support_coverage": 0.0,
            "unresolved_contradiction_ratio": 1.0,
            "utility": 0.0,
        }
    snapshot = rounds[-1]
    if not isinstance(snapshot, dict):
        return {
            "support_coverage": 0.0,
            "unresolved_contradiction_ratio": 1.0,
            "utility": 0.0,
        }
    return snapshot


def action_error_count(graph_payload: dict[str, Any]) -> int:
    metadata = graph_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return 0
    errors = metadata.get("action_errors", [])
    return len(errors) if isinstance(errors, list) else 0


def action_source_counts(graph_payload: dict[str, Any]) -> dict[str, int]:
    counts = {
        "llm": 0,
        "deterministic": 0,
        "deterministic_fallback": 0,
    }
    actions = graph_payload.get("actions", [])
    if (
        isinstance(actions, list)
        and actions
        and any(isinstance(action, dict) and "source" in action for action in actions)
    ):
        for action in actions:
            if not isinstance(action, dict):
                continue
            source = str(action.get("source", "deterministic") or "deterministic").strip()
            counts[source] = counts.get(source, 0) + 1
        return counts

    metadata = graph_payload.get("metadata", {})
    progress_log = metadata.get("progress_log", []) if isinstance(metadata, dict) else []
    if not isinstance(progress_log, list):
        return counts
    for entry in progress_log:
        if not isinstance(entry, dict) or entry.get("stage") != "action_applied":
            continue
        details = entry.get("details", {})
        if not isinstance(details, dict):
            continue
        source = str(details.get("action_source", "deterministic") or "deterministic").strip()
        counts[source] = counts.get(source, 0) + 1
    return counts


def row_method_name(row: dict[str, Any]) -> str:
    return normalize_method_name(str(row.get("method_name") or row.get("baseline_name")))


def aggregate_lookup(aggregate_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["benchmark"]), row_method_name(row)): row
        for row in aggregate_rows
    }


def overall_lookup(overall_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row_method_name(row): row for row in overall_rows}


def token_multiplier_lookup(overall_rows: list[dict[str, Any]]) -> dict[str, float]:
    direct_row = next((row for row in overall_rows if row_method_name(row) == "direct"), None)
    direct_tokens = float(direct_row["mean_total_tokens"]) if direct_row is not None else 0.0
    lookup: dict[str, float] = {}
    for row in overall_rows:
        baseline_name = row_method_name(row)
        mean_tokens = float(row["mean_total_tokens"])
        lookup[baseline_name] = 0.0 if direct_tokens <= 0 else round(mean_tokens / direct_tokens, 2)
    return lookup


def selected_ours_rows(batch_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in batch_summary.get("selected_rows", [])
        if row_method_name(row) == "ours-eig"
    ]
    return sorted(
        rows,
        key=lambda row: (
            BENCHMARK_ORDER.index(str(row["benchmark"])),
            int(row["selector"]),
        ),
    )


def graph_summary_records(batch_summary: dict[str, Any]) -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, float | str]]] = {
        "Overall": [],
        "AI Idea Bench 2025": [],
        "LiveIdeaBench": [],
    }

    for row in selected_ours_rows(batch_summary):
        run_dir = resolve_run_dir(str(row["run_dir"]))
        summary_payload = read_run_summary(run_dir)
        graph_payload = read_run_graph(run_dir)
        snapshot = final_round_snapshot(summary_payload)
        benchmark_key = BENCHMARK_DISPLAY_NAMES.get(str(row["benchmark"]), str(row["benchmark"]))
        grouped["Overall"].append(
            {
                "graph": float(row["graph_process"]),
                "support": float(snapshot.get("support_coverage", 0.0) or 0.0),
                "ucr": float(snapshot.get("unresolved_contradiction_ratio", 1.0) or 0.0),
                "utility": float(snapshot.get("utility", 0.0) or 0.0),
                "rounds": float(summary_payload.get("executed_round_count", 0) or 0.0),
                "errors": float(action_error_count(graph_payload)),
                "mature": 1.0 if str(summary_payload.get("stop_reason", "")).startswith("mature_at_") else 0.0,
            }
        )
        grouped[benchmark_key].append(grouped["Overall"][-1])

    records: list[dict[str, float | str]] = []
    for label in ("Overall", "AI Idea Bench 2025", "LiveIdeaBench"):
        rows = grouped[label]
        if not rows:
            continue
        count = float(len(rows))
        records.append(
            {
                "slice": label,
                "graph": sum(float(item["graph"]) for item in rows) / count,
                "support": sum(float(item["support"]) for item in rows) / count,
                "ucr": sum(float(item["ucr"]) for item in rows) / count,
                "utility": sum(float(item["utility"]) for item in rows) / count,
                "rounds": sum(float(item["rounds"]) for item in rows) / count,
                "errors": sum(float(item["errors"]) for item in rows) / count,
                "mature_rate": sum(float(item["mature"]) for item in rows) / count,
            }
        )
    return records


def render_graph_summary_table(batch_summary: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Compact idea-graph analysis for the current EIG prototype on the cross-benchmark small batch. Higher Graph, Support, Utility, and Mature Rate are better; lower UCR is better.}",
        r"\label{tab:cross_benchmark_graph_summary}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Slice & Graph & Support & UCR & Utility & Mature Rate \\",
        r"\midrule",
    ]
    for row in graph_summary_records(batch_summary):
        lines.append(
            f"{row['slice']} & {float(row['graph']):.2f} & {float(row['support']):.2f} & {float(row['ucr']):.2f} & {float(row['utility']):.2f} & {float(row['mature_rate']):.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_main_quality_table(batch_summary: dict[str, Any]) -> str:
    aggregate_rows = reevaluated_aggregate_rows(batch_summary)
    lookup = aggregate_lookup(aggregate_rows)

    best_by_metric: dict[tuple[str, str], float] = {}
    for benchmark in BENCHMARK_ORDER:
        for metric in (
            "mean_overall_score",
            "mean_benchmark_alignment",
            "mean_expert_style_quality",
        ):
            values = [
                float(lookup[(benchmark, method)][metric])
                for method in METHOD_ORDER
                if (benchmark, method) in lookup
            ]
            if values:
                best_by_metric[(benchmark, metric)] = max(values)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Small-batch cross-benchmark comparison on AI Idea Bench 2025 and LiveIdeaBench under the same benchmark-facing input/output contract. External baselines are reported through their configured benchmark-faithful adapters when exact benchmark entrypoints are unavailable. Higher is better. The main outcome numbers are recomputed with the corrected output-only local scorer; graph-process evidence is reported separately from this outcome table.}",
        r"\label{tab:cross_benchmark_quality_main}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{AI Idea Bench 2025} & \multicolumn{3}{c}{LiveIdeaBench} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & Overall & Align. & Expert & Overall & Align. & Expert \\",
        r"\midrule",
    ]

    for method in METHOD_ORDER:
        method_text = METHOD_DISPLAY_NAMES.get(method, method)
        row_parts = [method_text]
        for benchmark in BENCHMARK_ORDER:
            row = lookup.get((benchmark, method))
            if row is None:
                row_parts.extend(["--", "--", "--"])
                continue

            metric_values = {
                "mean_overall_score": f"{float(row['mean_overall_score']):.2f}",
                "mean_benchmark_alignment": f"{float(row['mean_benchmark_alignment']):.2f}",
                "mean_expert_style_quality": f"{float(row['mean_expert_style_quality']):.2f}",
            }
            for metric_name in (
                "mean_overall_score",
                "mean_benchmark_alignment",
                "mean_expert_style_quality",
            ):
                metric_text = metric_values[metric_name]
                if float(row[metric_name]) == best_by_metric.get((benchmark, metric_name)):
                    metric_text = rf"\textbf{{{metric_text}}}"
                row_parts.append(metric_text)
        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_graph_process_table(batch_summary: dict[str, Any]) -> str:
    selected_rows = selected_ours_rows(batch_summary)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Per-instance graph diagnostics for the current EIG prototype across the eight-target cross-benchmark batch. Higher Graph, Support, and Utility are better; lower unresolved contradiction ratio (UCR) and Errors are better.}",
        r"\label{tab:cross_benchmark_graph_process}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Case & Graph & Support & UCR & Utility & Rounds & Errors & Stop \\",
        r"\midrule",
    ]

    for row in selected_rows:
        run_dir = resolve_run_dir(str(row["run_dir"]))
        summary_payload = read_run_summary(run_dir)
        graph_payload = read_run_graph(run_dir)
        snapshot = final_round_snapshot(summary_payload)
        case_prefix = BENCHMARK_SHORT_NAMES.get(str(row["benchmark"]), str(row["benchmark"]))
        case_name = f"{case_prefix}-{row['selector']}"
        graph_score = f"{float(row['graph_process']):.2f}"
        support = f"{float(snapshot.get('support_coverage', 0.0) or 0.0):.2f}"
        ucr = f"{float(snapshot.get('unresolved_contradiction_ratio', 1.0) or 0.0):.2f}"
        utility = f"{float(snapshot.get('utility', 0.0) or 0.0):.1f}"
        rounds = int(summary_payload.get("executed_round_count", 0) or 0)
        errors = action_error_count(graph_payload)
        stop_reason = format_stop_reason(str(summary_payload.get("stop_reason", "")))
        lines.append(
            f"{case_name} & {graph_score} & {support} & {ucr} & {utility} & {rounds} & {errors} & {stop_reason} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_cost_table(batch_summary: dict[str, Any]) -> str:
    overall_rows = reevaluated_overall_rows(batch_summary)
    lookup = overall_lookup(overall_rows)
    multipliers = token_multiplier_lookup(overall_rows)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Supplementary cost analysis on the eight-target cross-benchmark batch. Calls and tokens are averaged over the selected run for each target.}",
        r"\label{tab:cross_benchmark_cost}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Calls & Tokens & $\times$ Direct \\",
        r"\midrule",
    ]

    for method in METHOD_ORDER:
        row = lookup.get(method)
        if row is None:
            continue
        method_text = METHOD_DISPLAY_NAMES.get(method, method)
        lines.append(
            f"{method_text} & {float(row['mean_llm_call_count']):.1f} & {float(row['mean_total_tokens']):.0f} & {multipliers.get(method, 0.0):.2f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_reliability_table(batch_summary: dict[str, Any]) -> str:
    grouped: dict[str, list[dict[str, float]]] = {
        "Overall": [],
        "AI Idea Bench 2025": [],
        "LiveIdeaBench": [],
    }

    for row in selected_ours_rows(batch_summary):
        run_dir = resolve_run_dir(str(row["run_dir"]))
        graph_payload = read_run_graph(run_dir)
        counts = action_source_counts(graph_payload)
        total_actions = max(1, sum(int(value) for value in counts.values()))
        benchmark_key = BENCHMARK_DISPLAY_NAMES.get(str(row["benchmark"]), str(row["benchmark"]))
        record = {
            "llm_share": counts.get("llm", 0) / total_actions,
            "fallback_share": counts.get("deterministic_fallback", 0) / total_actions,
            "errors": float(action_error_count(graph_payload)),
        }
        grouped["Overall"].append(record)
        grouped[benchmark_key].append(record)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Supplementary reliability analysis for the current EIG prototype. LLM share and fallback share are measured over applied graph actions; errors count invalid or unapplied LLM actions that triggered a deterministic fallback.}",
        r"\label{tab:cross_benchmark_reliability}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Slice & LLM Share & Fallback Share & Errors \\",
        r"\midrule",
    ]

    for label in ("Overall", "AI Idea Bench 2025", "LiveIdeaBench"):
        rows = grouped[label]
        if not rows:
            continue
        count = float(len(rows))
        llm_share = sum(item["llm_share"] for item in rows) / count
        fallback_share = sum(item["fallback_share"] for item in rows) / count
        errors = sum(item["errors"] for item in rows) / count
        lines.append(f"{label} & {llm_share:.2f} & {fallback_share:.2f} & {errors:.2f} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_hard_case_table(
    old_summary: dict[str, Any],
    new_summary: dict[str, Any],
    old_graph: dict[str, Any],
    new_graph: dict[str, Any],
) -> str:
    def row(label: str, summary_payload: dict[str, Any], graph_payload: dict[str, Any]) -> str:
        category_scores = summary_payload.get("idea_evaluation", {}).get("category_scores", {})
        overall = float(summary_payload.get("idea_evaluation", {}).get("overall_score", 0.0) or 0.0)
        graph_score = float(category_scores.get("graph_process", 0.0) or 0.0)
        rounds = int(summary_payload.get("executed_round_count", 0) or 0)
        actions = int(summary_payload.get("action_count", 0) or 0)
        errors = action_error_count(graph_payload)
        stop_reason = format_stop_reason(str(summary_payload.get("stop_reason", "")))
        return f"{label} & {stop_reason} & {rounds} & {actions} & {errors} & {overall:.2f} & {graph_score:.2f} \\\\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Hard-case safeguard analysis on AI Idea Bench 2025 case AI-18 before and after the completeness safeguard.}",
        r"\label{tab:hard_case_safeguard}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Version & Stop & Rounds & Actions & Errors & Overall & Graph \\",
        r"\midrule",
        row("Before safeguard", old_summary, old_graph),
        row("After safeguard", new_summary, new_graph),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def plot_hard_case_trajectory(old_summary: dict[str, Any], new_summary: dict[str, Any], output_path: Path) -> None:
    configure_plot_style()

    old_rounds = old_summary.get("rounds", [])
    new_rounds = new_summary.get("rounds", [])

    def unpack(rounds: list[dict[str, Any]], key: str) -> tuple[list[int], list[float], int | None]:
        xs: list[int] = []
        ys: list[float] = []
        mature_round: int | None = None
        for item in rounds:
            if not isinstance(item, dict):
                continue
            round_idx = numeric_round(str(item.get("round", "")))
            xs.append(round_idx)
            ys.append(float(item.get(key, 0.0) or 0.0))
            if bool(item.get("is_mature", False)):
                mature_round = round_idx
        return xs, ys, mature_round

    metrics = [
        ("support_coverage", "Support coverage"),
        ("unresolved_contradiction_ratio", "Unresolved contradiction ratio"),
        ("utility", "Utility"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8), sharex=False)
    colors = {"old": "#7f7f7f", "new": "#1f77b4"}

    for ax, (key, ylabel) in zip(axes, metrics):
        x_old, y_old, mature_old = unpack(old_rounds, key)
        x_new, y_new, mature_new = unpack(new_rounds, key)
        ax.plot(
            x_old,
            y_old,
            marker="o",
            linestyle="--",
            linewidth=1.8,
            color=colors["old"],
            label="Before safeguard",
        )
        ax.plot(
            x_new,
            y_new,
            marker="o",
            linestyle="-",
            linewidth=1.8,
            color=colors["new"],
            label="After safeguard",
        )
        if mature_old is not None and mature_old in x_old:
            idx = x_old.index(mature_old)
            ax.scatter([mature_old], [y_old[idx]], marker="*", s=70, color=colors["old"], zorder=3)
        if mature_new is not None and mature_new in x_new:
            idx = x_new.index(mature_new)
            ax.scatter([mature_new], [y_new[idx]], marker="*", s=70, color=colors["new"], zorder=3)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(set(x_old + x_new)))

    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(output_path)
    plt.close(fig)


def plot_cross_benchmark_round_trajectory(batch_summary: dict[str, Any], output_path: Path) -> None:
    configure_plot_style()
    grouped: dict[str, dict[int, list[dict[str, float]]]] = {
        "AI Idea Bench 2025": {},
        "LiveIdeaBench": {},
    }

    for row in selected_ours_rows(batch_summary):
        run_dir = resolve_run_dir(str(row["run_dir"]))
        summary_payload = read_run_summary(run_dir)
        benchmark_label = BENCHMARK_DISPLAY_NAMES.get(str(row["benchmark"]), str(row["benchmark"]))
        rounds = summary_payload.get("rounds", [])
        for item in rounds:
            if not isinstance(item, dict):
                continue
            round_idx = numeric_round(str(item.get("round", "")))
            grouped.setdefault(benchmark_label, {}).setdefault(round_idx, []).append(
                {
                    "support": float(item.get("support_coverage", 0.0) or 0.0),
                    "ucr": float(item.get("unresolved_contradiction_ratio", 1.0) or 0.0),
                    "utility": float(item.get("utility", 0.0) or 0.0),
                }
            )

    metrics = [
        ("support", "Support coverage"),
        ("ucr", "Unresolved contradiction ratio"),
        ("utility", "Utility"),
    ]
    colors = {"AI Idea Bench 2025": "#1f77b4", "LiveIdeaBench": "#d62728"}
    linestyles = {"AI Idea Bench 2025": "-", "LiveIdeaBench": "--"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8), sharex=False)

    for ax, (metric_key, ylabel) in zip(axes, metrics):
        for benchmark_label in ("AI Idea Bench 2025", "LiveIdeaBench"):
            round_map = grouped.get(benchmark_label, {})
            xs = sorted(round_map)
            ys = []
            for round_idx in xs:
                values = [item[metric_key] for item in round_map[round_idx]]
                ys.append(sum(values) / len(values))
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                linestyle=linestyles[benchmark_label],
                color=colors[benchmark_label],
                label=benchmark_label,
            )
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        all_xs = sorted({round_idx for rows in grouped.values() for round_idx in rows})
        ax.set_xticks(all_xs)

    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(output_path)
    plt.close(fig)


def plot_graph_quality_scatter(batch_summary: dict[str, Any], output_path: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    colors = {"AI_Idea_Bench_2025": "#1f77b4", "liveideabench": "#d62728"}

    for row in selected_ours_rows(batch_summary):
        benchmark_key = str(row["benchmark"])
        x = float(row["graph_process"])
        y = float(row["overall_score"])
        label = f"{BENCHMARK_SHORT_NAMES.get(benchmark_key, benchmark_key)}-{row['selector']}"
        ax.scatter(x, y, color=colors.get(benchmark_key, "#333333"), s=32)
        ax.annotate(label, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)

    ax.set_xlabel("Graph-process score")
    ax.set_ylabel("Overall score")
    ax.set_title("EIG graph quality vs final quality")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_markdown_notes(batch_summary: dict[str, Any]) -> str:
    aggregate_rows = reevaluated_aggregate_rows(batch_summary)
    overall_rows = reevaluated_overall_rows(batch_summary)
    aggregate = aggregate_lookup(aggregate_rows)
    overall = overall_lookup(overall_rows)
    ours = overall.get("ours-eig", {})
    best_non_graph = max(
        (
            row
            for key, row in overall.items()
            if key != "ours-eig"
        ),
        key=lambda row: float(row["mean_overall_score"]),
    )
    delta = float(ours.get("mean_overall_score", 0.0)) - float(best_non_graph["mean_overall_score"])
    ai_ours = aggregate.get(("AI_Idea_Bench_2025", "ours-eig"), {})
    live_ours = aggregate.get(("liveideabench", "ours-eig"), {})
    ai_best = max(
        (
            row
            for key, row in aggregate.items()
            if key[0] == "AI_Idea_Bench_2025" and key[1] != "ours-eig"
        ),
        key=lambda row: float(row["mean_overall_score"]),
    )
    live_best = max(
        (
            row
            for key, row in aggregate.items()
            if key[0] == "liveideabench" and key[1] != "ours-eig"
        ),
        key=lambda row: float(row["mean_overall_score"]),
    )

    lines = [
        "# Cross-Benchmark Batch Notes",
        "",
        "## Main Takeaways",
        "",
        (
            f"- Across the eight-target small batch, the current EIG prototype reaches "
            f"`{float(ours.get('mean_overall_score', 0.0)):.3f}` overall versus "
            f"`{float(best_non_graph['mean_overall_score']):.3f}` for the strongest non-graph baseline "
            f"(`{best_non_graph['baseline_name']}`), a delta of `{delta:.3f}`."
        ),
        (
            f"- On `AI_Idea_Bench_2025`, ours records "
            f"`{float(ai_ours.get('mean_overall_score', 0.0)):.3f}` overall versus "
            f"`{float(ai_best['mean_overall_score']):.3f}` for `{ai_best['baseline_name']}`."
        ),
        (
            f"- On `liveideabench`, ours records "
            f"`{float(live_ours.get('mean_overall_score', 0.0)):.3f}` overall versus "
            f"`{float(live_best['mean_overall_score']):.3f}` for `{live_best['baseline_name']}`."
        ),
        "",
        "## Interpretation",
        "",
        "- Under the corrected output-only rubric, the current EIG prototype is not yet the strongest final-idea generator in this pilot.",
        "- The graph-summary, reliability, and trajectory plots remain useful as process diagnostics, but they should not be conflated with endpoint quality.",
        "- The current bottleneck is benchmark-facing alignment at synthesis time, especially on the lighter-context LiveIdeaBench slice.",
        "- Cost is substantially higher for the multi-agent graph and should stay in a separate appendix-style analysis rather than the main comparison table.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()

    batch_summary = load_json(args.batch_summary)
    old_summary = load_json(args.hard_case_old_summary)
    new_summary = load_json(args.hard_case_new_summary)
    old_graph = load_json(args.hard_case_old_summary.with_name("graph.json"))
    new_graph = load_json(args.hard_case_new_summary.with_name("graph.json"))

    generated_dir = args.paper_dir / "generated"
    figures_dir = args.paper_dir / "figures"
    ensure_dir(generated_dir)
    ensure_dir(figures_dir)

    write_text(
        generated_dir / "cross_benchmark_quality_main_table.tex",
        render_main_quality_table(batch_summary),
    )
    write_text(
        generated_dir / "cross_benchmark_graph_process_table.tex",
        render_graph_process_table(batch_summary),
    )
    write_text(
        generated_dir / "cross_benchmark_graph_summary_table.tex",
        render_graph_summary_table(batch_summary),
    )
    write_text(
        generated_dir / "cross_benchmark_cost_table.tex",
        render_cost_table(batch_summary),
    )
    write_text(
        generated_dir / "cross_benchmark_reliability_table.tex",
        render_reliability_table(batch_summary),
    )
    write_text(
        generated_dir / "cross_benchmark_hard_case_table.tex",
        render_hard_case_table(old_summary, new_summary, old_graph, new_graph),
    )
    write_text(
        generated_dir / "cross_benchmark_notes.md",
        render_markdown_notes(batch_summary),
    )
    plot_hard_case_trajectory(old_summary, new_summary, figures_dir / "hard_case_round_trajectory.pdf")
    plot_cross_benchmark_round_trajectory(batch_summary, figures_dir / "cross_benchmark_round_trajectory.pdf")
    plot_graph_quality_scatter(batch_summary, figures_dir / "cross_benchmark_graph_quality_scatter.pdf")

    print("Generated paper artifacts:")
    print(generated_dir / "cross_benchmark_quality_main_table.tex")
    print(generated_dir / "cross_benchmark_graph_process_table.tex")
    print(generated_dir / "cross_benchmark_graph_summary_table.tex")
    print(generated_dir / "cross_benchmark_cost_table.tex")
    print(generated_dir / "cross_benchmark_reliability_table.tex")
    print(generated_dir / "cross_benchmark_hard_case_table.tex")
    print(generated_dir / "cross_benchmark_notes.md")
    print(figures_dir / "hard_case_round_trajectory.pdf")
    print(figures_dir / "cross_benchmark_round_trajectory.pdf")
    print(figures_dir / "cross_benchmark_graph_quality_scatter.pdf")


if __name__ == "__main__":
    main()
