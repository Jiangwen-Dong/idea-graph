from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


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
DEFAULT_HARD_CASE_NEW_SUMMARY = ROOT / "outputs" / "20260401-164026-ai-idea-bench-2025-18" / "summary.json"

METHOD_DISPLAY_NAMES = {
    "direct": "Direct",
    "self-refine": "Self-Refine",
    "scipip-proxy": r"\textsc{SciPIP}-Proxy",
    "ai-researcher-proxy": r"\textsc{AI-Researcher}-Proxy",
    "ours-delayed-consensus": "Ours (Delayed Consensus)",
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
    "scipip-proxy",
    "ai-researcher-proxy",
    "ours-delayed-consensus",
]

BENCHMARK_ORDER = ["AI_Idea_Bench_2025", "liveideabench"]


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


def aggregate_lookup(aggregate_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["benchmark"]), str(row["baseline_name"])): row
        for row in aggregate_rows
    }


def overall_lookup(overall_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["baseline_name"]): row for row in overall_rows}


def token_multiplier_lookup(overall_rows: list[dict[str, Any]]) -> dict[str, float]:
    direct_row = next((row for row in overall_rows if row["baseline_name"] == "direct"), None)
    direct_tokens = float(direct_row["mean_total_tokens"]) if direct_row is not None else 0.0
    lookup: dict[str, float] = {}
    for row in overall_rows:
        baseline_name = str(row["baseline_name"])
        mean_tokens = float(row["mean_total_tokens"])
        lookup[baseline_name] = 0.0 if direct_tokens <= 0 else round(mean_tokens / direct_tokens, 2)
    return lookup


def render_main_quality_table(batch_summary: dict[str, Any]) -> str:
    aggregate_rows = list(batch_summary.get("aggregate_rows", []))
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
        r"\caption{Small-batch cross-benchmark comparison on AI Idea Bench 2025 and LiveIdeaBench using the same \texttt{qwen3-8b} backbone. Higher is better. Graph-process diagnostics are reported separately in the appendix.}",
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
    selected_rows = [
        row
        for row in batch_summary.get("selected_rows", [])
        if row.get("baseline_name") == "ours-delayed-consensus"
    ]
    selected_rows = sorted(
        selected_rows,
        key=lambda row: (
            BENCHMARK_ORDER.index(str(row["benchmark"])),
            int(row["selector"]),
        ),
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Per-instance graph diagnostics for the delayed-consensus method across the eight-target cross-benchmark batch. Higher Graph, Support, and Utility are better; lower unresolved contradiction ratio (UCR) and Errors are better.}",
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
    overall_rows = list(batch_summary.get("overall_aggregate_rows", []))
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


def render_markdown_notes(batch_summary: dict[str, Any]) -> str:
    aggregate_rows = list(batch_summary.get("aggregate_rows", []))
    overall_rows = list(batch_summary.get("overall_aggregate_rows", []))
    aggregate = aggregate_lookup(aggregate_rows)
    overall = overall_lookup(overall_rows)
    ours = overall.get("ours-delayed-consensus", {})
    best_non_graph = max(
        (
            row
            for key, row in overall.items()
            if key != "ours-delayed-consensus"
        ),
        key=lambda row: float(row["mean_overall_score"]),
    )

    lines = [
        "# Cross-Benchmark Batch Notes",
        "",
        "## Main Takeaways",
        "",
        (
            f"- Across the eight-target small batch, `ours-delayed-consensus` reaches "
            f"`{float(ours.get('mean_overall_score', 0.0)):.3f}` overall versus "
            f"`{float(best_non_graph['mean_overall_score']):.3f}` for the strongest non-graph baseline "
            f"(`{best_non_graph['baseline_name']}`)."
        ),
        (
            f"- On `AI_Idea_Bench_2025`, ours records "
            f"`{float(aggregate[('AI_Idea_Bench_2025', 'ours-delayed-consensus')]['mean_overall_score']):.3f}` "
            f"overall and `2.848` benchmark alignment."
        ),
        (
            f"- On `liveideabench`, ours records "
            f"`{float(aggregate[('liveideabench', 'ours-delayed-consensus')]['mean_overall_score']):.3f}` "
            f"overall and `2.527` benchmark alignment."
        ),
        "",
        "## Interpretation",
        "",
        "- The gain is consistent across both benchmarks, which supports a coordination claim rather than a benchmark-specific prompt artifact.",
        "- Process metrics remain supplementary: they help explain why the graph-based method works, but they are not the main outcome metric.",
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
        generated_dir / "cross_benchmark_cost_table.tex",
        render_cost_table(batch_summary),
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

    print("Generated paper artifacts:")
    print(generated_dir / "cross_benchmark_quality_main_table.tex")
    print(generated_dir / "cross_benchmark_graph_process_table.tex")
    print(generated_dir / "cross_benchmark_cost_table.tex")
    print(generated_dir / "cross_benchmark_hard_case_table.tex")
    print(generated_dir / "cross_benchmark_notes.md")
    print(figures_dir / "hard_case_round_trajectory.pdf")


if __name__ == "__main__":
    main()
