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
DEFAULT_PILOT_SUMMARY = ROOT / "outputs" / "pilots" / "20260401-161640-april01-quality-batch-v2" / "pilot_summary.json"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper tables and figures from saved pilot outputs.")
    parser.add_argument("--pilot-summary", type=Path, default=DEFAULT_PILOT_SUMMARY)
    parser.add_argument("--hard-case-old-summary", type=Path, default=DEFAULT_HARD_CASE_OLD_SUMMARY)
    parser.add_argument("--hard-case-new-summary", type=Path, default=DEFAULT_HARD_CASE_NEW_SUMMARY)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_run_dir(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate
    name = candidate.name
    fallback = ROOT / "outputs" / "pilots" / "20260401-161640-april01-quality-batch-v2" / "runs" / name
    if fallback.exists():
        return fallback
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


def render_quality_table(aggregate_rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Quality-first same-backbone pilot on AI Idea Bench 2025 (indices 13, 15, 18, 21, and 33) using \texttt{qwen3-8b}. Higher is better.}",
        r"\label{tab:pilot_quality_main}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Overall & Align. & Expert \\",
        r"\midrule",
    ]
    best_overall = max(row["mean_overall_score"] for row in aggregate_rows)
    best_align = max(row["mean_benchmark_alignment"] for row in aggregate_rows)
    best_expert = max(row["mean_expert_style_quality"] for row in aggregate_rows)
    for row in aggregate_rows:
        method = METHOD_DISPLAY_NAMES.get(row["baseline_name"], row["baseline_name"])
        overall = f"{row['mean_overall_score']:.2f}"
        align = f"{row['mean_benchmark_alignment']:.2f}"
        expert = f"{row['mean_expert_style_quality']:.2f}"
        if row["mean_overall_score"] == best_overall:
            overall = rf"\textbf{{{overall}}}"
        if row["mean_benchmark_alignment"] == best_align:
            align = rf"\textbf{{{align}}}"
        if row["mean_expert_style_quality"] == best_expert:
            expert = rf"\textbf{{{expert}}}"
        lines.append(f"{method} & {overall} & {align} & {expert} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_graph_process_table(ours_rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Per-instance process diagnostics for the delayed-consensus method on the five-instance pilot. Higher support and utility are better; lower unresolved contradiction ratio (UCR) is better.}",
        r"\label{tab:pilot_graph_process}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Index & Graph & Support & UCR & Utility & Rounds & Errors & Stop \\",
        r"\midrule",
    ]
    for row in ours_rows:
        run_dir = resolve_run_dir(row["run_dir"])
        summary_payload = read_run_summary(run_dir)
        graph_payload = read_run_graph(run_dir)
        snapshot = final_round_snapshot(summary_payload)
        index = row["benchmark_index"]
        graph_score = f"{row['graph_process']:.2f}"
        support = f"{float(snapshot.get('support_coverage', 0.0) or 0.0):.2f}"
        ucr = f"{float(snapshot.get('unresolved_contradiction_ratio', 1.0) or 0.0):.2f}"
        utility = f"{float(snapshot.get('utility', 0.0) or 0.0):.1f}"
        rounds = int(summary_payload.get("executed_round_count", 0) or 0)
        errors = action_error_count(graph_payload)
        stop_reason = format_stop_reason(str(summary_payload.get("stop_reason", "")))
        lines.append(
            f"{index} & {graph_score} & {support} & {ucr} & {utility} & {rounds} & {errors} & {stop_reason} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_cost_table(aggregate_rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Supplementary cost analysis for the five-instance pilot. Tokens are averaged over the selected run for each benchmark index.}",
        r"\label{tab:pilot_cost}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Calls & Tokens & $\times$ Direct \\",
        r"\midrule",
    ]
    for row in aggregate_rows:
        method = METHOD_DISPLAY_NAMES.get(row["baseline_name"], row["baseline_name"])
        lines.append(
            f"{method} & {row['mean_llm_call_count']:.1f} & {row['mean_total_tokens']:.0f} & {row['token_multiplier_vs_direct']:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_hard_case_table(old_summary: dict[str, Any], new_summary: dict[str, Any], old_graph: dict[str, Any], new_graph: dict[str, Any]) -> str:
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
        r"\caption{Hard-case fallback analysis on AI Idea Bench 2025 index 18 before and after the core-type coverage safeguard.}",
        r"\label{tab:pilot_hard_case_safeguard}",
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
        ax.plot(x_old, y_old, marker="o", linestyle="--", linewidth=1.8, color=colors["old"], label="Before safeguard")
        ax.plot(x_new, y_new, marker="o", linestyle="-", linewidth=1.8, color=colors["new"], label="After safeguard")
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


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()

    pilot_summary = load_json(args.pilot_summary)
    aggregate_rows = list(pilot_summary.get("aggregate_rows", []))
    selected_rows = list(pilot_summary.get("selected_rows", []))
    ours_rows = [row for row in selected_rows if row.get("baseline_name") == "ours-delayed-consensus"]

    old_summary = load_json(args.hard_case_old_summary)
    new_summary = load_json(args.hard_case_new_summary)
    old_graph = load_json(args.hard_case_old_summary.with_name("graph.json"))
    new_graph = load_json(args.hard_case_new_summary.with_name("graph.json"))

    generated_dir = args.paper_dir / "generated"
    figures_dir = args.paper_dir / "figures"
    ensure_dir(generated_dir)
    ensure_dir(figures_dir)

    write_text(generated_dir / "pilot_quality_main_table.tex", render_quality_table(aggregate_rows))
    write_text(generated_dir / "pilot_graph_process_table.tex", render_graph_process_table(ours_rows))
    write_text(generated_dir / "pilot_cost_table.tex", render_cost_table(aggregate_rows))
    write_text(
        generated_dir / "pilot_hard_case_table.tex",
        render_hard_case_table(old_summary, new_summary, old_graph, new_graph),
    )
    plot_hard_case_trajectory(old_summary, new_summary, figures_dir / "hard_case_round_trajectory.pdf")

    print("Generated paper artifacts:")
    print(generated_dir / "pilot_quality_main_table.tex")
    print(generated_dir / "pilot_graph_process_table.tex")
    print(generated_dir / "pilot_cost_table.tex")
    print(generated_dir / "pilot_hard_case_table.tex")
    print(figures_dir / "hard_case_round_trajectory.pdf")


if __name__ == "__main__":
    main()
