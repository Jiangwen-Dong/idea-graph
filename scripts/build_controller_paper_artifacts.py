from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_ROWS = ROOT / "outputs" / "quality_batches" / "pilot_main32_critic_graph_merged_20260420-111214" / "selected_rows.csv"
DEFAULT_ABLATION_ROWS = ROOT / "outputs" / "quality_batches" / "pilot_ablation32_controller_merged_20260420-111214" / "selected_rows.csv"
DEFAULT_PAPER_DIR = ROOT.parent / "paper" / "ideation_2026"

CASE_INSTANCE = "liveideabench-parasites-302"
EIG_METHOD = "ours-eig-critic-graph-twohead"
BASELINE_CASE_METHODS = ("self-refine", "direct")

METHOD_LABELS = {
    "ours-eig-critic-graph-twohead": "EIG (Ours)",
    "ours-eig-critic-text": "EIG-Text",
    "ours-eig-critic-no-commit": "EIG-NoCommit",
    "ours-eig-fixed-control": "EIG-Fixed",
    "ours-eig-random-control": "EIG-Random",
}
METHOD_ORDER = ["EIG (Ours)", "EIG-Text", "EIG-NoCommit", "EIG-Fixed", "EIG-Random"]


@dataclass
class ControllerBehaviorSummary:
    stop_counts: dict[str, Counter[int]]
    quality_points: list[dict[str, float | str]]
    method_counts: dict[str, int]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    with _open_read(path) as handle:
        return json.load(handle)


def read_text(path: Path) -> str:
    with _open_read(path) as handle:
        return handle.read()


def _open_read(path: Path):
    resolved = path.resolve(strict=False)
    if os.name == "nt":
        return open("\\\\?\\" + str(resolved), "r", encoding="utf-8")
    return open(resolved, "r", encoding="utf-8")


def resolve_run_dir(path_str: str, root: Path = ROOT) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate
    rooted = root / candidate
    if rooted.exists():
        return rooted
    raise FileNotFoundError(f"Could not resolve run directory: {path_str}")


def _first_heading(markdown: str) -> str:
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _section(markdown: str, heading: str) -> str:
    target = f"## {heading}".casefold()
    lines = markdown.splitlines()
    start: int | None = None
    for idx, line in enumerate(lines):
        if line.strip().casefold() == target:
            start = idx + 1
            break
    if start is None:
        return ""
    collected: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.strip():
            collected.append(line.strip())
    return " ".join(collected)


def _shorten(text: str, width: int = 88, max_lines: int = 3) -> str:
    wrapped = textwrap.wrap(" ".join(text.split()), width=width)
    if not wrapped:
        return ""
    clipped = wrapped[:max_lines]
    if len(wrapped) > max_lines:
        clipped[-1] = clipped[-1].rstrip(".") + "..."
    return "\n".join(clipped)


def _action_label(action: dict[str, Any]) -> str:
    round_name = str(action.get("round_name") or "")
    kind = str(action.get("kind") or "").replace("_", " ")
    if round_name and kind:
        return f"{round_name}: {kind}"
    return " ".join(part for part in [round_name, kind] if part)


def _highlight_actions(actions: list[dict[str, Any]]) -> list[str]:
    priority = ["request_evidence", "attach_evidence", "propose_repair"]
    selected: list[dict[str, Any]] = []
    for kind in priority:
        selected.extend([action for action in actions if str(action.get("kind")) == kind][:1])
    if not selected:
        selected = [action for action in actions[:3] if isinstance(action, dict)]
    return [_action_label(action) for action in selected if isinstance(action, dict)]


def load_case_study_data(
    selected_rows_path: Path,
    *,
    root: Path = ROOT,
    instance_name: str = CASE_INSTANCE,
) -> dict[str, Any]:
    rows = [row for row in read_csv_rows(selected_rows_path) if row.get("instance_name") == instance_name]
    if not rows:
        raise ValueError(f"No selected rows found for instance: {instance_name}")

    baseline_row = next(
        (row for method in BASELINE_CASE_METHODS for row in rows if row.get("baseline_name") == method),
        None,
    )
    eig_row = next((row for row in rows if row.get("baseline_name") == EIG_METHOD), None)
    if baseline_row is None or eig_row is None:
        raise ValueError(f"Need both a baseline and {EIG_METHOD} row for {instance_name}")

    baseline_dir = resolve_run_dir(str(baseline_row["run_dir"]), root=root)
    eig_dir = resolve_run_dir(str(eig_row["run_dir"]), root=root)
    baseline_proposal = read_text(baseline_dir / "final_proposal.md")
    eig_proposal = read_text(eig_dir / "final_proposal.md")
    eig_summary = load_json(eig_dir / "summary.json")
    eig_graph = load_json(eig_dir / "graph.json")

    actions = eig_graph.get("actions", [])
    action_labels = _highlight_actions([action for action in actions if isinstance(action, dict)])
    repair_nodes = [
        node.get("text", "")
        for node in (eig_graph.get("nodes", {}) or {}).values()
        if isinstance(node, dict) and str(node.get("type")) == "Repair"
    ]
    preferred_repair = next((text for text in repair_nodes if "life stage" in text.casefold()), "")

    return {
        "instance_name": instance_name,
        "baseline_method": baseline_row.get("baseline_name", ""),
        "baseline_title": baseline_row.get("title") or _first_heading(baseline_proposal),
        "baseline_score": float(baseline_row.get("overall_score", 0.0) or 0.0),
        "baseline_native": float(baseline_row.get("native_average_normalized_10", 0.0) or 0.0),
        "baseline_method_text": _section(baseline_proposal, "Proposed Method"),
        "baseline_eval_text": _section(baseline_proposal, "Experiment Plan"),
        "eig_title": eig_row.get("title") or _first_heading(eig_proposal),
        "eig_score": float(eig_row.get("overall_score", 0.0) or 0.0),
        "eig_native": float(eig_row.get("native_average_normalized_10", 0.0) or 0.0),
        "eig_method_text": _section(eig_proposal, "Proposed Method"),
        "eig_eval_text": _section(eig_proposal, "Experiment Plan"),
        "eig_rounds": int(eig_summary.get("executed_round_count", 0) or 0),
        "eig_node_count": int(eig_summary.get("node_count", 0) or 0),
        "eig_edge_count": int(eig_summary.get("edge_count", 0) or 0),
        "eig_action_count": int(eig_summary.get("action_count", 0) or 0),
        "action_path": "\n".join(f"- {label}" for label in action_labels),
        "repair_text": preferred_repair or (repair_nodes[0] if repair_nodes else ""),
    }


def summarize_controller_behavior(rows: Iterable[dict[str, str]]) -> ControllerBehaviorSummary:
    stop_counts: dict[str, Counter[int]] = defaultdict(Counter)
    quality_points: list[dict[str, float | str]] = []
    method_counts: Counter[str] = Counter()
    for row in rows:
        label = METHOD_LABELS.get(str(row.get("baseline_name", "")))
        if not label:
            continue
        try:
            round_count = int(float(row.get("executed_round_count", 0) or 0))
            quality = float(row.get("overall_score", 0.0) or 0.0)
        except ValueError:
            continue
        stop_counts[label][round_count] += 1
        method_counts[label] += 1
        quality_points.append({"method": label, "round": float(round_count), "quality": quality})
    return ControllerBehaviorSummary(dict(stop_counts), quality_points, dict(method_counts))


def render_human_eval_instruction_packet() -> str:
    return """# Blind Human Evaluation Instruction Packet

## Goal

This study evaluates final scientific proposals produced by different systems. We sample a small balanced subset from the held-out paper-evaluation groups across AI Idea Bench 2025 and LiveIdeaBench. Reviewers read anonymized proposals and score only the final proposal quality, not the internal reasoning trace or method identity.

## What reviewers see

For each benchmark instance, reviewers receive the benchmark-provided context and a shuffled set of anonymized proposals. Method names, runtime metadata, graph traces, scores, and generation logs are hidden. Reviewers should not search for the hidden target paper or use external references beyond the provided context.

## Rating scale

Use a 1-5 integer scale for each criterion. A score of 1 means poor or missing, 3 means acceptable but limited, and 5 means strong. Use the full scale when proposals differ meaningfully.

## Criteria

- novelty: Does the proposal make a non-trivial research claim relative to the provided context?
- significance: Would solving the proposed problem matter scientifically or practically?
- feasibility: Is the method concrete enough to execute with plausible data, models, and evaluation?
- clarity: Is the proposal coherent, specific, and easy to understand?
- context adherence: Does the proposal stay faithful to the benchmark prompt and visible evidence?
- overall quality: Considering all criteria together, how strong is the proposal?

## Reviewer instructions

1. Read the benchmark context first.
2. Read all anonymized proposals for the same benchmark instance before scoring.
3. Score each proposal independently on all six criteria.
4. Do not reward a proposal for being longer unless the extra detail improves scientific content.
5. Penalize proposals that invent unsupported context, ignore the benchmark topic, or provide only generic method names without a concrete validation plan.
6. If two proposals are similar, use feasibility, context adherence, and evaluation specificity to break ties.

## Reporting

The paper reports mean ratings for each method and criterion, together with the number of benchmark instances, number of raters, assignment pattern, and an inter-rater agreement summary.
"""


def render_case_study_drawing_note(data: dict[str, Any]) -> str:
    return f"""# Qualitative Case Study Note

## Instance

- benchmark instance: `{data['instance_name']}`

## Baseline draft

- title: {data['baseline_title']}
- pilot overall score: {data['baseline_score']:.2f}
- method summary: {data['baseline_method_text']}

## EIG final proposal

- title: {data['eig_title']}
- pilot overall score: {data['eig_score']:.2f}
- method summary: {data['eig_method_text']}
- evaluation summary: {data['eig_eval_text']}

## Repair flow

{data['action_path']}

## Repair target

{data['repair_text']}

## Suggested figure flow

1. Show the strongest text-only draft as a generic host-parasite interaction proposal.
2. Show the graph-side transition from evidence request to evidence attachment to repair.
3. Show that the EIG proposal shifts the scientific object toward parasite life cycles, temporal graph structure, and a sharper evaluation plan.
"""


def build_controller_behavior_exports(
    summary: ControllerBehaviorSummary,
) -> tuple[list[dict[str, int | str]], list[dict[str, float | str]]]:
    distribution_rows: list[dict[str, int | str]] = []
    for method in METHOD_ORDER:
        counts = summary.stop_counts.get(method)
        if not counts:
            continue
        for round_id in sorted(counts):
            distribution_rows.append({"method": method, "round": round_id, "count": counts[round_id]})
    point_rows = list(summary.quality_points)
    return distribution_rows, point_rows


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _add_text_box(ax, xy: tuple[float, float], wh: tuple[float, float], title: str, body: str, color: str) -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=color,
        facecolor=color + "18",
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(x + 0.02, y + h - 0.055, title, transform=ax.transAxes, weight="bold", color=color, va="top")
    ax.text(x + 0.02, y + h - 0.105, body, transform=ax.transAxes, va="top", linespacing=1.18)


def plot_qualitative_case(data: dict[str, Any], output_path: Path) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.55))
    for ax in axes:
        ax.set_axis_off()

    blue = "#1f77b4"
    orange = "#d65f00"
    green = "#2ca02c"
    gray = "#6b6b6b"

    axes[0].set_title("(a) Strong text-only draft", loc="left", weight="bold")
    baseline_body = (
        f"Title:\n{_shorten(str(data['baseline_title']), 46, 2)}\n\n"
        f"Method: {_shorten(data['baseline_method_text'], 54, 3)}\n\n"
        f"Pilot overall: {data['baseline_score']:.2f}"
    )
    _add_text_box(axes[0], (0.02, 0.08), (0.94, 0.80), "Generic host-parasite framing", baseline_body, gray)

    axes[1].set_title("(b) Graph-local repair path", loc="left", weight="bold")
    stats = f"{data['eig_rounds']} rounds, {data['eig_node_count']} nodes, {data['eig_edge_count']} edges, {data['eig_action_count']} actions"
    _add_text_box(axes[1], (0.05, 0.72), (0.90, 0.18), "Runtime trace", stats, blue)
    _add_text_box(
        axes[1],
        (0.05, 0.42),
        (0.90, 0.22),
        "Selected actions",
        _shorten(data["action_path"], 62, 4),
        orange,
    )
    _add_text_box(
        axes[1],
        (0.05, 0.08),
        (0.90, 0.25),
        "Repair target",
        _shorten(data["repair_text"], 62, 4),
        green,
    )

    axes[2].set_title("(c) Final EIG proposal", loc="left", weight="bold")
    eig_body = (
        f"Title:\n{_shorten(str(data['eig_title']), 46, 2)}\n\n"
        f"Method: {_shorten(data['eig_method_text'], 54, 3)}\n\n"
        f"Eval: {_shorten(data['eig_eval_text'], 54, 2)}\n\n"
        f"Pilot overall: {data['eig_score']:.2f}"
    )
    _add_text_box(axes[2], (0.02, 0.08), (0.94, 0.80), "Life-cycle-specific proposal", eig_body, green)

    fig.suptitle("Qualitative case study: graph repair sharpens a parasite ideation task", y=1.04, weight="bold")
    fig.tight_layout(w_pad=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_controller_behavior(summary: ControllerBehaviorSummary, output_path: Path) -> None:
    configure_plot_style()
    methods = [method for method in METHOD_ORDER if summary.method_counts.get(method, 0) > 0]
    rounds = sorted({round_id for counts in summary.stop_counts.values() for round_id in counts})
    colors = {
        2: "#8dd3c7",
        3: "#80b1d3",
        4: "#fdb462",
        5: "#b3b3b3",
    }
    method_colors = {
        "EIG (Ours)": "#1f77b4",
        "EIG-Text": "#9467bd",
        "EIG-NoCommit": "#ff7f0e",
        "EIG-Fixed": "#7f7f7f",
        "EIG-Random": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    ax = axes[0]
    bottom = [0] * len(methods)
    for round_id in rounds:
        values = [summary.stop_counts.get(method, Counter()).get(round_id, 0) for method in methods]
        ax.bar(methods, values, bottom=bottom, label=f"Round {round_id}", color=colors.get(round_id, "#cccccc"))
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_ylabel("Number of runs")
    ax.set_title("(a) Stop-round distribution", loc="left", weight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    offsets = {method: (idx - (len(methods) - 1) / 2) * 0.045 for idx, method in enumerate(methods)}
    for method in methods:
        pts = [point for point in summary.quality_points if point["method"] == method]
        xs = [float(point["round"]) + offsets[method] for point in pts]
        ys = [float(point["quality"]) for point in pts]
        ax.scatter(xs, ys, s=20, alpha=0.55, label=method, color=method_colors.get(method, "#333333"))
        if ys:
            ax.scatter(
                [sum(float(point["round"]) for point in pts) / len(pts) + offsets[method]],
                [sum(ys) / len(ys)],
                s=65,
                marker="D",
                edgecolor="black",
                linewidth=0.6,
                color=method_colors.get(method, "#333333"),
                zorder=3,
            )
    ax.set_xlabel("Realized rounds")
    ax.set_ylabel("Proposal quality")
    ax.set_title("(b) Quality vs. realized rounds", loc="left", weight="bold")
    ax.set_xticks(rounds)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    fig.tight_layout(w_pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build controller paper figures and human-evaluation packet.")
    parser.add_argument("--main-selected-rows", type=Path, default=DEFAULT_MAIN_ROWS)
    parser.add_argument("--ablation-selected-rows", type=Path, default=DEFAULT_ABLATION_ROWS)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--case-instance", default=CASE_INSTANCE)
    return parser


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    paper_dir = args.paper_dir
    figures_dir = paper_dir / "figures"
    generated_dir = paper_dir / "generated"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    case_data = load_case_study_data(args.main_selected_rows, root=ROOT, instance_name=args.case_instance)
    behavior = summarize_controller_behavior(read_csv_rows(args.ablation_selected_rows))

    fig2_path = figures_dir / "fig-qualitative-case-study.pdf"
    fig3_path = figures_dir / "fig-controller-behavior.pdf"
    packet_path = generated_dir / "human_eval_instruction_packet.md"
    case_note_path = generated_dir / "qualitative_case_study_parasites.md"
    round_distribution_path = generated_dir / "controller_behavior_round_distribution.csv"
    quality_points_path = generated_dir / "controller_behavior_quality_points.csv"

    plot_qualitative_case(case_data, fig2_path)
    plot_controller_behavior(behavior, fig3_path)
    packet_path.write_text(render_human_eval_instruction_packet(), encoding="utf-8")
    case_note_path.write_text(render_case_study_drawing_note(case_data), encoding="utf-8")
    distribution_rows, point_rows = build_controller_behavior_exports(behavior)
    _write_csv(round_distribution_path, distribution_rows, ["method", "round", "count"])
    _write_csv(quality_points_path, point_rows, ["method", "round", "quality"])

    print("Generated controller paper artifacts:")
    print(fig2_path)
    print(fig3_path)
    print(packet_path)
    print(case_note_path)
    print(round_distribution_path)
    print(quality_points_path)
    print("Controller behavior coverage:")
    for method in METHOD_ORDER:
        if method in behavior.method_counts:
            print(f"- {method}: n={behavior.method_counts[method]}")


if __name__ == "__main__":
    main()
