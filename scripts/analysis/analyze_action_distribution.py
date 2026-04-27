from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BATCH_ROOT = Path("outputs/quality_batches/pilot_main32_critic_graph_merged_20260420-111214")
DEFAULT_BASELINE = "ours-eig-critic-graph-twohead"
DEFAULT_BENCHMARK = "liveideabench"
DEFAULT_CATEGORY = "parasites"

EDIT_ACTION_ORDER = [
    "add_support_edge",
    "attach_evidence",
    "propose_repair",
    "request_evidence",
    "add_dependency_edge",
    "add_contradiction_edge",
    "mark_overlap",
    "skip",
]
COMMIT_ACTION_ORDER = ["continue", "commit"]
SIGNAL_ORDER = ["grounding", "completeness", "contradiction_load", "maturity"]
BASELINE_LABELS = {
    "ours-eig-critic-graph-twohead": "EIG (Ours)",
    "ours-eig-critic-text": "EIG-Text",
    "ours-eig-critic-no-commit": "EIG-NoCommit",
    "ours-eig-fixed-control": "EIG-Fixed",
    "ours-eig-random-control": "EIG-Random",
}
ACTION_LABELS = {
    "add_support_edge": "Support",
    "attach_evidence": "Evidence",
    "propose_repair": "Repair",
    "request_evidence": "Request",
    "add_dependency_edge": "Dependency",
    "add_contradiction_edge": "Contradiction",
    "mark_overlap": "Overlap",
    "freeze_branch": "Freeze",
    "skip": "Skip",
    "continue": "Continue",
    "commit": "Commit",
}
SIGNAL_LABELS = {
    "grounding": "Grounding",
    "completeness": "Completeness",
    "contradiction_load": "Contradiction load",
    "maturity": "Maturity",
}
SIGNAL_MARKERS = {
    "grounding": "o",
    "completeness": "s",
    "contradiction_load": "^",
    "maturity": "D",
}
STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "soft": {
        "signal_colors": {
            "grounding": "#4C78A8",
            "completeness": "#54A24B",
            "contradiction_load": "#E45756",
            "maturity": "#B279A2",
        },
        "action_panel_colors": {
            "edit": "#5F87B3",
            "commit": "#B9C6D3",
        },
        "edit_action_colors": {
            "add_support_edge": "#4C78A8",
            "attach_evidence": "#54A24B",
            "propose_repair": "#F58518",
            "request_evidence": "#B279A2",
            "add_dependency_edge": "#72B7B2",
            "add_contradiction_edge": "#E45756",
            "mark_overlap": "#9D7BB0",
            "skip": "#D1D7DF",
        },
        "commit_action_colors": {
            "continue": "#C2CEDB",
            "commit": "#5F87B3",
        },
        "font.size": 9.6,
        "axes.labelsize": 10.8,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 9.3,
        "ytick.labelsize": 9.3,
        "legend.fontsize": 8.6,
        "controller_figsize": (8.35, 3.14),
        "controller_top": 0.80,
        "signal_figsize": (4.55, 4.05),
        "signal_linewidth": 2.15,
        "signal_markersize": 5.4,
        "signal_markeredgewidth": 1.15,
        "signal_fill_alpha": 0.11,
        "signal_fontsize": 10.8,
        "signal_axes_labelsize": 12.8,
        "signal_tick_labelsize": 11.2,
        "signal_legend_fontsize": 10.2,
        "controller_bar_label_fontsize": 7.9,
        "controller_legend_fontsize": 8.0,
    },
    "contrast": {
        "signal_colors": {
            "grounding": "#0072B2",
            "completeness": "#009E73",
            "contradiction_load": "#D55E00",
            "maturity": "#CC79A7",
        },
        "action_panel_colors": {
            "edit": "#4C78A8",
            "commit": "#A7B8C9",
        },
        "edit_action_colors": {
            "add_support_edge": "#0072B2",
            "attach_evidence": "#009E73",
            "propose_repair": "#E69F00",
            "request_evidence": "#CC79A7",
            "add_dependency_edge": "#56B4E9",
            "add_contradiction_edge": "#D55E00",
            "mark_overlap": "#7E6AA2",
            "skip": "#C8CED6",
        },
        "commit_action_colors": {
            "continue": "#B7C6D8",
            "commit": "#4C78A8",
        },
        "font.size": 9.8,
        "axes.labelsize": 11.0,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.8,
        "ytick.labelsize": 9.8,
        "legend.fontsize": 8.9,
        "controller_figsize": (8.45, 3.28),
        "controller_top": 0.80,
        "signal_figsize": (4.65, 4.15),
        "signal_linewidth": 2.3,
        "signal_markersize": 5.9,
        "signal_markeredgewidth": 1.25,
        "signal_fill_alpha": 0.10,
        "signal_fontsize": 11.0,
        "signal_axes_labelsize": 13.0,
        "signal_tick_labelsize": 11.5,
        "signal_legend_fontsize": 10.5,
        "controller_bar_label_fontsize": 8.4,
        "controller_legend_fontsize": 8.5,
    },
}

CURRENT_STYLE_PRESET = "contrast"
CURRENT_STYLE = STYLE_PRESETS[CURRENT_STYLE_PRESET]
SIGNAL_COLORS = CURRENT_STYLE["signal_colors"]
ACTION_PANEL_COLORS = CURRENT_STYLE["action_panel_colors"]
EDIT_ACTION_COLORS = CURRENT_STYLE["edit_action_colors"]
COMMIT_ACTION_COLORS = CURRENT_STYLE["commit_action_colors"]


def _apply_style_preset(name: str) -> None:
    global CURRENT_STYLE_PRESET, CURRENT_STYLE, SIGNAL_COLORS, ACTION_PANEL_COLORS, EDIT_ACTION_COLORS, COMMIT_ACTION_COLORS

    if name not in STYLE_PRESETS:
        raise ValueError(f"Unknown style preset: {name}")
    CURRENT_STYLE_PRESET = name
    CURRENT_STYLE = STYLE_PRESETS[name]
    SIGNAL_COLORS = CURRENT_STYLE["signal_colors"]
    ACTION_PANEL_COLORS = CURRENT_STYLE["action_panel_colors"]
    EDIT_ACTION_COLORS = CURRENT_STYLE["edit_action_colors"]
    COMMIT_ACTION_COLORS = CURRENT_STYLE["commit_action_colors"]


class ActionDistribution:
    def __init__(
        self,
        *,
        benchmark: str,
        category: str,
        baseline_name: str,
        run_count: int,
        edit_counts: Counter[str],
        commit_counts: Counter[str],
    ) -> None:
        self.benchmark = benchmark
        self.category = category
        self.baseline_name = baseline_name
        self.run_count = run_count
        self.edit_counts = dict(edit_counts)
        self.commit_counts = dict(commit_counts)


class RoundActionDistribution:
    def __init__(
        self,
        *,
        benchmark: str,
        category: str,
        baseline_name: str,
        run_count: int,
        round_edit_counts: dict[str, dict[str, int]],
    ) -> None:
        self.benchmark = benchmark
        self.category = category
        self.baseline_name = baseline_name
        self.run_count = run_count
        self.round_edit_counts = {
            str(round_name): dict(Counter(counts))
            for round_name, counts in round_edit_counts.items()
        }


class RoundSignalDistribution:
    def __init__(
        self,
        *,
        benchmark: str,
        category: str,
        baseline_name: str,
        run_count: int,
        round_signal_means: dict[str, dict[str, float]],
        round_signal_stds: dict[str, dict[str, float]],
        round_signal_counts: dict[str, int],
    ) -> None:
        self.benchmark = benchmark
        self.category = category
        self.baseline_name = baseline_name
        self.run_count = run_count
        self.round_signal_means = {
            str(round_name): {str(signal): float(value) for signal, value in values.items()}
            for round_name, values in round_signal_means.items()
        }
        self.round_signal_stds = {
            str(round_name): {str(signal): float(value) for signal, value in values.items()}
            for round_name, values in round_signal_stds.items()
        }
        self.round_signal_counts = {
            str(round_name): int(count)
            for round_name, count in round_signal_counts.items()
        }


class StopRoundDistribution:
    def __init__(
        self,
        *,
        benchmark: str,
        category: str,
        baseline_name: str,
        run_count: int,
        stop_round_counts: Counter[str],
    ) -> None:
        self.benchmark = benchmark
        self.category = category
        self.baseline_name = baseline_name
        self.run_count = run_count
        self.stop_round_counts = dict(stop_round_counts)


def build_manual_distribution(
    *,
    benchmark: str,
    category: str,
    baseline_name: str,
    run_count: int,
    edit_counts: dict[str, int],
    commit_counts: dict[str, int],
) -> ActionDistribution:
    return ActionDistribution(
        benchmark=benchmark,
        category=category,
        baseline_name=baseline_name,
        run_count=run_count,
        edit_counts=Counter(edit_counts),
        commit_counts=Counter(commit_counts),
    )


def build_manual_round_distribution(
    *,
    benchmark: str,
    category: str,
    baseline_name: str,
    run_count: int,
    round_edit_counts: dict[str, dict[str, int]],
) -> RoundActionDistribution:
    return RoundActionDistribution(
        benchmark=benchmark,
        category=category,
        baseline_name=baseline_name,
        run_count=run_count,
        round_edit_counts=round_edit_counts,
    )


def _safe_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    if os.name == "nt":
        return "\\\\?\\" + str(resolved)
    return str(resolved)


def _path_exists(path: Path) -> bool:
    return os.path.exists(_safe_path(path))


def _strip_windows_extended_prefix(raw_path: str) -> str:
    if os.name != "nt":
        return raw_path
    if raw_path.startswith("\\\\?\\UNC\\"):
        return "\\" + raw_path[7:]
    if raw_path.startswith("\\\\?\\"):
        return raw_path[4:]
    return raw_path


def _add_discovered_graph_path(path: Path, *, seen: set[Path], paths: list[Path]) -> None:
    resolved = path.resolve(strict=False)
    if resolved in seen:
        return
    seen.add(resolved)
    paths.append(path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(_safe_path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _read_selected_rows(batch_root: Path) -> list[dict[str, str]]:
    selected_paths = [batch_root / "selected_rows.csv"]
    if not selected_paths[0].exists():
        selected_paths = sorted(batch_root.glob("*/selected_rows.csv"))
    if not selected_paths:
        return []

    rows: list[dict[str, str]] = []
    for path in selected_paths:
        with open(_safe_path(path), "r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _discover_raw_graph_paths(batch_root: Path) -> list[Path]:
    patterns = [
        "shards/*/runs/*/graph.json",
        "shards/*/runs/*/*/graph.json",
        "*/shards/*/runs/*/graph.json",
        "*/shards/*/runs/*/*/graph.json",
        "runs/*/graph.json",
        "runs/*/*/graph.json",
        "*/runs/*/graph.json",
        "*/runs/*/*/graph.json",
    ]
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        for path in batch_root.glob(pattern):
            _add_discovered_graph_path(path, seen=seen, paths=paths)

    # `Path.glob(...)` can miss valid run paths on Windows when batch directories
    # get long or include several nested rerun folders. Supplement it with
    # `os.walk(...)` over the extended-length path so completed runs are still found.
    for dirpath, _, filenames in os.walk(_safe_path(batch_root)):
        if "graph.json" not in filenames:
            continue
        graph_path = Path(_strip_windows_extended_prefix(dirpath)) / "graph.json"
        if "runs" not in {part.lower() for part in graph_path.parts}:
            continue
        _add_discovered_graph_path(graph_path, seen=seen, paths=paths)
    return paths


def _raw_row_from_graph_path(graph_path: Path) -> dict[str, str] | None:
    payload = _read_json(graph_path)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    benchmark = str(metadata.get("benchmark", "")).strip()
    instance_name = str(metadata.get("instance_name", "")).strip()
    baseline_name = str(metadata.get("baseline_name", "")).strip()
    if not benchmark or not instance_name or not baseline_name:
        return None
    return {
        "benchmark": benchmark,
        "instance_name": instance_name,
        "baseline_name": baseline_name,
        "run_dir": str(graph_path.parent),
    }


def _read_run_rows(batch_root: Path) -> list[dict[str, str]]:
    rows = _read_selected_rows(batch_root)
    if rows:
        return rows
    raw_rows: list[dict[str, str]] = []
    for graph_path in _discover_raw_graph_paths(batch_root):
        row = _raw_row_from_graph_path(graph_path)
        if row is not None:
            raw_rows.append(row)
    if not raw_rows:
        raise FileNotFoundError(
            f"No selected_rows.csv or raw graph.json runs found under {batch_root}."
        )
    return raw_rows


def _normalize_label(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value.strip().lower())


def _liveideabench_category(instance_name: str) -> str:
    match = re.match(r"^liveideabench-(.+)-\d+$", instance_name.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _row_category(row: dict[str, str]) -> str:
    benchmark = row.get("benchmark", "").strip()
    instance_name = row.get("instance_name", "").strip()
    if benchmark.lower() == "liveideabench":
        return _liveideabench_category(instance_name) or "liveideabench"
    return benchmark or "unknown"


def _category_matches(row: dict[str, str], category: str) -> bool:
    if not category or _normalize_label(category) == "all":
        return True
    return _normalize_label(_row_category(row)) == _normalize_label(category)


def _benchmark_matches(row: dict[str, str], benchmark: str) -> bool:
    if not benchmark or _normalize_label(benchmark) == "all":
        return True
    return row.get("benchmark", "").strip().lower() == benchmark.lower()


def _resolve_run_dir(batch_root: Path, run_dir_value: str) -> Path:
    candidate = Path(str(run_dir_value))
    if candidate.is_absolute() and _path_exists(candidate):
        return candidate
    candidates = [
        candidate,
        Path.cwd() / candidate,
        batch_root / candidate,
        batch_root.parent / candidate,
    ]
    for path in candidates:
        if _path_exists(path):
            return path.resolve()
    return candidate.resolve(strict=False)


def _selected_edit_actions(graph: dict[str, Any]) -> list[str]:
    metadata = graph.get("metadata", {})
    if isinstance(metadata, dict):
        edit_rows = metadata.get("parallel_edit_rows", [])
        if isinstance(edit_rows, list) and edit_rows:
            actions: list[str] = []
            for row in edit_rows:
                if not isinstance(row, dict):
                    continue
                action = str(row.get("selected_action_kind", "")).strip()
                if action:
                    actions.append(action)
            if actions:
                return actions
        controller_rows = metadata.get("runtime_controller_log", [])
        if isinstance(controller_rows, list) and controller_rows:
            actions = []
            for row in controller_rows:
                if not isinstance(row, dict):
                    continue
                action = str(row.get("selected_kind", "")).strip()
                if not action:
                    selected_candidate = row.get("selected_candidate")
                    if isinstance(selected_candidate, dict):
                        action = str(selected_candidate.get("kind", "")).strip()
                if action and action != "commit":
                    actions.append(action)
            if actions:
                return actions

    graph_actions = graph.get("actions", [])
    if isinstance(graph_actions, list):
        return [
            str(action.get("kind", "")).strip()
            for action in graph_actions
            if isinstance(action, dict) and str(action.get("kind", "")).strip()
        ]
    return []


def _round_action_from_controller_row(row: dict[str, Any]) -> tuple[str, str] | None:
    round_name = str(row.get("round", row.get("round_name", ""))).strip()
    action = str(row.get("selected_kind", "")).strip()
    if not action:
        selected_candidate = row.get("selected_candidate")
        if isinstance(selected_candidate, dict):
            action = str(selected_candidate.get("kind", "")).strip()
    if not round_name or not action or action == "commit":
        return None
    return round_name, action


def _selected_round_edit_actions(graph: dict[str, Any]) -> list[tuple[str, str]]:
    metadata = graph.get("metadata", {})
    if isinstance(metadata, dict):
        edit_rows = metadata.get("parallel_edit_rows", [])
        if isinstance(edit_rows, list) and edit_rows:
            actions: list[tuple[str, str]] = []
            for row in edit_rows:
                if not isinstance(row, dict):
                    continue
                round_name = str(row.get("round_name", row.get("round", ""))).strip()
                action = str(row.get("selected_action_kind", "")).strip()
                if round_name and action:
                    actions.append((round_name, action))
            if actions:
                return actions
        controller_rows = metadata.get("runtime_controller_log", [])
        if isinstance(controller_rows, list) and controller_rows:
            actions = []
            for row in controller_rows:
                if not isinstance(row, dict):
                    continue
                item = _round_action_from_controller_row(row)
                if item is not None:
                    actions.append(item)
            if actions:
                return actions

    graph_actions = graph.get("actions", [])
    if isinstance(graph_actions, list):
        return [
            (
                str(action.get("round_name", action.get("round", ""))).strip(),
                str(action.get("kind", "")).strip(),
            )
            for action in graph_actions
            if isinstance(action, dict)
            and str(action.get("round_name", action.get("round", ""))).strip()
            and str(action.get("kind", "")).strip()
        ]
    return []


def _selected_commit_actions(graph: dict[str, Any]) -> list[str]:
    metadata = graph.get("metadata", {})
    rows: list[Any] = []
    if isinstance(metadata, dict):
        post_round_rows = metadata.get("post_round_commit_rows", [])
        if isinstance(post_round_rows, list) and post_round_rows:
            rows = post_round_rows
        elif isinstance(metadata.get("parallel_round_traces"), list):
            rows = [trace.get("post_round_commit", {}) for trace in metadata["parallel_round_traces"] if isinstance(trace, dict)]

    actions: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label: object | None = None
        supervision = row.get("commit_supervision")
        if isinstance(supervision, dict):
            label = supervision.get("label")
        if label is None:
            label = row.get("should_commit")
        if isinstance(label, str):
            normalized = label.strip().lower()
            if normalized in {"1", "true", "commit"}:
                actions.append("commit")
            elif normalized in {"0", "false", "continue"}:
                actions.append("continue")
        elif label is not None:
            actions.append("commit" if bool(label) else "continue")
    return actions


def _ordered_items(
    counts: dict[str, int],
    preferred_order: Iterable[str],
    *,
    include_zero_preferred: bool = False,
) -> list[tuple[str, int]]:
    ordered: list[tuple[str, int]] = []
    seen: set[str] = set()
    for action in preferred_order:
        if include_zero_preferred:
            ordered.append((action, counts.get(action, 0)))
            seen.add(action)
        elif action in counts:
            ordered.append((action, counts[action]))
            seen.add(action)
    for action in sorted(action for action in counts if action not in seen):
        ordered.append((action, counts[action]))
    return ordered


def _benchmark_display_name(benchmark: str) -> str:
    if benchmark.strip().lower() == "all":
        return "All benchmarks"
    if benchmark.strip().lower() == "liveideabench":
        return "LiveIdeaBench"
    return benchmark


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")

    matplotlib.rcParams.update(
        {
            "font.size": CURRENT_STYLE["font.size"],
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": CURRENT_STYLE["axes.labelsize"],
            "axes.titlesize": CURRENT_STYLE["axes.titlesize"],
            "xtick.labelsize": CURRENT_STYLE["xtick.labelsize"],
            "ytick.labelsize": CURRENT_STYLE["ytick.labelsize"],
            "legend.fontsize": CURRENT_STYLE["legend.fontsize"],
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "mathtext.fontset": "stix",
        }
    )


def _action_items(counts: dict[str, int], preferred_order: Iterable[str]) -> list[tuple[str, int]]:
    return _ordered_items(counts, preferred_order, include_zero_preferred=True)


def _active_action_items(counts: dict[str, int], preferred_order: Iterable[str]) -> list[tuple[str, int]]:
    return _ordered_items(
        {action: count for action, count in counts.items() if count > 0},
        preferred_order,
        include_zero_preferred=False,
    )


def _colors_for_actions(
    actions: list[str],
    *,
    fallback_color: str,
    color_map: dict[str, str] | None = None,
) -> list[str]:
    if color_map is None:
        return [fallback_color for _ in actions]
    return [color_map.get(action, fallback_color) for action in actions]


def _style_axis(ax: Any) -> None:
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=3.0, width=0.7, pad=2)


def _display_round_label(round_name: str) -> str:
    match = re.fullmatch(r"Round(\d+)", str(round_name))
    if match:
        return match.group(1)
    return str(round_name)


def _draw_action_panel(
    ax: Any,
    items: list[tuple[str, int]],
    *,
    label: str,
    color: str,
    color_map: dict[str, str] | None = None,
) -> None:
    actions = [ACTION_LABELS.get(action, action.replace("_", "\n")) for action, _ in items]
    counts = [count for _, count in items]
    action_keys = [action for action, _ in items]
    colors = _colors_for_actions(action_keys, fallback_color=color, color_map=color_map)
    bars = ax.bar(
        actions,
        counts,
        color=colors,
        alpha=0.96,
        width=0.62,
        edgecolor="#FFFFFF",
        linewidth=0.9,
    )
    ax.set_title(label, pad=5, fontweight="semibold")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle=":", linewidth=0.55, color="#9A9A9A", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(y=0.16)
    _style_axis(ax)
    for bar, count in zip(bars, counts):
        if count == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )


def _draw_fraction_action_panel(
    ax: Any,
    items: list[tuple[str, int]],
    *,
    title: str,
    color: str,
    color_map: dict[str, str] | None = None,
) -> None:
    actions = [ACTION_LABELS.get(action, action.replace("_", "\n")) for action, _ in items]
    total = sum(count for _, count in items)
    fractions = [(count / total) if total else 0.0 for _, count in items]
    action_keys = [action for action, _ in items]
    colors = _colors_for_actions(action_keys, fallback_color=color, color_map=color_map)
    bars = ax.bar(
        actions,
        fractions,
        color=colors,
        alpha=0.95,
        width=0.62,
        edgecolor="#FFFFFF",
        linewidth=0.9,
    )
    ax.set_title(title, pad=5, fontweight="semibold")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, max(0.05, min(1.0, max(fractions or [0.0]) * 1.18)))
    ax.grid(axis="y", linestyle=":", linewidth=0.55, color="#9A9A9A", alpha=0.35)
    ax.set_axisbelow(True)
    _style_axis(ax)
    for bar, fraction in zip(bars, fractions):
        if fraction <= 0.0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{100.0 * fraction:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )


def _candidate_categories(
    rows: list[dict[str, str]],
    *,
    baseline_name: str,
    benchmark: str,
) -> list[str]:
    categories: Counter[str] = Counter()
    for row in rows:
        if row.get("baseline_name", "").strip() != baseline_name:
            continue
        if not _benchmark_matches(row, benchmark):
            continue
        category = _row_category(row)
        if not category:
            continue
        categories[category] += 1
    return [
        category
        for category, _ in sorted(categories.items(), key=lambda item: (-item[1], item[0]))
    ]


def _pick_topic_category(
    rows: list[dict[str, str]],
    *,
    baseline_name: str,
    benchmark: str,
    topic_category: str,
) -> str:
    if topic_category and topic_category.lower() not in {"auto", "largest"}:
        return topic_category
    categories = _candidate_categories(rows, baseline_name=baseline_name, benchmark=benchmark)
    if not categories:
        raise ValueError(
            f"No topic categories found for baseline={baseline_name} and benchmark={benchmark}."
        )
    return categories[0]


def collect_action_distribution(
    batch_root: Path,
    *,
    baseline_name: str = DEFAULT_BASELINE,
    benchmark: str = DEFAULT_BENCHMARK,
    category: str = DEFAULT_CATEGORY,
) -> ActionDistribution:
    rows = _read_run_rows(batch_root)
    edit_counts: Counter[str] = Counter()
    commit_counts: Counter[str] = Counter()
    run_count = 0

    for row in rows:
        if row.get("baseline_name", "").strip() != baseline_name:
            continue
        if not _benchmark_matches(row, benchmark):
            continue
        if not _category_matches(row, category):
            continue
        run_dir = _resolve_run_dir(batch_root, row.get("run_dir", ""))
        graph_path = run_dir / "graph.json"
        if not _path_exists(graph_path):
            continue
        graph = _read_json(graph_path)
        edit_counts.update(_selected_edit_actions(graph))
        commit_counts.update(_selected_commit_actions(graph))
        run_count += 1

    if run_count == 0:
        raise ValueError(
            f"No matching graph.json files found for baseline={baseline_name}, "
            f"benchmark={benchmark}, category={category} under {batch_root}."
        )

    return ActionDistribution(
        benchmark=benchmark,
        category=category,
        baseline_name=baseline_name,
        run_count=run_count,
        edit_counts=edit_counts,
        commit_counts=commit_counts,
    )


def collect_round_action_distribution(
    batch_root: Path,
    *,
    baseline_name: str = DEFAULT_BASELINE,
    benchmark: str = DEFAULT_BENCHMARK,
    category: str = DEFAULT_CATEGORY,
) -> RoundActionDistribution:
    rows = _read_run_rows(batch_root)
    round_counts: dict[str, Counter[str]] = {}
    run_count = 0

    for row in rows:
        if row.get("baseline_name", "").strip() != baseline_name:
            continue
        if not _benchmark_matches(row, benchmark):
            continue
        if not _category_matches(row, category):
            continue
        run_dir = _resolve_run_dir(batch_root, row.get("run_dir", ""))
        graph_path = run_dir / "graph.json"
        if not _path_exists(graph_path):
            continue
        graph = _read_json(graph_path)
        for round_name, action in _selected_round_edit_actions(graph):
            round_counts.setdefault(round_name, Counter()).update([action])
        run_count += 1

    if run_count == 0:
        raise ValueError(
            f"No matching graph.json files found for baseline={baseline_name}, "
            f"benchmark={benchmark}, category={category} under {batch_root}."
        )

    return RoundActionDistribution(
        benchmark=benchmark,
        category=category,
        baseline_name=baseline_name,
        run_count=run_count,
        round_edit_counts={round_name: dict(counts) for round_name, counts in round_counts.items()},
    )


def _round_name_from_state_id(state_id: str) -> str:
    match = re.search(r"::(Round\d+)::", str(state_id))
    if match:
        return match.group(1)
    return ""


def _selected_round_signals(graph: dict[str, Any]) -> list[tuple[str, dict[str, float]]]:
    metadata = graph.get("metadata", {})
    if isinstance(metadata, dict):
        post_round_rows = metadata.get("post_round_commit_rows", [])
        if isinstance(post_round_rows, list) and post_round_rows:
            signals_by_round: list[tuple[str, dict[str, float]]] = []
            for row in post_round_rows:
                if not isinstance(row, dict):
                    continue
                round_name = str(row.get("round_name", row.get("round", ""))).strip()
                if not round_name:
                    round_name = _round_name_from_state_id(str(row.get("state_id", "")).strip())
                raw_signals = row.get("graph_signals", {})
                if not round_name or not isinstance(raw_signals, dict):
                    continue
                parsed_signals: dict[str, float] = {}
                for signal_name in SIGNAL_ORDER:
                    value = raw_signals.get(signal_name)
                    try:
                        parsed_signals[signal_name] = float(value)
                    except (TypeError, ValueError):
                        continue
                if parsed_signals:
                    signals_by_round.append((round_name, parsed_signals))
            if signals_by_round:
                return signals_by_round

        traces = metadata.get("parallel_round_traces", [])
        if isinstance(traces, list) and traces:
            signals_by_round = []
            for trace in traces:
                if not isinstance(trace, dict):
                    continue
                round_name = str(trace.get("round", "")).strip()
                post_round_commit = trace.get("post_round_commit", {})
                if not round_name or not isinstance(post_round_commit, dict):
                    continue
                raw_signals = post_round_commit.get("graph_signals", {})
                if not isinstance(raw_signals, dict):
                    continue
                parsed_signals = {}
                for signal_name in SIGNAL_ORDER:
                    value = raw_signals.get(signal_name)
                    try:
                        parsed_signals[signal_name] = float(value)
                    except (TypeError, ValueError):
                        continue
                if parsed_signals:
                    signals_by_round.append((round_name, parsed_signals))
            if signals_by_round:
                return signals_by_round

    return []


def collect_round_signal_distribution(
    batch_root: Path,
    *,
    baseline_name: str = DEFAULT_BASELINE,
    benchmark: str = DEFAULT_BENCHMARK,
    category: str = DEFAULT_CATEGORY,
) -> RoundSignalDistribution:
    rows = _read_run_rows(batch_root)
    round_signal_values: dict[str, dict[str, list[float]]] = {}
    round_signal_counts: Counter[str] = Counter()
    run_count = 0

    for row in rows:
        if row.get("baseline_name", "").strip() != baseline_name:
            continue
        if not _benchmark_matches(row, benchmark):
            continue
        if not _category_matches(row, category):
            continue
        run_dir = _resolve_run_dir(batch_root, row.get("run_dir", ""))
        graph_path = run_dir / "graph.json"
        if not _path_exists(graph_path):
            continue
        graph = _read_json(graph_path)
        seen_rounds: set[str] = set()
        for round_name, signal_values in _selected_round_signals(graph):
            bucket = round_signal_values.setdefault(round_name, {})
            for signal_name in SIGNAL_ORDER:
                if signal_name not in signal_values:
                    continue
                bucket.setdefault(signal_name, []).append(float(signal_values[signal_name]))
            if round_name not in seen_rounds:
                round_signal_counts.update([round_name])
                seen_rounds.add(round_name)
        run_count += 1

    if run_count == 0:
        raise ValueError(
            f"No matching graph.json files found for baseline={baseline_name}, "
            f"benchmark={benchmark}, category={category} under {batch_root}."
        )

    round_signal_means: dict[str, dict[str, float]] = {}
    round_signal_stds: dict[str, dict[str, float]] = {}
    for round_name, signal_map in round_signal_values.items():
        mean_map: dict[str, float] = {}
        std_map: dict[str, float] = {}
        for signal_name in SIGNAL_ORDER:
            values = signal_map.get(signal_name, [])
            if not values:
                continue
            mean_map[signal_name] = round(float(statistics.mean(values)), 6)
            std_map[signal_name] = round(float(statistics.pstdev(values)), 6)
        if mean_map:
            round_signal_means[round_name] = mean_map
            round_signal_stds[round_name] = std_map

    return RoundSignalDistribution(
        benchmark=benchmark,
        category=category,
        baseline_name=baseline_name,
        run_count=run_count,
        round_signal_means=round_signal_means,
        round_signal_stds=round_signal_stds,
        round_signal_counts=dict(round_signal_counts),
    )


def collect_comparison_distributions(
    batch_root: Path,
    *,
    baseline_name: str = DEFAULT_BASELINE,
    benchmark: str = DEFAULT_BENCHMARK,
    topic_category: str = "auto",
) -> tuple[ActionDistribution, ActionDistribution]:
    rows = _read_run_rows(batch_root)
    chosen_topic = _pick_topic_category(
        rows,
        baseline_name=baseline_name,
        benchmark=benchmark,
        topic_category=topic_category,
    )
    aggregate = collect_action_distribution(
        batch_root,
        baseline_name=baseline_name,
        benchmark=benchmark,
        category="all",
    )
    topic = collect_action_distribution(
        batch_root,
        baseline_name=baseline_name,
        benchmark=benchmark,
        category=chosen_topic,
    )
    return aggregate, topic


def _rows_for_csv(distribution: ActionDistribution) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    total_edit = sum(distribution.edit_counts.values())
    total_commit = sum(distribution.commit_counts.values())
    for action, count in _action_items(distribution.edit_counts, EDIT_ACTION_ORDER):
        rows.append(
            {
                "action_type": "edit",
                "action": action,
                "count": str(count),
                "fraction": f"{count / total_edit:.4f}" if total_edit else "0.0000",
            }
        )
    for action, count in _action_items(distribution.commit_counts, COMMIT_ACTION_ORDER):
        rows.append(
            {
                "action_type": "commit",
                "action": action,
                "count": str(count),
                "fraction": f"{count / total_commit:.4f}" if total_commit else "0.0000",
            }
        )
    return rows


def write_distribution_csv(distribution: ActionDistribution, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(_safe_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action_type", "action", "count", "fraction"])
        writer.writeheader()
        writer.writerows(_rows_for_csv(distribution))


def write_summary_json(distribution: ActionDistribution, path: Path) -> None:
    payload = {
        "benchmark": distribution.benchmark,
        "category": distribution.category,
        "baseline_name": distribution.baseline_name,
        "run_count": distribution.run_count,
        "edit_total": sum(distribution.edit_counts.values()),
        "commit_total": sum(distribution.commit_counts.values()),
        "edit_counts": distribution.edit_counts,
        "commit_counts": distribution.commit_counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_action_distribution_from_csv(
    path: Path,
    *,
    benchmark: str = "csv",
    category: str = "csv",
    baseline_name: str = "csv",
    run_count: int = 0,
) -> ActionDistribution:
    edit_counts: Counter[str] = Counter()
    commit_counts: Counter[str] = Counter()
    with open(_safe_path(path), "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            action_type = str(row.get("action_type", "")).strip().lower()
            action = str(row.get("action", "")).strip()
            if not action:
                continue
            count = int(str(row.get("count", "0")).strip() or "0")
            if action_type == "edit":
                edit_counts[action] += count
            elif action_type == "commit":
                commit_counts[action] += count
    return ActionDistribution(
        benchmark=benchmark,
        category=category,
        baseline_name=baseline_name,
        run_count=run_count,
        edit_counts=edit_counts,
        commit_counts=commit_counts,
    )


def _round_sort_key(round_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(round_name))
    if match:
        return (int(match.group(1)), str(round_name))
    return (999, str(round_name))


def _ordered_round_names(round_counts: dict[str, dict[str, int]]) -> list[str]:
    return sorted(round_counts, key=_round_sort_key)


def _rows_for_round_csv(distribution: RoundActionDistribution) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for round_name in _ordered_round_names(distribution.round_edit_counts):
        counts = distribution.round_edit_counts.get(round_name, {})
        total = sum(counts.values())
        for action, count in _action_items(counts, EDIT_ACTION_ORDER):
            rows.append(
                {
                    "round": round_name,
                    "action": action,
                    "count": str(count),
                    "fraction": f"{count / total:.4f}" if total else "0.0000",
                }
            )
    return rows


def write_round_distribution_csv(distribution: RoundActionDistribution, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(_safe_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "action", "count", "fraction"])
        writer.writeheader()
        writer.writerows(_rows_for_round_csv(distribution))


def load_round_action_distribution_from_csv(
    path: Path,
    *,
    benchmark: str = "csv",
    category: str = "csv",
    baseline_name: str = "csv",
    run_count: int = 0,
) -> RoundActionDistribution:
    round_edit_counts: dict[str, Counter[str]] = {}
    with open(_safe_path(path), "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            round_name = str(row.get("round", "")).strip()
            action = str(row.get("action", "")).strip()
            if not round_name or not action:
                continue
            count = int(str(row.get("count", "0")).strip() or "0")
            round_edit_counts.setdefault(round_name, Counter())[action] += count
    return RoundActionDistribution(
        benchmark=benchmark,
        category=category,
        baseline_name=baseline_name,
        run_count=run_count,
        round_edit_counts={round_name: dict(counts) for round_name, counts in round_edit_counts.items()},
    )


def _rows_for_round_signal_csv(distribution: RoundSignalDistribution) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for round_name in _ordered_round_names(distribution.round_signal_means):
        mean_map = distribution.round_signal_means.get(round_name, {})
        std_map = distribution.round_signal_stds.get(round_name, {})
        run_count = distribution.round_signal_counts.get(round_name, 0)
        for signal_name in SIGNAL_ORDER:
            if signal_name not in mean_map:
                continue
            rows.append(
                {
                    "round": round_name,
                    "signal": signal_name,
                    "mean": f"{float(mean_map[signal_name]):.4f}",
                    "std": f"{float(std_map.get(signal_name, 0.0)):.4f}",
                    "run_count": str(run_count),
                }
            )
    return rows


def _rows_for_stop_round_csv(distribution: StopRoundDistribution) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    total = sum(distribution.stop_round_counts.values())
    for round_name in _ordered_round_names(distribution.stop_round_counts):
        count = int(distribution.stop_round_counts.get(round_name, 0))
        rows.append(
            {
                "round": round_name,
                "count": str(count),
                "fraction": f"{count / total:.4f}" if total else "0.0000",
            }
        )
    return rows


def write_round_signal_csv(distribution: RoundSignalDistribution, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(_safe_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["round", "signal", "mean", "std", "run_count"],
        )
        writer.writeheader()
        writer.writerows(_rows_for_round_signal_csv(distribution))


def write_stop_round_csv(distribution: StopRoundDistribution, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(_safe_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "count", "fraction"])
        writer.writeheader()
        writer.writerows(_rows_for_stop_round_csv(distribution))


def load_round_signal_distribution_from_csv(
    path: Path,
    *,
    benchmark: str = "csv",
    category: str = "csv",
    baseline_name: str = "csv",
    run_count: int = 0,
) -> RoundSignalDistribution:
    round_signal_means: dict[str, dict[str, float]] = {}
    round_signal_stds: dict[str, dict[str, float]] = {}
    round_signal_counts: dict[str, int] = {}
    with open(_safe_path(path), "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            round_name = str(row.get("round", "")).strip()
            signal_name = str(row.get("signal", "")).strip()
            if not round_name or not signal_name:
                continue
            round_signal_means.setdefault(round_name, {})[signal_name] = float(
                str(row.get("mean", "0.0")).strip() or "0.0"
            )
            round_signal_stds.setdefault(round_name, {})[signal_name] = float(
                str(row.get("std", "0.0")).strip() or "0.0"
            )
            round_signal_counts[round_name] = int(str(row.get("run_count", "0")).strip() or "0")
    return RoundSignalDistribution(
        benchmark=benchmark,
        category=category,
        baseline_name=baseline_name,
        run_count=run_count,
        round_signal_means=round_signal_means,
        round_signal_stds=round_signal_stds,
        round_signal_counts=round_signal_counts,
    )


def load_stop_round_distribution_from_csv(
    path: Path,
    *,
    benchmark: str = "csv",
    category: str = "csv",
    baseline_name: str = "csv",
    run_count: int = 0,
) -> StopRoundDistribution:
    stop_round_counts: Counter[str] = Counter()
    with open(_safe_path(path), "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            round_name = str(row.get("round", "")).strip()
            if not round_name:
                continue
            stop_round_counts[round_name] += int(str(row.get("count", "0")).strip() or "0")
    return StopRoundDistribution(
        benchmark=benchmark,
        category=category,
        baseline_name=baseline_name,
        run_count=run_count,
        stop_round_counts=stop_round_counts,
    )


def create_action_distribution_figure(distribution: ActionDistribution) -> tuple[Any, Any]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.2, 3.05),
        gridspec_kw={"width_ratios": [4.6, 1.2]},
        constrained_layout=True,
    )
    _draw_action_panel(
        axes[0],
        _active_action_items(distribution.edit_counts, EDIT_ACTION_ORDER),
        label="Edit actions",
        color=ACTION_PANEL_COLORS["edit"],
        color_map=EDIT_ACTION_COLORS,
    )
    _draw_action_panel(
        axes[1],
        _active_action_items(distribution.commit_counts, COMMIT_ACTION_ORDER),
        label="Commit decisions",
        color=ACTION_PANEL_COLORS["commit"],
        color_map=COMMIT_ACTION_COLORS,
    )
    axes[0].tick_params(axis="x", rotation=0)
    axes[1].tick_params(axis="x", rotation=0)
    return fig, axes


def create_overall_calibration_comparison_figure(
    left: ActionDistribution,
    right: ActionDistribution,
    *,
    left_label: str,
    right_label: str,
) -> tuple[Any, Any]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.2),
        sharey=True,
        constrained_layout=True,
    )
    _draw_fraction_action_panel(
        axes[0],
        _action_items(left.edit_counts, EDIT_ACTION_ORDER),
        title=left_label,
        color="#577590",
        color_map=EDIT_ACTION_COLORS,
    )
    _draw_fraction_action_panel(
        axes[1],
        _action_items(right.edit_counts, EDIT_ACTION_ORDER),
        title=right_label,
        color="#C76F32",
        color_map=EDIT_ACTION_COLORS,
    )
    for ax, distribution in zip(axes, (left, right), strict=True):
        total = sum(distribution.edit_counts.values())
        ax.set_xlabel(f"Edit actions\n{distribution.run_count} episodes, {total} decisions")
        ax.tick_params(axis="x", rotation=0)
    return fig, axes


def plot_overall_calibration_comparison_figure(
    left: ActionDistribution,
    right: ActionDistribution,
    *,
    left_label: str,
    right_label: str,
    path: Path,
) -> None:
    fig, _ = create_overall_calibration_comparison_figure(
        left,
        right,
        left_label=left_label,
        right_label=right_label,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_safe_path(path))
    if path.suffix.lower() == ".pdf":
        fig.savefig(_safe_path(path.with_suffix(".png")))
    import matplotlib.pyplot as plt

    plt.close(fig)


def _draw_round_stacked_panel(
    ax: Any,
    distribution: RoundActionDistribution,
    *,
    title: str,
) -> None:
    rounds = _ordered_round_names(distribution.round_edit_counts)
    x = list(range(len(rounds)))
    bottom = [0.0 for _ in rounds]
    for index, action in enumerate(EDIT_ACTION_ORDER):
        fractions: list[float] = []
        for round_name in rounds:
            counts = distribution.round_edit_counts.get(round_name, {})
            total = sum(counts.values())
            fractions.append((counts.get(action, 0) / total) if total else 0.0)
        ax.bar(
            x,
            fractions,
            bottom=bottom,
            width=0.68,
            color=EDIT_ACTION_COLORS.get(action, ACTION_PANEL_COLORS["edit"]),
            label=ACTION_LABELS.get(action, action.replace("_", " ")).replace("\n", " "),
            edgecolor="#FFFFFF",
            linewidth=0.6,
        )
        bottom = [base + value for base, value in zip(bottom, fractions, strict=True)]
    ax.set_title(title, pad=5, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(rounds)
    ax.set_ylabel("Fraction within round")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.55, color="#9A9A9A", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Round\n{distribution.run_count} episodes")
    _style_axis(ax)


def create_round_calibration_comparison_figure(
    left: RoundActionDistribution,
    right: RoundActionDistribution,
    *,
    left_label: str,
    right_label: str,
) -> tuple[Any, Any]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.6, 3.25),
        sharey=True,
        constrained_layout=True,
    )
    _draw_round_stacked_panel(axes[0], left, title=left_label)
    _draw_round_stacked_panel(axes[1], right, title=right_label)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=7.2,
    )
    return fig, axes


def plot_round_calibration_comparison_figure(
    left: RoundActionDistribution,
    right: RoundActionDistribution,
    *,
    left_label: str,
    right_label: str,
    path: Path,
) -> None:
    fig, _ = create_round_calibration_comparison_figure(
        left,
        right,
        left_label=left_label,
        right_label=right_label,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_safe_path(path))
    if path.suffix.lower() == ".pdf":
        fig.savefig(_safe_path(path.with_suffix(".png")))
    import matplotlib.pyplot as plt

    plt.close(fig)


def create_round_action_count_figure(distribution: RoundActionDistribution) -> tuple[Any, Any]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rounds = _ordered_round_names(distribution.round_edit_counts)
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 3.4), constrained_layout=True)
    center_positions = list(range(len(rounds)))

    active_actions_global = [
        action
        for action in EDIT_ACTION_ORDER
        if any(distribution.round_edit_counts.get(round_name, {}).get(action, 0) > 0 for round_name in rounds)
    ]

    for center, round_name in zip(center_positions, rounds, strict=True):
        counts = distribution.round_edit_counts.get(round_name, {})
        active_items = [
            (action, int(counts.get(action, 0)))
            for action in EDIT_ACTION_ORDER
            if int(counts.get(action, 0)) > 0
        ]
        if not active_items:
            continue
        round_span = 0.76
        bar_width = min(0.22, round_span / max(len(active_items), 1))
        offsets = [
            (index - (len(active_items) - 1) / 2.0) * bar_width
            for index in range(len(active_items))
        ]
        for offset, (action, count) in zip(offsets, active_items, strict=True):
            ax.bar(
                center + offset,
                count,
                width=bar_width * 0.92,
                color=EDIT_ACTION_COLORS.get(action, ACTION_PANEL_COLORS["edit"]),
                alpha=0.96,
                edgecolor="#FFFFFF",
                linewidth=0.8,
            )

    legend_handles = [
        Patch(facecolor=EDIT_ACTION_COLORS.get(action, ACTION_PANEL_COLORS["edit"]), edgecolor="#FFFFFF", linewidth=0.8)
        for action in active_actions_global
    ]
    legend_labels = [ACTION_LABELS.get(action, action.replace("_", " ")) for action in active_actions_global]
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncol=min(4, max(1, len(legend_handles))),
            fontsize=7.3,
            columnspacing=0.9,
            handlelength=1.2,
        )

    ax.set_ylabel("Action count")
    ax.set_xlabel("Round")
    ax.set_xticks(center_positions)
    ax.set_xticklabels(rounds)
    ax.grid(axis="y", linestyle=":", linewidth=0.55, color="#9A9A9A", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(x=0.06, y=0.16)
    _style_axis(ax)
    return fig, ax


def plot_round_action_count_figure(distribution: RoundActionDistribution, path: Path) -> None:
    fig, _ = create_round_action_count_figure(distribution)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_safe_path(path))
    if path.suffix.lower() == ".pdf":
        fig.savefig(_safe_path(path.with_suffix(".png")))
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_action_distribution(distribution: ActionDistribution, path: Path) -> None:
    fig, _ = create_action_distribution_figure(distribution)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_safe_path(path))
    if path.suffix.lower() == ".pdf":
        fig.savefig(_safe_path(path.with_suffix(".png")))
    import matplotlib.pyplot as plt

    plt.close(fig)


def create_stop_round_distribution_figure(distribution: StopRoundDistribution) -> tuple[Any, Any]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    rounds = _ordered_round_names(distribution.stop_round_counts)
    display_rounds = [_display_round_label(round_name) for round_name in rounds]
    counts = [int(distribution.stop_round_counts.get(round_name, 0)) for round_name in rounds]
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 2.9), constrained_layout=True)
    bars = ax.bar(
        display_rounds,
        counts,
        color=COMMIT_ACTION_COLORS["continue"],
        alpha=0.96,
        width=0.62,
        edgecolor="#FFFFFF",
        linewidth=0.9,
    )
    ax.set_ylabel("Episodes")
    ax.set_xlabel("Round")
    ax.grid(axis="y", linestyle=":", linewidth=0.55, color="#9A9A9A", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(y=0.16)
    _style_axis(ax)
    total = max(1, sum(counts))
    for bar, count in zip(bars, counts, strict=True):
        if count <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({100.0 * count / total:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#333333",
        )
    return fig, ax


def plot_stop_round_distribution(distribution: StopRoundDistribution, path: Path) -> None:
    fig, _ = create_stop_round_distribution_figure(distribution)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_safe_path(path))
    if path.suffix.lower() == ".pdf":
        fig.savefig(_safe_path(path.with_suffix(".png")))
    import matplotlib.pyplot as plt

    plt.close(fig)


def create_controller_behavior_figure(
    stop_distribution: StopRoundDistribution,
    round_action_distribution: RoundActionDistribution,
) -> tuple[Any, list[Any]]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    stop_rounds = _ordered_round_names(stop_distribution.stop_round_counts)
    display_stop_rounds = [_display_round_label(round_name) for round_name in stop_rounds]
    stop_counts = [int(stop_distribution.stop_round_counts.get(round_name, 0)) for round_name in stop_rounds]
    stop_total = max(1, sum(stop_counts))

    action_rounds = _ordered_round_names(round_action_distribution.round_edit_counts)
    display_action_rounds = [_display_round_label(round_name) for round_name in action_rounds]
    active_actions = [
        action
        for action in EDIT_ACTION_ORDER
        if any(round_action_distribution.round_edit_counts.get(round_name, {}).get(action, 0) > 0 for round_name in action_rounds)
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=CURRENT_STYLE["controller_figsize"],
        gridspec_kw={"width_ratios": [0.92, 1.48]},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=CURRENT_STYLE["controller_top"], wspace=0.28)
    stop_ax, action_ax = axes

    bars = stop_ax.bar(
        display_stop_rounds,
        stop_counts,
        color=COMMIT_ACTION_COLORS["commit"],
        alpha=0.95,
        width=0.58,
        edgecolor="#FFFFFF",
        linewidth=0.9,
        zorder=3,
    )
    stop_ax.set_ylabel("Episodes")
    stop_ax.set_xlabel("Round")
    stop_ax.grid(axis="y", linestyle=":", linewidth=0.55, color="#9A9A9A", alpha=0.35)
    stop_ax.set_axisbelow(True)
    stop_ax.margins(y=0.16)
    _style_axis(stop_ax)
    for bar, count in zip(bars, stop_counts, strict=True):
        if count <= 0:
            continue
        stop_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({100.0 * count / stop_total:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=CURRENT_STYLE["controller_bar_label_fontsize"],
            color="#333333",
        )

    bottoms = [0.0 for _ in action_rounds]
    for action in active_actions:
        fractions = []
        for round_name in action_rounds:
            counts = round_action_distribution.round_edit_counts.get(round_name, {})
            total = sum(int(value) for value in counts.values())
            action_count = int(counts.get(action, 0))
            fractions.append((action_count / total) if total > 0 else 0.0)
        action_ax.bar(
            display_action_rounds,
            fractions,
            bottom=bottoms,
            width=0.64,
            color=EDIT_ACTION_COLORS.get(action, ACTION_PANEL_COLORS["edit"]),
            alpha=0.97,
            edgecolor="#FFFFFF",
            linewidth=0.8,
            label=ACTION_LABELS.get(action, action.replace("_", " ")),
            zorder=3,
        )
        bottoms = [base + frac for base, frac in zip(bottoms, fractions, strict=True)]

    action_ax.set_ylabel("Edit fraction")
    action_ax.set_xlabel("Round")
    action_ax.set_ylim(0.0, 1.0)
    action_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    action_ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    action_ax.grid(axis="y", linestyle=":", linewidth=0.55, color="#9A9A9A", alpha=0.35)
    action_ax.set_axisbelow(True)
    action_ax.margins(x=0.04, y=0.03)
    _style_axis(action_ax)

    legend_handles = [
        Patch(
            facecolor=EDIT_ACTION_COLORS.get(action, ACTION_PANEL_COLORS["edit"]),
            edgecolor="#FFFFFF",
            linewidth=0.7,
        )
        for action in active_actions
    ]
    legend_labels = [ACTION_LABELS.get(action, action.replace("_", " ")) for action in active_actions]
    if legend_handles:
        action_ax.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=min(3, max(1, len(legend_handles))),
            fontsize=CURRENT_STYLE["controller_legend_fontsize"],
            columnspacing=1.0,
            handlelength=1.15,
            handletextpad=0.45,
            borderaxespad=0.0,
        )
    return fig, [stop_ax, action_ax]


def plot_controller_behavior_figure(
    stop_distribution: StopRoundDistribution,
    round_action_distribution: RoundActionDistribution,
    path: Path,
) -> None:
    fig, _ = create_controller_behavior_figure(stop_distribution, round_action_distribution)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_safe_path(path))
    if path.suffix.lower() == ".pdf":
        fig.savefig(_safe_path(path.with_suffix(".png")))
    import matplotlib.pyplot as plt

    plt.close(fig)


def create_round_signal_trajectory_figure(
    distribution: RoundSignalDistribution,
) -> tuple[Any, list[Any]]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    rounds = _ordered_round_names(distribution.round_signal_means)
    x = list(range(len(rounds)))
    display_rounds = [_display_round_label(round_name) for round_name in rounds]
    plt.rcParams.update(
        {
            "font.size": CURRENT_STYLE["signal_fontsize"],
            "axes.labelsize": CURRENT_STYLE["signal_axes_labelsize"],
            "xtick.labelsize": CURRENT_STYLE["signal_tick_labelsize"],
            "ytick.labelsize": CURRENT_STYLE["signal_tick_labelsize"],
            "legend.fontsize": CURRENT_STYLE["signal_legend_fontsize"],
        }
    )
    fig, axis = plt.subplots(
        1,
        1,
        figsize=CURRENT_STYLE["signal_figsize"],
        constrained_layout=True,
    )
    for signal_name in SIGNAL_ORDER:
        means = [
            float(distribution.round_signal_means.get(round_name, {}).get(signal_name, 0.0))
            for round_name in rounds
        ]
        color = SIGNAL_COLORS[signal_name]
        axis.plot(
            x,
            means,
            color=color,
            marker=SIGNAL_MARKERS[signal_name],
            linewidth=CURRENT_STYLE["signal_linewidth"],
            markersize=CURRENT_STYLE["signal_markersize"],
            markerfacecolor="#FFFFFF",
            markeredgecolor=color,
            markeredgewidth=CURRENT_STYLE["signal_markeredgewidth"],
            label=SIGNAL_LABELS[signal_name],
        )
    axis.set_title("")
    axis.set_ylabel("Post-round signal")
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xticks(x)
    axis.set_xticklabels(display_rounds)
    if x:
        axis.set_xlim(-0.08, len(x) - 1 + 0.08)
    axis.set_xlabel("Round")
    axis.grid(axis="y", linestyle=":", linewidth=0.55, color="#9A9A9A", alpha=0.35)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", pad=2.0)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.6,
        borderaxespad=0.1,
    )
    _style_axis(axis)
    return fig, [axis]


def plot_round_signal_trajectory_figure(
    distribution: RoundSignalDistribution,
    path: Path,
) -> None:
    fig, _ = create_round_signal_trajectory_figure(distribution)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_safe_path(path))
    if path.suffix.lower() == ".pdf":
        fig.savefig(_safe_path(path.with_suffix(".png")))
    import matplotlib.pyplot as plt

    plt.close(fig)


def _stop_round_from_summary(summary: dict[str, Any]) -> str:
    stop_reason = str(summary.get("stop_reason", "")).strip()
    match = re.search(r"(Round\d+)", stop_reason)
    if match:
        return match.group(1)
    matured_at_round = str(summary.get("matured_at_round", "")).strip()
    if matured_at_round:
        return matured_at_round
    rounds = summary.get("rounds", [])
    if isinstance(rounds, list) and rounds:
        last = rounds[-1]
        if isinstance(last, dict):
            round_name = str(last.get("round", "")).strip()
            if round_name:
                return round_name
    executed_round_count = summary.get("executed_round_count")
    try:
        executed_round_int = int(executed_round_count)
    except (TypeError, ValueError):
        executed_round_int = 0
    if executed_round_int > 0:
        return f"Round{executed_round_int}"
    return ""


def _stop_round_from_graph(graph: dict[str, Any]) -> str:
    metadata = graph.get("metadata", {})
    if isinstance(metadata, dict):
        post_round_rows = metadata.get("post_round_commit_rows", [])
        if isinstance(post_round_rows, list) and post_round_rows:
            last = post_round_rows[-1]
            if isinstance(last, dict):
                round_name = str(last.get("round_name", last.get("round", ""))).strip()
                if not round_name:
                    round_name = _round_name_from_state_id(str(last.get("state_id", "")).strip())
                if round_name:
                    return round_name
            return f"Round{len(post_round_rows)}"
        traces = metadata.get("parallel_round_traces", [])
        if isinstance(traces, list) and traces:
            last = traces[-1]
            if isinstance(last, dict):
                round_name = str(last.get("round", "")).strip()
                if round_name:
                    return round_name
            return f"Round{len(traces)}"
    return ""


def collect_stop_round_distribution(
    batch_root: Path,
    *,
    baseline_name: str = DEFAULT_BASELINE,
    benchmark: str = DEFAULT_BENCHMARK,
    category: str = DEFAULT_CATEGORY,
) -> StopRoundDistribution:
    rows = _read_run_rows(batch_root)
    stop_round_counts: Counter[str] = Counter()
    run_count = 0

    for row in rows:
        if row.get("baseline_name", "").strip() != baseline_name:
            continue
        if not _benchmark_matches(row, benchmark):
            continue
        if not _category_matches(row, category):
            continue
        run_dir = _resolve_run_dir(batch_root, row.get("run_dir", ""))
        summary_path = run_dir / "summary.json"
        graph_path = run_dir / "graph.json"

        round_name = ""
        if _path_exists(summary_path):
            round_name = _stop_round_from_summary(_read_json(summary_path))
        if not round_name and _path_exists(graph_path):
            round_name = _stop_round_from_graph(_read_json(graph_path))
        if not round_name:
            continue
        stop_round_counts.update([round_name])
        run_count += 1

    if run_count == 0:
        raise ValueError(
            f"No matching stop-round artifacts found for baseline={baseline_name}, "
            f"benchmark={benchmark}, category={category} under {batch_root}."
        )

    return StopRoundDistribution(
        benchmark=benchmark,
        category=category,
        baseline_name=baseline_name,
        run_count=run_count,
        stop_round_counts=stop_round_counts,
    )


def _plot_distribution_row(axes: tuple[Any, Any], distribution: ActionDistribution) -> None:
    edit_ax, commit_ax = axes
    _draw_action_panel(
        edit_ax,
        _active_action_items(distribution.edit_counts, EDIT_ACTION_ORDER),
        label="Edit actions",
        color=ACTION_PANEL_COLORS["edit"],
        color_map=EDIT_ACTION_COLORS,
    )
    _draw_action_panel(
        commit_ax,
        _active_action_items(distribution.commit_counts, COMMIT_ACTION_ORDER),
        label="Commit decisions",
        color=ACTION_PANEL_COLORS["commit"],
        color_map=COMMIT_ACTION_COLORS,
    )
    edit_ax.tick_params(axis="x", rotation=0)
    commit_ax.tick_params(axis="x", rotation=0)


def create_comparison_figure(
    aggregate: ActionDistribution,
    topic: ActionDistribution,
) -> tuple[Any, Any]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.6, 5.15),
        gridspec_kw={"width_ratios": [4.6, 1.2], "height_ratios": [1.0, 1.0]},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.94, bottom=0.11, hspace=0.32, wspace=0.16)
    _plot_distribution_row((axes[0][0], axes[0][1]), aggregate)
    _plot_distribution_row((axes[1][0], axes[1][1]), topic)

    fig.text(
        0.12,
        0.965,
        f"All {_benchmark_display_name(aggregate.benchmark)} topics (n={aggregate.run_count})",
        ha="left",
        va="top",
        fontsize=8.8,
        color="#2F2F2F",
    )
    fig.text(
        0.12,
        0.49,
        f"Largest topic: {topic.category} (n={topic.run_count})",
        ha="left",
        va="top",
        fontsize=8.8,
        color="#2F2F2F",
    )
    return fig, axes


def plot_comparison_figure(
    aggregate: ActionDistribution,
    topic: ActionDistribution,
    path: Path,
) -> None:
    fig, _ = create_comparison_figure(aggregate, topic)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_safe_path(path))
    if path.suffix.lower() == ".pdf":
        fig.savefig(_safe_path(path.with_suffix(".png")))
    import matplotlib.pyplot as plt

    plt.close(fig)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "all"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze EIG edit-action and commit-decision distributions and export a paper-ready bar chart."
    )
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT, help="Batch directory containing selected_rows.csv.")
    parser.add_argument("--baseline-name", default=DEFAULT_BASELINE, help="Baseline/method name to analyze.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark to filter, e.g. liveideabench.")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="Topic category to filter, or 'all'.")
    parser.add_argument(
        "--topic-category",
        default="auto",
        help="Topic category for comparison mode; use 'auto' to select the largest available category.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../paper/ideation_2026/supporting"),
        help="Directory for CSV and summary JSON outputs.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=None,
        help="Path for the generated PDF/PNG figure. Defaults to ../paper/ideation_2026/figures.",
    )
    parser.add_argument(
        "--comparison-figure-path",
        type=Path,
        default=None,
        help="If set, export a 2x2 aggregate-vs-topic comparison figure instead of a single distribution figure.",
    )
    parser.add_argument(
        "--left-batch-root",
        type=Path,
        default=None,
        help="Batch root for the left side of calibration-comparison figures.",
    )
    parser.add_argument(
        "--right-batch-root",
        type=Path,
        default=None,
        help="Batch root for the right side of calibration-comparison figures.",
    )
    parser.add_argument(
        "--left-label",
        default="No calibration",
        help="Label for the left calibration-comparison column.",
    )
    parser.add_argument(
        "--right-label",
        default="Calibration",
        help="Label for the right calibration-comparison column.",
    )
    parser.add_argument(
        "--overall-comparison-figure",
        type=Path,
        default=None,
        help="If set with left/right batch roots, export the 1x2 overall action-distribution comparison.",
    )
    parser.add_argument(
        "--round-comparison-figure",
        type=Path,
        default=None,
        help="If set with left/right batch roots, export the 1x2 per-round stacked action-distribution comparison.",
    )
    parser.add_argument(
        "--signal-figure-path",
        type=Path,
        default=None,
        help="If set, export a round-signal trajectory figure from post-round graph signal snapshots.",
    )
    parser.add_argument(
        "--action-csv-path",
        type=Path,
        default=None,
        help="If set, draw the action-distribution figure directly from an existing CSV export.",
    )
    parser.add_argument(
        "--round-signal-csv-path",
        type=Path,
        default=None,
        help="If set, draw the round-signal figure directly from an existing CSV export.",
    )
    parser.add_argument(
        "--round-action-csv-path",
        type=Path,
        default=None,
        help="If set, draw the per-round action-count figure directly from an existing CSV export.",
    )
    parser.add_argument(
        "--stop-round-csv-path",
        type=Path,
        default=None,
        help="If set, draw the stop-round distribution figure directly from an existing CSV export.",
    )
    parser.add_argument(
        "--stop-round-figure-path",
        type=Path,
        default=None,
        help="If set, export a stop-round distribution figure.",
    )
    parser.add_argument(
        "--round-action-figure-path",
        type=Path,
        default=None,
        help="If set, export a per-round action-count figure.",
    )
    parser.add_argument(
        "--controller-behavior-figure-path",
        type=Path,
        default=None,
        help="If set with stop-round and round-action inputs, export a combined 1x2 controller-behavior figure.",
    )
    parser.add_argument(
        "--style-preset",
        choices=sorted(STYLE_PRESETS.keys()),
        default=CURRENT_STYLE_PRESET,
        help="Visual style preset for exported figures.",
    )
    args = parser.parse_args()
    _apply_style_preset(args.style_preset)

    if (
        args.action_csv_path is not None
        or args.round_signal_csv_path is not None
        or args.round_action_csv_path is not None
        or args.stop_round_csv_path is not None
    ):
        loaded_round_action_distribution = None
        loaded_stop_round_distribution = None
        if args.action_csv_path is not None:
            distribution = load_action_distribution_from_csv(
                args.action_csv_path,
                benchmark=args.benchmark,
                category=args.category,
                baseline_name=args.baseline_name,
            )
            figure_path = args.figure_path or args.action_csv_path.with_name(
                f"{args.action_csv_path.stem}_figure.pdf"
            )
            plot_action_distribution(distribution, figure_path)
            print(f"Drew action figure from CSV: {args.action_csv_path}")
            print(f"Wrote figure: {figure_path}")
            if figure_path.suffix.lower() == ".pdf":
                print(f"Wrote figure preview: {figure_path.with_suffix('.png')}")
        if args.round_signal_csv_path is not None:
            round_signal_distribution = load_round_signal_distribution_from_csv(
                args.round_signal_csv_path,
                benchmark=args.benchmark,
                category=args.category,
                baseline_name=args.baseline_name,
            )
            signal_figure_path = args.signal_figure_path or args.round_signal_csv_path.with_name(
                f"{args.round_signal_csv_path.stem}_figure.pdf"
            )
            plot_round_signal_trajectory_figure(round_signal_distribution, signal_figure_path)
            print(f"Drew round-signal figure from CSV: {args.round_signal_csv_path}")
            print(f"Wrote round-signal figure: {signal_figure_path}")
            if signal_figure_path.suffix.lower() == ".pdf":
                print(
                    f"Wrote round-signal preview: {signal_figure_path.with_suffix('.png')}"
                )
        if args.round_action_csv_path is not None:
            round_action_distribution = load_round_action_distribution_from_csv(
                args.round_action_csv_path,
                benchmark=args.benchmark,
                category=args.category,
                baseline_name=args.baseline_name,
            )
            loaded_round_action_distribution = round_action_distribution
            round_action_figure_path = args.round_action_figure_path or args.round_action_csv_path.with_name(
                f"{args.round_action_csv_path.stem}_figure.pdf"
            )
            plot_round_action_count_figure(round_action_distribution, round_action_figure_path)
            print(f"Drew round-action figure from CSV: {args.round_action_csv_path}")
            print(f"Wrote round-action figure: {round_action_figure_path}")
            if round_action_figure_path.suffix.lower() == ".pdf":
                print(
                    f"Wrote round-action preview: {round_action_figure_path.with_suffix('.png')}"
                )
        if args.stop_round_csv_path is not None:
            stop_round_distribution = load_stop_round_distribution_from_csv(
                args.stop_round_csv_path,
                benchmark=args.benchmark,
                category=args.category,
                baseline_name=args.baseline_name,
            )
            loaded_stop_round_distribution = stop_round_distribution
            stop_round_figure_path = args.stop_round_figure_path or args.stop_round_csv_path.with_name(
                f"{args.stop_round_csv_path.stem}_figure.pdf"
            )
            plot_stop_round_distribution(stop_round_distribution, stop_round_figure_path)
            print(f"Drew stop-round figure from CSV: {args.stop_round_csv_path}")
            print(f"Wrote stop-round figure: {stop_round_figure_path}")
            if stop_round_figure_path.suffix.lower() == ".pdf":
                print(
                    f"Wrote stop-round preview: {stop_round_figure_path.with_suffix('.png')}"
                )
        if args.controller_behavior_figure_path is not None:
            if loaded_round_action_distribution is None or loaded_stop_round_distribution is None:
                raise ValueError(
                    "--controller-behavior-figure-path requires both --round-action-csv-path and --stop-round-csv-path."
                )
            plot_controller_behavior_figure(
                loaded_stop_round_distribution,
                loaded_round_action_distribution,
                args.controller_behavior_figure_path,
            )
            print(f"Wrote controller-behavior figure: {args.controller_behavior_figure_path}")
            if args.controller_behavior_figure_path.suffix.lower() == ".pdf":
                print(
                    f"Wrote controller-behavior preview: "
                    f"{args.controller_behavior_figure_path.with_suffix('.png')}"
                )
        return

    if args.left_batch_root is not None or args.right_batch_root is not None:
        if args.left_batch_root is None or args.right_batch_root is None:
            raise ValueError("--left-batch-root and --right-batch-root must be provided together.")
        if args.overall_comparison_figure is None and args.round_comparison_figure is None:
            raise ValueError(
                "Set --overall-comparison-figure, --round-comparison-figure, or both."
            )
        left = collect_action_distribution(
            args.left_batch_root,
            baseline_name=args.baseline_name,
            benchmark=args.benchmark,
            category=args.category,
        )
        right = collect_action_distribution(
            args.right_batch_root,
            baseline_name=args.baseline_name,
            benchmark=args.benchmark,
            category=args.category,
        )
        left_round = collect_round_action_distribution(
            args.left_batch_root,
            baseline_name=args.baseline_name,
            benchmark=args.benchmark,
            category=args.category,
        )
        right_round = collect_round_action_distribution(
            args.right_batch_root,
            baseline_name=args.baseline_name,
            benchmark=args.benchmark,
            category=args.category,
        )
        stem = f"action_distribution_calibration_{_slug(args.benchmark)}_{_slug(args.category)}"
        write_distribution_csv(left, args.output_dir / f"{stem}_left_overall.csv")
        write_distribution_csv(right, args.output_dir / f"{stem}_right_overall.csv")
        write_round_distribution_csv(left_round, args.output_dir / f"{stem}_left_by_round.csv")
        write_round_distribution_csv(right_round, args.output_dir / f"{stem}_right_by_round.csv")
        if args.overall_comparison_figure is not None:
            plot_overall_calibration_comparison_figure(
                left,
                right,
                left_label=args.left_label,
                right_label=args.right_label,
                path=args.overall_comparison_figure,
            )
        if args.round_comparison_figure is not None:
            plot_round_calibration_comparison_figure(
                left_round,
                right_round,
                left_label=args.left_label,
                right_label=args.right_label,
                path=args.round_comparison_figure,
            )
        print(
            f"Analyzed calibration comparison: {args.left_label} n={left.run_count}, "
            f"{args.right_label} n={right.run_count}."
        )
        if args.overall_comparison_figure is not None:
            print(f"Wrote overall comparison figure: {args.overall_comparison_figure}")
        if args.round_comparison_figure is not None:
            print(f"Wrote round comparison figure: {args.round_comparison_figure}")
        return

    if args.comparison_figure_path is not None:
        aggregate, topic = collect_comparison_distributions(
            args.batch_root,
            baseline_name=args.baseline_name,
            benchmark=args.benchmark,
            topic_category=args.topic_category,
        )
        aggregate_stem = f"action_distribution_{_slug(args.benchmark)}_all"
        topic_stem = f"action_distribution_{_slug(args.benchmark)}_{_slug(topic.category)}"
        comparison_stem = f"action_distribution_{_slug(args.benchmark)}_comparison_summary"
        write_distribution_csv(aggregate, args.output_dir / f"{aggregate_stem}.csv")
        write_summary_json(aggregate, args.output_dir / f"{aggregate_stem}_summary.json")
        write_distribution_csv(topic, args.output_dir / f"{topic_stem}.csv")
        write_summary_json(topic, args.output_dir / f"{topic_stem}_summary.json")
        comparison_payload = {
            "aggregate": {
                "benchmark": aggregate.benchmark,
                "category": aggregate.category,
                "baseline_name": aggregate.baseline_name,
                "run_count": aggregate.run_count,
                "edit_counts": aggregate.edit_counts,
                "commit_counts": aggregate.commit_counts,
            },
            "topic": {
                "benchmark": topic.benchmark,
                "category": topic.category,
                "baseline_name": topic.baseline_name,
                "run_count": topic.run_count,
                "edit_counts": topic.edit_counts,
                "commit_counts": topic.commit_counts,
            },
        }
        comparison_path = args.output_dir / f"{comparison_stem}.json"
        comparison_path.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")
        plot_comparison_figure(aggregate, topic, args.comparison_figure_path)
        print(
            f"Analyzed aggregate {aggregate.run_count} runs and topic {topic.category} "
            f"({topic.run_count} runs) for {args.baseline_name} on {args.benchmark}."
        )
        print(f"Wrote aggregate CSV: {args.output_dir / f'{aggregate_stem}.csv'}")
        print(f"Wrote topic CSV: {args.output_dir / f'{topic_stem}.csv'}")
        print(f"Wrote comparison summary: {comparison_path}")
        print(f"Wrote comparison figure: {args.comparison_figure_path}")
        if args.comparison_figure_path.suffix.lower() == ".pdf":
            print(
                f"Wrote comparison preview: {args.comparison_figure_path.with_suffix('.png')}"
            )
        return

    distribution = collect_action_distribution(
        args.batch_root,
        baseline_name=args.baseline_name,
        benchmark=args.benchmark,
        category=args.category,
    )
    round_action_distribution = None
    round_action_csv_path = None
    round_signal_distribution = None
    round_signal_csv_path = None
    stop_round_distribution = None
    stop_round_csv_path = None
    if args.round_action_figure_path is not None:
        round_action_distribution = collect_round_action_distribution(
            args.batch_root,
            baseline_name=args.baseline_name,
            benchmark=args.benchmark,
            category=args.category,
        )
    if args.signal_figure_path is not None:
        round_signal_distribution = collect_round_signal_distribution(
            args.batch_root,
            baseline_name=args.baseline_name,
            benchmark=args.benchmark,
            category=args.category,
        )
    if args.stop_round_figure_path is not None:
        stop_round_distribution = collect_stop_round_distribution(
            args.batch_root,
            baseline_name=args.baseline_name,
            benchmark=args.benchmark,
            category=args.category,
        )
    if args.controller_behavior_figure_path is not None:
        if round_action_distribution is None:
            round_action_distribution = collect_round_action_distribution(
                args.batch_root,
                baseline_name=args.baseline_name,
                benchmark=args.benchmark,
                category=args.category,
            )
        if stop_round_distribution is None:
            stop_round_distribution = collect_stop_round_distribution(
                args.batch_root,
                baseline_name=args.baseline_name,
                benchmark=args.benchmark,
                category=args.category,
            )
    stem = f"action_distribution_{_slug(args.benchmark)}_{_slug(args.category)}"
    round_action_stem = f"round_action_distribution_{_slug(args.benchmark)}_{_slug(args.category)}"
    signal_stem = f"round_signal_trajectory_{_slug(args.benchmark)}_{_slug(args.category)}"
    stop_round_stem = f"stop_round_distribution_{_slug(args.benchmark)}_{_slug(args.category)}"
    csv_path = args.output_dir / f"{stem}.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    if round_action_distribution is not None:
        round_action_csv_path = args.output_dir / f"{round_action_stem}.csv"
    if round_signal_distribution is not None:
        round_signal_csv_path = args.output_dir / f"{signal_stem}.csv"
    if stop_round_distribution is not None:
        stop_round_csv_path = args.output_dir / f"{stop_round_stem}.csv"
    figure_path = args.figure_path or Path("../paper/ideation_2026/figures") / f"{stem}.pdf"

    write_distribution_csv(distribution, csv_path)
    write_summary_json(distribution, summary_path)
    plot_action_distribution(distribution, figure_path)
    if round_action_distribution is not None and round_action_csv_path is not None:
        write_round_distribution_csv(round_action_distribution, round_action_csv_path)
        plot_round_action_count_figure(round_action_distribution, args.round_action_figure_path)
    if round_signal_distribution is not None and round_signal_csv_path is not None:
        write_round_signal_csv(round_signal_distribution, round_signal_csv_path)
        plot_round_signal_trajectory_figure(round_signal_distribution, args.signal_figure_path)
    if stop_round_distribution is not None and stop_round_csv_path is not None:
        write_stop_round_csv(stop_round_distribution, stop_round_csv_path)
        plot_stop_round_distribution(stop_round_distribution, args.stop_round_figure_path)
    if (
        args.controller_behavior_figure_path is not None
        and stop_round_distribution is not None
        and round_action_distribution is not None
    ):
        plot_controller_behavior_figure(
            stop_round_distribution,
            round_action_distribution,
            args.controller_behavior_figure_path,
        )

    print(
        f"Analyzed {distribution.run_count} runs for {args.baseline_name} "
        f"on {args.benchmark}/{args.category}."
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote figure: {figure_path}")
    if figure_path.suffix.lower() == ".pdf":
        print(f"Wrote figure preview: {figure_path.with_suffix('.png')}")
    if round_action_distribution is not None and round_action_csv_path is not None:
        print(f"Wrote round-action CSV: {round_action_csv_path}")
        print(f"Wrote round-action figure: {args.round_action_figure_path}")
        if args.round_action_figure_path.suffix.lower() == ".pdf":
            print(
                f"Wrote round-action preview: {args.round_action_figure_path.with_suffix('.png')}"
            )
    if round_signal_distribution is not None and round_signal_csv_path is not None:
        print(f"Wrote round-signal CSV: {round_signal_csv_path}")
        print(f"Wrote round-signal figure: {args.signal_figure_path}")
        if args.signal_figure_path.suffix.lower() == ".pdf":
            print(
                f"Wrote round-signal preview: {args.signal_figure_path.with_suffix('.png')}"
            )
    if stop_round_distribution is not None and stop_round_csv_path is not None:
        print(f"Wrote stop-round CSV: {stop_round_csv_path}")
        print(f"Wrote stop-round figure: {args.stop_round_figure_path}")
        if args.stop_round_figure_path.suffix.lower() == ".pdf":
            print(
                f"Wrote stop-round preview: {args.stop_round_figure_path.with_suffix('.png')}"
            )
    if args.controller_behavior_figure_path is not None:
        print(f"Wrote controller-behavior figure: {args.controller_behavior_figure_path}")
        if args.controller_behavior_figure_path.suffix.lower() == ".pdf":
            print(
                f"Wrote controller-behavior preview: "
                f"{args.controller_behavior_figure_path.with_suffix('.png')}"
            )


if __name__ == "__main__":
    main()
