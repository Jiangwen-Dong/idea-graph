from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import ensure_parent_dir, write_text_file
from idea_graph.paper_eval_pool import (
    load_blocked_group_ids,
    select_paper_eval_candidates,
)

DEFAULT_OUTPUT_ROOT = ROOT / "outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v2"
DEFAULT_AIIB_METADATA = ROOT / "data/benchmarks/ai_idea_bench_2025/target_paper_data.json"
DEFAULT_LIVE_CSV = ROOT / "data/benchmarks/liveideabench/liveideabench_hf.csv"
DEFAULT_BLOCKED_CANDIDATES = (
    ROOT
    / "outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_candidate_pool_v1/candidate_instances.json",
    ROOT
    / "outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/candidate_instances.json",
    ROOT
    / "outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v1/candidate_instances.json",
)
DEFAULT_BLOCKED_SPLITS = (
    ROOT
    / "outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl",
    ROOT
    / "outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2_partitions/partition_manifest.jsonl",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen paper_eval_candidate_pool_v2 selection."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where candidate_instances.json, README.md, and pool_stats.json are written.",
    )
    parser.add_argument(
        "--target-aiib",
        type=int,
        required=True,
        help="Number of AI_Idea_Bench_2025 instances to freeze.",
    )
    parser.add_argument(
        "--target-live",
        type=int,
        required=True,
        help="Number of LiveIdeaBench instances to freeze.",
    )
    parser.add_argument(
        "--aiib-metadata",
        type=Path,
        default=DEFAULT_AIIB_METADATA,
        help="Path to AI_Idea_Bench_2025 metadata JSON.",
    )
    parser.add_argument(
        "--live-csv",
        type=Path,
        default=DEFAULT_LIVE_CSV,
        help="Path to the LiveIdeaBench CSV.",
    )
    parser.add_argument(
        "--blocked-candidate-file",
        action="append",
        type=Path,
        default=list(DEFAULT_BLOCKED_CANDIDATES),
        help="Path to a previous candidate_instances.json to treat as blocked (repeatable).",
    )
    parser.add_argument(
        "--blocked-split-registry",
        action="append",
        type=Path,
        default=list(DEFAULT_BLOCKED_SPLITS),
        help="Path to a split registry manifest to block (repeatable).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidate_sources = _existing_paths(args.blocked_candidate_file)
    split_sources = _existing_paths(args.blocked_split_registry)
    blocked_ids = load_blocked_group_ids(
        blocked_candidate_files=candidate_sources,
        blocked_split_registries=split_sources,
    )

    rows = select_paper_eval_candidates(
        aiib_metadata=args.aiib_metadata,
        live_csv=args.live_csv,
        target_aiib=args.target_aiib,
        target_live=args.target_live,
        blocked_candidate_files=candidate_sources,
        blocked_split_registries=split_sources,
    )

    ensure_parent_dir(args.output_root / "candidate_instances.json")
    write_json(args.output_root / "candidate_instances.json", rows)
    ensure_parent_dir(args.output_root / "README.md")
    write_text_file(
        args.output_root / "README.md",
        build_readme(
            target_aiib=args.target_aiib,
            target_live=args.target_live,
            total_rows=len(rows),
            blocked_candidate_files=candidate_sources,
            blocked_split_registries=split_sources,
        ),
    )
    ensure_parent_dir(args.output_root / "pool_stats.json")
    write_json(
        args.output_root / "pool_stats.json",
        build_pool_stats(rows, args.target_aiib, args.target_live, blocked_ids),
    )

    print(f"Candidate rows: {len(rows)}")
    print(f"Blocked group ids: {len(blocked_ids)}")
    print(f"Output dir: {args.output_root}")


def write_json(path: Path, payload: object) -> None:
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    write_text_file(path, content + "\n")


def build_readme(
    *,
    target_aiib: int,
    target_live: int,
    total_rows: int,
    blocked_candidate_files: Sequence[Path],
    blocked_split_registries: Sequence[Path],
) -> str:
    blocked_candidates = blocked_candidate_files or []
    blocked_registries = blocked_split_registries or []
    blocked_sources = []
    for path in blocked_candidates + blocked_registries:
        blocked_sources.append(f"- `{path}`")

    lines = [
        "# paper_eval_candidate_pool_v2",
        "",
        "This directory stores the frozen benchmark instances that remain untouched until the learned-controller paper evaluation.",
        "",
        "Status:",
        "",
        "- frozen",
        "- no new tuning/naming changes allowed",
        "",
        "Files:",
        "",
        "- `candidate_instances.json`",
        "- `README.md`",
        "- `pool_stats.json`",
        "",
        "Selection policy:",
        "",
        f"- {target_aiib} `AI_Idea_Bench_2025` instances",
        f"- {target_live} `LiveIdeaBench` instances",
        "- zero overlap with any blocked development or past paper-eval candidates",
        f"- total frozen candidates: {total_rows}",
        "",
        "Blocked sources:",
        "",
        *blocked_sources,
        "",
        f"Generated on {datetime.now(timezone.utc).isoformat()}",
    ]
    return "\n".join(lines)


def build_pool_stats(
    rows: Sequence[dict[str, object]],
    target_aiib: int,
    target_live: int,
    blocked_group_ids: set[str],
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    benchmark_counts: dict[str, int] = {"AI_Idea_Bench_2025": 0, "liveideabench": 0}
    for row in rows:
        benchmark = str(row.get("benchmark"))
        if benchmark in benchmark_counts:
            benchmark_counts[benchmark] += 1

    blocked_counts = {"AI_Idea_Bench_2025": 0, "liveideabench": 0}
    for group_id in blocked_group_ids:
        benchmark, _, _ = group_id.partition("::")
        if benchmark in blocked_counts:
            blocked_counts[benchmark] += 1

    benchmarks = {
        "AI_Idea_Bench_2025": {
            "target": target_aiib,
            "selected": benchmark_counts["AI_Idea_Bench_2025"],
            "blocked": blocked_counts["AI_Idea_Bench_2025"],
        },
        "liveideabench": {
            "target": target_live,
            "selected": benchmark_counts["liveideabench"],
            "blocked": blocked_counts["liveideabench"],
        },
    }
    total = {
        "target": target_aiib + target_live,
        "selected": len(rows),
        "blocked": len(blocked_group_ids),
    }
    return {"generated_at": now, "benchmarks": benchmarks, "total": total}


def _existing_paths(paths: Sequence[Path] | None) -> list[Path]:
    if not paths:
        return []
    return [path for path in paths if path and path.exists()]


if __name__ == "__main__":
    main()
