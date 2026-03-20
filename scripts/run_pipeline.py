from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.benchmarks import (
    ai_idea_bench_2025_instance_from_record,
    download_ai_idea_bench_2025,
    download_liveideabench,
    get_ai_idea_bench_2025_record,
    get_liveideabench_record,
    liveideabench_instance_from_record,
)
from idea_graph.engine import run_experiment
from idea_graph.io import load_instance, write_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic idea-graph pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "sample_instance.json",
        help="Path to a JSON instance file.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["ai_idea_bench_2025", "liveideabench"],
        help="Use an official benchmark loader instead of a local instance JSON.",
    )
    parser.add_argument(
        "--benchmark-index",
        type=int,
        default=0,
        help="Benchmark selector when --benchmark is provided. Meaning depends on the benchmark.",
    )
    parser.add_argument(
        "--benchmark-keyword",
        help="Optional keyword filter for benchmarks that support keyword-based selection, such as liveideabench.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "data" / "benchmarks",
        help="Base cache directory for benchmark files.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download official benchmark metadata automatically if it is not cached locally.",
    )
    parser.add_argument(
        "--include-paper-assets",
        action="store_true",
        help="Also download the official paper archive when downloading the benchmark. This is about 31 GB.",
    )
    parser.add_argument(
        "--allow-large-download",
        action="store_true",
        help="Required to confirm that you really want the large paper-asset download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory where run artifacts will be written.",
    )
    return parser


def resolve_benchmark_root(base_root: Path, benchmark_name: str) -> Path:
    if base_root.name == benchmark_name:
        return base_root
    return base_root / benchmark_name


def main() -> None:
    args = build_parser().parse_args()
    if args.include_paper_assets and not args.allow_large_download:
        raise SystemExit(
            "--include-paper-assets is blocked by default because the archive is about 31 GB. "
            "Re-run with --allow-large-download if you really want it."
        )
    if args.benchmark == "ai_idea_bench_2025":
        benchmark_root = resolve_benchmark_root(args.benchmark_root, args.benchmark)
        if args.download_if_missing:
            download_ai_idea_bench_2025(
                benchmark_root,
                include_papers=args.include_paper_assets,
                extract_papers=False,
            )
        record = get_ai_idea_bench_2025_record(benchmark_root, args.benchmark_index)
        instance = ai_idea_bench_2025_instance_from_record(record, benchmark_root=benchmark_root)
    elif args.benchmark == "liveideabench":
        benchmark_root = resolve_benchmark_root(args.benchmark_root, args.benchmark)
        if args.include_paper_assets:
            raise SystemExit("liveideabench does not provide a separate paper-asset archive.")
        if args.download_if_missing:
            download_liveideabench(benchmark_root)
        record = get_liveideabench_record(
            benchmark_root,
            args.benchmark_index,
            keyword=args.benchmark_keyword,
        )
        instance = liveideabench_instance_from_record(
            record,
            benchmark_root=benchmark_root,
        )
    else:
        benchmark_root = None
        instance = load_instance(args.input)

    graph = run_experiment(
        topic=instance.topic,
        literature=list(instance.literature),
        metadata=dict(instance.metadata),
    )
    run_dir = write_run_artifacts(
        graph,
        output_root=args.output_dir,
        instance=instance,
    )

    print("== Instance ==")
    print(instance.name)
    print()

    if args.benchmark:
        print("== Benchmark ==")
        print(f"Name: {args.benchmark}")
        metadata = instance.metadata
        actual_index = metadata.get("benchmark_index", metadata.get("row_index", args.benchmark_index))
        print(f"Requested selector: {args.benchmark_index}")
        print(f"Resolved benchmark index: {actual_index}")
        if args.benchmark_keyword:
            print(f"Keyword filter: {args.benchmark_keyword}")
        print(f"Root: {benchmark_root}")
        print()

    print("== Topic ==")
    print(graph.topic)
    print()

    print("== Graph Summary ==")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print(f"Branches: {len(graph.branches)}")
    print(f"Actions: {len(graph.actions)}")
    print()

    print("== Round Maturity ==")
    for round_name, snapshot in graph.round_summaries:
        print(
            f"{round_name}: support={snapshot.support_coverage}, "
            f"contradictions={snapshot.unresolved_contradiction_ratio}, "
            f"utility={snapshot.utility}, stable={snapshot.utility_stable}, "
            f"complete={snapshot.completeness}, mature={snapshot.is_mature}"
        )
    print()

    print("== Final Proposal ==")
    assert graph.final_proposal is not None
    print(f"Problem: {graph.final_proposal.problem}")
    print(f"Hypothesis: {graph.final_proposal.hypothesis}")
    print(f"Method: {graph.final_proposal.method}")
    print(f"Evaluation: {graph.final_proposal.evaluation}")
    print()

    print("== Artifacts ==")
    print(run_dir)


if __name__ == "__main__":
    main()
