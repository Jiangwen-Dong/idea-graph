from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory where run artifacts will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    instance = load_instance(args.input)
    graph = run_experiment(
        topic=str(instance["topic"]),
        literature=list(instance["literature"]),
    )
    run_dir = write_run_artifacts(
        graph,
        output_root=args.output_dir,
        instance_name=str(instance["name"]),
        source_path=str(instance["source_path"]),
    )

    print("== Instance ==")
    print(instance["name"])
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
