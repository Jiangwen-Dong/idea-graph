from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.evaluation import evaluate_graph, format_evaluation_markdown
from idea_graph.models import (
    Branch,
    Edge,
    FinalProposal,
    GraphAction,
    IdeaGraph,
    MaturitySnapshot,
    Node,
    Provenance,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an existing idea-graph run directory.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a run directory that contains graph.json.",
    )
    return parser


def graph_from_payload(payload: dict[str, object]) -> IdeaGraph:
    graph = IdeaGraph(
        topic=str(payload.get("topic", "")).strip(),
        literature=[str(item).strip() for item in payload.get("literature", []) if str(item).strip()],
        metadata=dict(payload.get("metadata", {}) or {}),
    )

    nodes_payload = payload.get("nodes", {})
    if isinstance(nodes_payload, dict):
        for node_id, node_payload in nodes_payload.items():
            if not isinstance(node_payload, dict):
                continue
            provenance = [
                Provenance(
                    role=str(item.get("role", "")).strip(),
                    branch_id=str(item.get("branch_id", "")).strip(),
                    source=str(item.get("source", "")).strip(),
                )
                for item in node_payload.get("provenance", [])
                if isinstance(item, dict)
            ]
            graph.nodes[str(node_id)] = Node(
                id=str(node_payload.get("id", node_id)).strip(),
                type=str(node_payload.get("type", "")).strip(),
                text=str(node_payload.get("text", "")).strip(),
                role=str(node_payload.get("role", "")).strip(),
                branch_id=str(node_payload.get("branch_id", "")).strip(),
                confidence=float(node_payload.get("confidence", 0.0) or 0.0),
                evidence=[str(item).strip() for item in node_payload.get("evidence", []) if str(item).strip()],
                status=str(node_payload.get("status", "active")).strip() or "active",
                provenance=provenance,
            )

    edges_payload = payload.get("edges", [])
    if isinstance(edges_payload, list):
        for edge_payload in edges_payload:
            if not isinstance(edge_payload, dict):
                continue
            graph.edges.append(
                Edge(
                    id=str(edge_payload.get("id", "")).strip(),
                    source_id=str(edge_payload.get("source_id", "")).strip(),
                    relation=str(edge_payload.get("relation", "")).strip(),
                    target_id=str(edge_payload.get("target_id", "")).strip(),
                    role=str(edge_payload.get("role", "")).strip(),
                    branch_id=str(edge_payload.get("branch_id", "")).strip(),
                    evidence_id=str(edge_payload.get("evidence_id", "")).strip() or None,
                    note=str(edge_payload.get("note", "")).strip(),
                    resolved=bool(edge_payload.get("resolved", False)),
                )
            )

    branches_payload = payload.get("branches", {})
    if isinstance(branches_payload, dict):
        for branch_id, branch_payload in branches_payload.items():
            if not isinstance(branch_payload, dict):
                continue
            graph.branches[str(branch_id)] = Branch(
                id=str(branch_payload.get("id", branch_id)).strip(),
                role=str(branch_payload.get("role", "")).strip(),
                node_ids=[str(item).strip() for item in branch_payload.get("node_ids", []) if str(item).strip()],
                edge_ids=[str(item).strip() for item in branch_payload.get("edge_ids", []) if str(item).strip()],
                frozen=bool(branch_payload.get("frozen", False)),
                rejected=bool(branch_payload.get("rejected", False)),
                notes=[str(item).strip() for item in branch_payload.get("notes", []) if str(item).strip()],
            )

    actions_payload = payload.get("actions", [])
    if isinstance(actions_payload, list):
        for action_payload in actions_payload:
            if not isinstance(action_payload, dict):
                continue
            graph.actions.append(
                GraphAction(
                    id=str(action_payload.get("id", "")).strip(),
                    round_name=str(action_payload.get("round_name", "")).strip(),
                    role=str(action_payload.get("role", "")).strip(),
                    kind=str(action_payload.get("kind", "")).strip(),
                    target_ids=[str(item).strip() for item in action_payload.get("target_ids", []) if str(item).strip()],
                    payload=dict(action_payload.get("payload", {}) or {}),
                    rationale=str(action_payload.get("rationale", "")).strip(),
                )
            )

    round_summaries_payload = payload.get("round_summaries", [])
    if isinstance(round_summaries_payload, list):
        for item in round_summaries_payload:
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], dict):
                continue
            graph.round_summaries.append(
                (
                    str(item[0]).strip(),
                    MaturitySnapshot(
                        support_coverage=float(item[1].get("support_coverage", 0.0) or 0.0),
                        unresolved_contradiction_ratio=float(item[1].get("unresolved_contradiction_ratio", 1.0) or 0.0),
                        utility=float(item[1].get("utility", 0.0) or 0.0),
                        utility_stable=bool(item[1].get("utility_stable", False)),
                        completeness=bool(item[1].get("completeness", False)),
                        is_mature=bool(item[1].get("is_mature", False)),
                    ),
                )
            )

    graph.matured_at_round = payload.get("matured_at_round")
    final_subgraph_payload = payload.get("final_subgraph")
    graph.final_subgraph = dict(final_subgraph_payload) if isinstance(final_subgraph_payload, dict) else None
    final_proposal_payload = payload.get("final_proposal")
    if isinstance(final_proposal_payload, dict):
        graph.final_proposal = FinalProposal(
            title=str(final_proposal_payload.get("title", "")).strip(),
            abstract=str(final_proposal_payload.get("abstract", "")).strip(),
            problem=str(final_proposal_payload.get("problem", "")).strip(),
            existing_methods=str(final_proposal_payload.get("existing_methods", "")).strip(),
            motivation=str(final_proposal_payload.get("motivation", "")).strip(),
            hypothesis=str(final_proposal_payload.get("hypothesis", "")).strip(),
            method=str(final_proposal_payload.get("method", "")).strip(),
            evaluation=str(final_proposal_payload.get("evaluation", "")).strip(),
            significance=str(final_proposal_payload.get("significance", "")).strip(),
            caveats=str(final_proposal_payload.get("caveats", "")).strip(),
        )
    return graph


def main() -> None:
    args = build_parser().parse_args()
    graph_path = args.run_dir / "graph.json"
    if not graph_path.exists():
        raise SystemExit(f"Could not find graph.json at {graph_path}")

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{graph_path} does not contain a JSON object.")

    graph = graph_from_payload(payload)
    evaluation = evaluate_graph(graph)

    (args.run_dir / "evaluation.json").write_text(
        json.dumps(evaluation.as_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (args.run_dir / "evaluation.md").write_text(
        format_evaluation_markdown(evaluation),
        encoding="utf-8",
    )
    summary_path = args.run_dir / "summary.json"
    if summary_path.exists():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary_payload, dict):
            summary_payload["idea_evaluation"] = evaluation.as_dict()
            summary_path.write_text(
                json.dumps(summary_payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

    print("== Idea Evaluation ==")
    print(f"Overall: {evaluation.overall_score}/10")
    for category, score in evaluation.category_scores.items():
        print(f"{category}: {score}/10")
    print(args.run_dir)


if __name__ == "__main__":
    main()
