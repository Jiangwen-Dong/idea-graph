from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import re

from .engine import graph_as_dict
from .models import IdeaGraph


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "run"


def load_instance(path: str | Path) -> dict[str, object]:
    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))

    topic = str(data.get("topic", "")).strip()
    literature = data.get("literature", [])
    name = str(data.get("name", file_path.stem)).strip() or file_path.stem

    if not topic:
        raise ValueError(f"Instance file {file_path} is missing a non-empty 'topic'.")
    if not isinstance(literature, list) or not literature:
        raise ValueError(f"Instance file {file_path} must contain a non-empty 'literature' list.")

    normalized_literature = [str(item).strip() for item in literature if str(item).strip()]
    if not normalized_literature:
        raise ValueError(f"Instance file {file_path} contains no usable literature strings.")

    return {
        "name": name,
        "topic": topic,
        "literature": normalized_literature,
        "metadata": data.get("metadata", {}),
        "source_path": str(file_path),
    }


def build_run_summary(graph: IdeaGraph, instance_name: str, source_path: str) -> dict[str, object]:
    final_proposal = {
        "problem": graph.final_proposal.problem if graph.final_proposal else "",
        "hypothesis": graph.final_proposal.hypothesis if graph.final_proposal else "",
        "method": graph.final_proposal.method if graph.final_proposal else "",
        "evaluation": graph.final_proposal.evaluation if graph.final_proposal else "",
        "significance": graph.final_proposal.significance if graph.final_proposal else "",
        "caveats": graph.final_proposal.caveats if graph.final_proposal else "",
    }
    return {
        "instance_name": instance_name,
        "source_path": source_path,
        "topic": graph.topic,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "branch_count": len(graph.branches),
        "action_count": len(graph.actions),
        "matured_at_round": graph.matured_at_round,
        "rounds": [
            {
                "round": round_name,
                "support_coverage": snapshot.support_coverage,
                "unresolved_contradiction_ratio": snapshot.unresolved_contradiction_ratio,
                "utility": snapshot.utility,
                "utility_stable": snapshot.utility_stable,
                "completeness": snapshot.completeness,
                "is_mature": snapshot.is_mature,
            }
            for round_name, snapshot in graph.round_summaries
        ],
        "final_proposal": final_proposal,
    }


def write_run_artifacts(
    graph: IdeaGraph,
    *,
    output_root: str | Path,
    instance_name: str,
    source_path: str,
) -> Path:
    output_root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{timestamp}-{slugify(instance_name)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    graph_payload = graph_as_dict(graph)
    summary_payload = build_run_summary(graph, instance_name=instance_name, source_path=source_path)

    (run_dir / "graph.json").write_text(
        json.dumps(graph_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    final_proposal_lines = [
        f"# {instance_name}",
        "",
        "## Problem",
        summary_payload["final_proposal"]["problem"],
        "",
        "## Hypothesis",
        summary_payload["final_proposal"]["hypothesis"],
        "",
        "## Method",
        summary_payload["final_proposal"]["method"],
        "",
        "## Evaluation",
        summary_payload["final_proposal"]["evaluation"],
        "",
        "## Significance",
        summary_payload["final_proposal"]["significance"],
        "",
        "## Caveats",
        summary_payload["final_proposal"]["caveats"],
        "",
    ]
    (run_dir / "final_proposal.md").write_text("\n".join(final_proposal_lines), encoding="utf-8")

    return run_dir
