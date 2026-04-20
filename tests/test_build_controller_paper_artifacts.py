from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_controller_paper_artifacts import (
    build_controller_behavior_exports,
    load_case_study_data,
    render_case_study_drawing_note,
    render_human_eval_instruction_packet,
    summarize_controller_behavior,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_case_study_data_extracts_baseline_and_eig_details(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    eig_dir = tmp_path / "eig"
    baseline_dir.mkdir()
    eig_dir.mkdir()
    (baseline_dir / "final_proposal.md").write_text(
        "# Mechanism-Aware Modeling of Parasite Host-Pathogen Interactions\n\n"
        "## Proposed Method\nUse a generic mechanism-aware predictor.",
        encoding="utf-8",
    )
    (eig_dir / "final_proposal.md").write_text(
        "# Mechanism-Aware Modeling of Parasite Life Cycles\n\n"
        "## Proposed Method\nUse a temporal graph neural network over parasite life stages.",
        encoding="utf-8",
    )
    (eig_dir / "summary.json").write_text(
        json.dumps({"executed_round_count": 3, "node_count": 20, "edge_count": 35, "action_count": 9}),
        encoding="utf-8",
    )
    (eig_dir / "graph.json").write_text(
        json.dumps(
            {
                "nodes": {
                    "N001": {"type": "Hypothesis", "text": "Dynamic host-parasite graph."},
                    "N020": {"type": "Repair", "text": "Integrate parasite life-stage knowledge."},
                },
                "actions": [
                    {"round_name": "Round1", "role": "NoveltyExaminer", "kind": "request_evidence"},
                    {"round_name": "Round3", "role": "MechanismProposer", "kind": "propose_repair"},
                ],
            }
        ),
        encoding="utf-8",
    )
    rows_path = tmp_path / "selected_rows.csv"
    _write_csv(
        rows_path,
        [
            {
                "instance_name": "liveideabench-parasites-302",
                "baseline_name": "self-refine",
                "overall_score": "6.93",
                "native_average_normalized_10": "7.67",
                "title": "Mechanism-Aware Modeling of Parasite Host-Pathogen Interactions",
                "run_dir": str(baseline_dir),
            },
            {
                "instance_name": "liveideabench-parasites-302",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "overall_score": "7.37",
                "native_average_normalized_10": "7.05",
                "title": "Mechanism-Aware Modeling of Parasite Life Cycles",
                "run_dir": str(eig_dir),
            },
        ],
    )

    data = load_case_study_data(rows_path, root=tmp_path)

    assert data["baseline_title"] == "Mechanism-Aware Modeling of Parasite Host-Pathogen Interactions"
    assert data["eig_title"] == "Mechanism-Aware Modeling of Parasite Life Cycles"
    assert data["eig_rounds"] == 3
    assert data["eig_node_count"] == 20
    assert "request evidence" in data["action_path"]
    assert "propose repair" in data["action_path"]


def test_summarize_controller_behavior_counts_rounds_and_quality_points() -> None:
    rows = [
        {"baseline_name": "ours-eig-critic-graph-twohead", "executed_round_count": "3", "overall_score": "6.0"},
        {"baseline_name": "ours-eig-critic-graph-twohead", "executed_round_count": "5", "overall_score": "6.4"},
        {"baseline_name": "ours-eig-fixed-control", "executed_round_count": "5", "overall_score": "5.8"},
        {"baseline_name": "ours-eig-random-control", "executed_round_count": "5", "overall_score": "5.9"},
        {"baseline_name": "ours-eig-critic-no-commit", "executed_round_count": "5", "overall_score": "6.1"},
    ]

    summary = summarize_controller_behavior(rows)

    assert summary.stop_counts["EIG (Ours)"][3] == 1
    assert summary.stop_counts["EIG (Ours)"][5] == 1
    assert summary.stop_counts["EIG-Fixed"][5] == 1
    assert summary.method_counts["EIG-NoCommit"] == 1
    assert len(summary.quality_points) == 5


def test_render_human_eval_instruction_packet_mentions_subset_blinding_and_rubric() -> None:
    packet = render_human_eval_instruction_packet()

    assert "small balanced subset" in packet
    assert "anonymized" in packet
    assert "novelty" in packet
    assert "significance" in packet
    assert "feasibility" in packet
    assert "context adherence" in packet


def test_render_case_study_drawing_note_contains_titles_and_flow() -> None:
    note = render_case_study_drawing_note(
        {
            "instance_name": "liveideabench-parasites-302",
            "baseline_title": "Mechanism-Aware Modeling of Parasite Host-Pathogen Interactions",
            "baseline_score": 6.93,
            "baseline_method_text": "Generic mechanism-aware predictor.",
            "eig_title": "Mechanism-Aware Modeling of Parasite Life Cycles",
            "eig_score": 7.37,
            "eig_method_text": "Temporal graph neural network over parasite life stages.",
            "eig_eval_text": "Out-of-distribution life-stage ablations.",
            "action_path": "- Round1: request evidence\n- Round2: attach evidence\n- Round3: propose repair",
            "repair_text": "Integrate explicit biological knowledge of parasite life stages.",
        }
    )

    assert "Suggested figure flow" in note
    assert "Parasite Life Cycles" in note
    assert "request evidence" in note
    assert "life stages" in note


def test_build_controller_behavior_exports_returns_distribution_and_points() -> None:
    summary = summarize_controller_behavior(
        [
            {"baseline_name": "ours-eig-critic-graph-twohead", "executed_round_count": "3", "overall_score": "6.0"},
            {"baseline_name": "ours-eig-critic-graph-twohead", "executed_round_count": "5", "overall_score": "6.4"},
            {"baseline_name": "ours-eig-fixed-control", "executed_round_count": "5", "overall_score": "5.8"},
        ]
    )

    distribution_rows, point_rows = build_controller_behavior_exports(summary)

    assert distribution_rows[0]["method"] == "EIG (Ours)"
    assert any(row["round"] == 3 and row["count"] == 1 for row in distribution_rows)
    assert len(point_rows) == 3
    assert point_rows[0]["method"] == "EIG (Ours)"
