from __future__ import annotations

import json
from pathlib import Path

import pytest

from idea_graph.split_eval_selection import (
    load_split_registry_rows,
    select_balanced_split_rows,
    shard_split_rows,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_select_balanced_split_rows_is_deterministic_and_balanced(tmp_path: Path) -> None:
    registry = tmp_path / "paper_eval.jsonl"
    _write_jsonl(
        registry,
        [
            {
                "group_id": "AI_Idea_Bench_2025::a1",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-1",
                "partition_role": "paper_eval",
                "source_split": "frozen",
            },
            {
                "group_id": "AI_Idea_Bench_2025::a2",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-2",
                "partition_role": "paper_eval",
                "source_split": "frozen",
            },
            {
                "group_id": "liveideabench::l1",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-robotics-1",
                "partition_role": "paper_eval",
                "source_split": "frozen",
            },
            {
                "group_id": "liveideabench::l2",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-robotics-2",
                "partition_role": "paper_eval",
                "source_split": "frozen",
            },
        ],
    )

    rows = load_split_registry_rows(registry, partition_role="paper_eval")
    selected_once = select_balanced_split_rows(rows, target_aiib=2, target_live=2, seed=7)
    selected_twice = select_balanced_split_rows(rows, target_aiib=2, target_live=2, seed=7)

    assert [row["group_id"] for row in selected_once] == [
        "AI_Idea_Bench_2025::a1",
        "AI_Idea_Bench_2025::a2",
        "liveideabench::l1",
        "liveideabench::l2",
    ]
    assert selected_once == selected_twice
    assert [row["group_id"] for row in shard_split_rows(selected_once, shard_count=2, shard_index=0)] == [
        "AI_Idea_Bench_2025::a1",
        "liveideabench::l1",
    ]
    assert [row["group_id"] for row in shard_split_rows(selected_once, shard_count=2, shard_index=1)] == [
        "AI_Idea_Bench_2025::a2",
        "liveideabench::l2",
    ]


def test_select_balanced_split_rows_rejects_insufficient_benchmark_rows() -> None:
    rows = [
        {
            "group_id": "AI_Idea_Bench_2025::a1",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-1",
            "partition_role": "paper_eval",
        }
    ]

    with pytest.raises(ValueError, match="liveideabench"):
        select_balanced_split_rows(rows, target_aiib=1, target_live=1, seed=0)


def test_shard_split_rows_rejects_invalid_shard_arguments() -> None:
    rows = [
        {
            "group_id": "AI_Idea_Bench_2025::a1",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-1",
        }
    ]

    with pytest.raises(ValueError, match="shard_count"):
        shard_split_rows(rows, shard_count=0, shard_index=0)
    with pytest.raises(ValueError, match="shard_index"):
        shard_split_rows(rows, shard_count=2, shard_index=2)
