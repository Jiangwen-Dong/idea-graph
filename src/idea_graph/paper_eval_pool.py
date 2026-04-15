from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

from .critic_pool_expansion import make_group_id


def select_paper_eval_candidates(
    *,
    aiib_metadata: Path | str,
    live_csv: Path | str,
    target_aiib: int,
    target_live: int,
    blocked_candidate_files: Sequence[Path | str] | None = None,
    blocked_split_registries: Sequence[Path | str] | None = None,
) -> list[dict[str, object]]:
    blocked_candidate_files = blocked_candidate_files or []
    blocked_split_registries = blocked_split_registries or []
    blocked_ids = load_blocked_group_ids(
        blocked_split_registries=blocked_split_registries,
        blocked_candidate_files=blocked_candidate_files,
    )

    aiib_rows = _aiib_candidate_rows(
        metadata_path=Path(aiib_metadata),
        blocked_group_ids=blocked_ids,
        target=target_aiib,
    )
    live_rows = _live_candidate_rows(
        csv_path=Path(live_csv),
        blocked_group_ids=blocked_ids,
        target=target_live,
    )

    combined = aiib_rows + live_rows
    combined.sort(key=lambda row: (row["benchmark"], row["instance_name"]))
    return combined


def _aiib_candidate_rows(
    *,
    metadata_path: Path,
    blocked_group_ids: set[str],
    target: int,
) -> list[dict[str, object]]:
    if target <= 0:
        return []

    raw = _read_json_array(metadata_path)
    rows: list[dict[str, object]] = []
    for entry in raw:
        if len(rows) >= target:
            break
        index = entry.get("index")
        if index is None:
            continue
        instance_name = f"ai-idea-bench-2025-{int(index)}"
        group_id = make_group_id("AI_Idea_Bench_2025", instance_name)
        if group_id in blocked_group_ids:
            continue
        notes = _aiib_notes(entry)
        rows.append(
            {
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": instance_name,
                "status": "frozen",
                "intended_role": "paper_eval",
                "notes": notes,
            }
        )

    if len(rows) < target:
        raise ValueError(
            f"Unable to select {target} AI_Idea_Bench_2025 candidates; only "
            f"{len(rows)} unblocked rows available."
        )
    return rows


def _aiib_notes(entry: dict[str, object]) -> str:
    summary = entry.get("summary") or {}
    topic = summary.get("revised_topic") or summary.get("topic") or "an unknown topic"
    return f"Frozen AIIB candidate; topic is {topic}."


def _live_candidate_rows(
    *,
    csv_path: Path,
    blocked_group_ids: set[str],
    target: int,
) -> list[dict[str, object]]:
    if target <= 0:
        return []

    rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, record in enumerate(reader):
            if len(rows) >= target:
                break
            keyword = str(record.get("keywords") or "").strip()
            if not keyword:
                continue
            instance_name = f"liveideabench-{keyword}-{row_index}"
            group_id = make_group_id("liveideabench", instance_name)
            if group_id in blocked_group_ids:
                continue
            rows.append(
                {
                    "benchmark": "liveideabench",
                    "instance_name": instance_name,
                    "status": "frozen",
                    "intended_role": "paper_eval",
                    "notes": f"Frozen LiveIdeaBench candidate; keyword is {keyword}.",
                }
            )

    if len(rows) < target:
        raise ValueError(
            f"Unable to select {target} liveideabench candidates; only "
            f"{len(rows)} unblocked rows available."
        )

    return rows


def load_blocked_group_ids(
    *,
    blocked_split_registries: Sequence[Path | str],
    blocked_candidate_files: Sequence[Path | str],
) -> set[str]:
    group_ids: set[str] = set()

    for registry_path in blocked_split_registries:
        for line_index, raw_line in enumerate(Path(registry_path).read_text(encoding="utf-8").splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{registry_path} line {line_index} must be a JSON object.")
            group_ids.update(_group_ids_from_row(payload))

    for candidate_path in blocked_candidate_files:
        rows = _read_json_array(Path(candidate_path))
        for row_index, row in enumerate(rows, start=1):
            group_ids.update(_group_ids_from_row(row))

    return group_ids


def _read_json_array(path: Path) -> list[dict[str, object]]:
    content = Path(path).read_text(encoding="utf-8")
    payload = json.loads(content)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array.")
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"{path} entry {index} must be a JSON object.")
        normalized.append(row)
    return normalized


def _group_ids_from_row(row: dict[str, object]) -> set[str]:
    candidate_ids: set[str] = set()
    group_id = str(row.get("group_id") or "").strip()
    if group_id:
        candidate_ids.add(group_id)
    benchmark = str(row.get("benchmark") or "").strip()
    instance_name = str(row.get("instance_name") or "").strip()
    if benchmark and instance_name:
        candidate_ids.add(make_group_id(benchmark, instance_name))
    if not candidate_ids:
        raise ValueError("Blocked row must include either group_id or benchmark+instance_name.")
    return candidate_ids
