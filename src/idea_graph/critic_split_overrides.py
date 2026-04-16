from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fs_utils import read_text_file, write_text_file

_ROLE_TO_SPLIT = {
    "critic_train": "train",
    "critic_dev": "validation",
}


def load_split_registry_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, raw_line in enumerate(read_text_file(path).splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_index} must contain a JSON object.")
        rows.append(dict(payload))
    return rows


def build_split_override_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    overrides: dict[str, str] = {}
    for row in rows:
        group_id = str(row.get("group_id", "")).strip()
        if not group_id:
            raise ValueError("split registry row is missing required group_id.")
        partition_role = str(row.get("partition_role", "")).strip()
        split = _ROLE_TO_SPLIT.get(partition_role)
        if split is None:
            continue
        existing = overrides.get(group_id)
        if existing is not None and existing != split:
            raise ValueError(f"Conflicting split override for group_id '{group_id}'.")
        overrides[group_id] = split
    return [
        {"group_id": group_id, "split": split}
        for group_id, split in sorted(overrides.items())
    ]


def write_split_override_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    write_text_file(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
    )
