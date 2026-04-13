from __future__ import annotations

from typing import Any, Mapping, Sequence


_ROLE_TO_SOURCE_SPLIT = {
    "critic_train": "train",
    "critic_dev": "validation",
}


def _normalize_str(value: object, *, field_name: str) -> str:
    normalized = str(value if value is not None else "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _validate_group_component(value: str, *, field_name: str) -> str:
    if "::" in value:
        raise ValueError(f"{field_name} must not contain '::'.")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} must not contain newline characters.")
    return value


def make_group_id(benchmark: str, instance_name: str) -> str:
    normalized_benchmark = _validate_group_component(
        _normalize_str(benchmark, field_name="benchmark"),
        field_name="benchmark",
    )
    normalized_instance_name = _validate_group_component(
        _normalize_str(instance_name, field_name="instance_name"),
        field_name="instance_name",
    )
    return f"{normalized_benchmark}::{normalized_instance_name}"


def build_expansion_partition_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    blocked_group_ids: set[str],
) -> list[dict[str, object]]:
    normalized_blocked_group_ids = {
        str(group_id).strip()
        for group_id in blocked_group_ids
        if str(group_id).strip()
    }
    rows_by_group: dict[str, dict[str, object]] = {}

    for candidate in candidates:
        benchmark = _normalize_str(candidate.get("benchmark"), field_name="benchmark")
        instance_name = _normalize_str(candidate.get("instance_name"), field_name="instance_name")
        partition_role = _normalize_str(candidate.get("partition_role"), field_name="partition_role")
        if partition_role not in _ROLE_TO_SOURCE_SPLIT:
            raise ValueError(
                f"Unsupported partition_role {partition_role!r}. "
                "Expected one of: critic_train, critic_dev."
            )

        group_id = make_group_id(benchmark, instance_name)
        if group_id in rows_by_group:
            raise ValueError(f"Duplicate candidate group_id: {group_id}")
        if group_id in normalized_blocked_group_ids:
            raise ValueError(f"Candidate group_id overlaps blocked group: {group_id}")

        rows_by_group[group_id] = {
            "group_id": group_id,
            "benchmark": benchmark,
            "instance_name": instance_name,
            "partition_role": partition_role,
            "source_split": _ROLE_TO_SOURCE_SPLIT[partition_role],
        }

    return sorted(
        rows_by_group.values(),
        key=lambda row: (
            str(row["benchmark"]),
            str(row["instance_name"]),
            str(row["group_id"]),
        ),
    )
