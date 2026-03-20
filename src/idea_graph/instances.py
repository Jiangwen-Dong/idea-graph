from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentInstance:
    name: str
    topic: str
    literature: list[str]
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        default_name: str = "instance",
        source_path: str = "",
    ) -> "ExperimentInstance":
        name = str(payload.get("name", default_name)).strip() or default_name
        topic = str(payload.get("topic", "")).strip()
        literature_payload = payload.get("literature", [])

        if not topic:
            raise ValueError("Experiment instance requires a non-empty topic.")
        if not isinstance(literature_payload, list) or not literature_payload:
            raise ValueError("Experiment instance requires a non-empty literature list.")

        literature = [str(item).strip() for item in literature_payload if str(item).strip()]
        if not literature:
            raise ValueError("Experiment instance contains no usable literature strings.")

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        instance_source_path = str(payload.get("source_path", source_path)).strip() or source_path
        return cls(
            name=name,
            topic=topic,
            literature=literature,
            source_path=instance_source_path,
            metadata=dict(metadata),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ExperimentInstance":
        file_path = Path(path)
        import json

        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Instance file {file_path} must contain a JSON object.")
        return cls.from_mapping(
            payload,
            default_name=file_path.stem,
            source_path=str(file_path),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
