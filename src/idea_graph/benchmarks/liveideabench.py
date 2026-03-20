from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
from pathlib import Path
from typing import Any

from ..instances import ExperimentInstance
from .common import download_file, write_json


BENCHMARK_NAME = "liveideabench"
DATASET_URL = (
    "https://huggingface.co/datasets/6cf/liveideabench/resolve/main/"
    "liveideabench_hf.csv?download=true"
)
README_URL = "https://huggingface.co/datasets/6cf/liveideabench/raw/main/README.md"
CSV_FILENAME = "liveideabench_hf.csv"
README_FILENAME = "README.liveideabench.md"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class LiveIdeaBenchPaths:
    root: Path
    csv_path: Path
    readme_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class LiveIdeaBenchRecord:
    row_index: int
    keyword: str
    idea_model: str
    critic_model: str
    idea: str
    full_response: str
    raw_critique: str
    parsed_score: str
    originality: float | None
    feasibility: float | None
    fluency: float | None
    average_score: float | None
    raw_record: dict[str, Any]


def default_paths(root: str | Path) -> LiveIdeaBenchPaths:
    root_path = Path(root)
    return LiveIdeaBenchPaths(
        root=root_path,
        csv_path=root_path / CSV_FILENAME,
        readme_path=root_path / README_FILENAME,
        manifest_path=root_path / MANIFEST_FILENAME,
    )


def _write_manifest(paths: LiveIdeaBenchPaths) -> None:
    payload = {
        "benchmark": BENCHMARK_NAME,
        "dataset_source": DATASET_URL,
        "readme_source": README_URL,
        "csv_path": str(paths.csv_path),
        "readme_path": str(paths.readme_path),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths.manifest_path, payload)


def download_liveideabench(root: str | Path, *, force: bool = False) -> LiveIdeaBenchPaths:
    paths = default_paths(root)
    paths.root.mkdir(parents=True, exist_ok=True)

    if force or not paths.readme_path.exists():
        download_file(README_URL, paths.readme_path)

    if force or not paths.csv_path.exists():
        download_file(DATASET_URL, paths.csv_path)

    _write_manifest(paths)
    return paths


def _optional_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_liveideabench_records(root: str | Path) -> list[LiveIdeaBenchRecord]:
    paths = default_paths(root)
    if not paths.csv_path.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found at {paths.csv_path}. Run the downloader first."
        )

    records: list[LiveIdeaBenchRecord] = []
    with paths.csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            record = LiveIdeaBenchRecord(
                row_index=row_index,
                keyword=str(row.get("keywords", "")).strip(),
                idea_model=str(row.get("idea_model", "")).strip(),
                critic_model=str(row.get("critic_model", "")).strip(),
                idea=str(row.get("idea", "")).strip(),
                full_response=str(row.get("full_response", "")).strip(),
                raw_critique=str(row.get("raw_critique", "")).strip(),
                parsed_score=str(row.get("parsed_score", "")).strip(),
                originality=_optional_float(row.get("orig")),
                feasibility=_optional_float(row.get("feas")),
                fluency=_optional_float(row.get("flue")),
                average_score=_optional_float(row.get("avg")),
                raw_record=dict(row),
            )
            records.append(record)
    return records


def get_liveideabench_record(
    root: str | Path,
    row_index: int = 0,
    *,
    keyword: str | None = None,
) -> LiveIdeaBenchRecord:
    records = load_liveideabench_records(root)
    if keyword:
        normalized = keyword.strip().lower()
        filtered = [record for record in records if record.keyword.lower() == normalized]
        if not filtered:
            raise KeyError(f"No liveideabench rows found for keyword '{keyword}'.")
        if not (0 <= row_index < len(filtered)):
            raise KeyError(
                f"Keyword '{keyword}' has {len(filtered)} matching rows; "
                f"requested row offset {row_index} is out of range."
            )
        return filtered[row_index]

    if not (0 <= row_index < len(records)):
        raise KeyError(
            f"Row index {row_index} is outside the available range 0..{len(records) - 1}."
        )
    return records[row_index]


def liveideabench_instance_from_record(
    record: LiveIdeaBenchRecord,
    *,
    benchmark_root: str | Path,
) -> ExperimentInstance:
    keyword = record.keyword or "unknown-topic"
    topic = f"Ideation topic keyword: {keyword}"
    literature = [
        f"Benchmark keyword: {keyword}",
        "This benchmark row provides a keyword prompt rather than retrieved literature.",
        "Use the keyword as the ideation seed and treat benchmark idea text only as held-out metadata.",
    ]

    return ExperimentInstance(
        name=f"liveideabench-{keyword}-{record.row_index}",
        topic=topic,
        literature=literature,
        source_path=str(default_paths(benchmark_root).csv_path),
        metadata={
            "benchmark": BENCHMARK_NAME,
            "row_index": record.row_index,
            "keyword": record.keyword,
            "idea_model": record.idea_model,
            "critic_model": record.critic_model,
            "idea": record.idea,
            "full_response": record.full_response,
            "raw_critique": record.raw_critique,
            "parsed_score": record.parsed_score,
            "originality": record.originality,
            "feasibility": record.feasibility,
            "fluency": record.fluency,
            "average_score": record.average_score,
            "raw_record": record.raw_record,
        },
    )


def record_to_dict(record: LiveIdeaBenchRecord) -> dict[str, Any]:
    return asdict(record)
