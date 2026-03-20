from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any
import zipfile

from ..instances import ExperimentInstance
from .common import download_file, write_json


BENCHMARK_NAME = "AI_Idea_Bench_2025"
README_URL = "https://raw.githubusercontent.com/yansheng-qiu/AI_Idea_Bench_2025/main/README.md"
METADATA_URL = (
    "https://huggingface.co/datasets/yanshengqiu/AI_Idea_Bench_2025/resolve/main/"
    "target_paper_data.json?download=true"
)
PAPERS_URL = (
    "https://huggingface.co/datasets/yanshengqiu/AI_Idea_Bench_2025/resolve/main/"
    "Idea_bench_data.zip?download=true"
)
METADATA_FILENAME = "target_paper_data.json"
PAPERS_ARCHIVE_FILENAME = "Idea_bench_data.zip"
README_FILENAME = "README.AI_Idea_Bench_2025.md"
MANIFEST_FILENAME = "manifest.json"
PAPERS_DIRNAME = "papers_data"


@dataclass(frozen=True)
class AIIdeaBench2025Paths:
    root: Path
    metadata_path: Path
    readme_path: Path
    manifest_path: Path
    papers_archive_path: Path
    papers_dir: Path


@dataclass(frozen=True)
class AIIdeaBench2025Record:
    benchmark_index: int
    topic: str
    revised_topic: str
    target_paper: str
    target_paper_path: str
    motivation: str
    method_summary: str
    reference_titles: list[str]
    reference_local_paths: list[str]
    raw_record: dict[str, Any]


def default_paths(root: str | Path) -> AIIdeaBench2025Paths:
    root_path = Path(root)
    return AIIdeaBench2025Paths(
        root=root_path,
        metadata_path=root_path / METADATA_FILENAME,
        readme_path=root_path / README_FILENAME,
        manifest_path=root_path / MANIFEST_FILENAME,
        papers_archive_path=root_path / PAPERS_ARCHIVE_FILENAME,
        papers_dir=root_path / PAPERS_DIRNAME,
    )


def _write_manifest(paths: AIIdeaBench2025Paths, *, include_papers: bool) -> None:
    payload = {
        "benchmark": BENCHMARK_NAME,
        "metadata_source": METADATA_URL,
        "papers_source": PAPERS_URL,
        "readme_source": README_URL,
        "metadata_path": str(paths.metadata_path),
        "papers_archive_path": str(paths.papers_archive_path) if include_papers else None,
        "papers_dir": str(paths.papers_dir),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_papers": include_papers,
        "papers_extracted": paths.papers_dir.exists(),
    }
    write_json(paths.manifest_path, payload)


def download_ai_idea_bench_2025(
    root: str | Path,
    *,
    include_papers: bool = False,
    extract_papers: bool = False,
    force: bool = False,
) -> AIIdeaBench2025Paths:
    paths = default_paths(root)
    paths.root.mkdir(parents=True, exist_ok=True)

    if force or not paths.readme_path.exists():
        download_file(README_URL, paths.readme_path)

    if force or not paths.metadata_path.exists():
        download_file(METADATA_URL, paths.metadata_path)

    if include_papers and (force or not paths.papers_archive_path.exists()):
        download_file(PAPERS_URL, paths.papers_archive_path)

    if include_papers and extract_papers:
        extract_ai_idea_bench_2025_papers(paths, force=force)

    _write_manifest(paths, include_papers=include_papers)
    return paths


def extract_ai_idea_bench_2025_papers(
    paths_or_root: AIIdeaBench2025Paths | str | Path,
    *,
    force: bool = False,
) -> AIIdeaBench2025Paths:
    paths = (
        paths_or_root
        if isinstance(paths_or_root, AIIdeaBench2025Paths)
        else default_paths(paths_or_root)
    )
    if not paths.papers_archive_path.exists():
        raise FileNotFoundError(
            f"Papers archive not found at {paths.papers_archive_path}. Download it first."
        )

    if paths.papers_dir.exists() and not force:
        return paths

    if force and paths.papers_dir.exists():
        shutil.rmtree(paths.papers_dir)

    with zipfile.ZipFile(paths.papers_archive_path, "r") as archive:
        archive.extractall(paths.root)

    _write_manifest(paths, include_papers=True)
    return paths


def _normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return [row for row in payload["data"] if isinstance(row, dict)]
        if "test" in payload and isinstance(payload["test"], list):
            return [row for row in payload["test"] if isinstance(row, dict)]

        rows: list[dict[str, Any]] = []
        for value in payload.values():
            if isinstance(value, list):
                rows.extend(row for row in value if isinstance(row, dict))
        if rows:
            return rows

    raise ValueError("Unsupported AI Idea Bench 2025 metadata payload format.")


def _record_index(row: dict[str, Any], fallback_index: int) -> int:
    raw_index = row.get("index", fallback_index)
    try:
        return int(raw_index)
    except (TypeError, ValueError):
        return fallback_index


def _summary_value(summary: dict[str, Any], key: str) -> str:
    value = summary.get(key, "")
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _nested_value(payload: dict[str, Any], *keys: str) -> str:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    if isinstance(current, str):
        return current.strip()
    return str(current).strip()


def _list_of_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for item in values:
        if isinstance(item, str):
            value = item.strip()
        else:
            value = str(item).strip()
        if value:
            normalized.append(value)
    return normalized


def _top_reference_titles_and_paths(top_references: Any) -> tuple[list[str], list[str]]:
    if isinstance(top_references, list):
        titles: list[str] = []
        paths: list[str] = []
        for item in top_references:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            paper_local_path = str(item.get("paper_local_path", "")).strip()
            if title:
                titles.append(title)
                paths.append(paper_local_path)
        return titles, paths

    if isinstance(top_references, dict):
        return (
            _list_of_strings(top_references.get("title", [])),
            _list_of_strings(top_references.get("paper_local_path", [])),
        )

    return [], []


def load_ai_idea_bench_2025_records(root: str | Path) -> list[AIIdeaBench2025Record]:
    paths = default_paths(root)
    if not paths.metadata_path.exists():
        raise FileNotFoundError(
            f"Benchmark metadata not found at {paths.metadata_path}. "
            f"Run the downloader first."
        )

    payload = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    rows = _normalize_payload(payload)

    records: list[AIIdeaBench2025Record] = []
    for fallback_index, row in enumerate(rows):
        summary = row.get("summary", {}) if isinstance(row.get("summary", {}), dict) else {}
        find_cite = row.get("find_cite", {}) if isinstance(row.get("find_cite", {}), dict) else {}
        top_references = find_cite.get("top_references", [])
        reference_titles, reference_local_paths = _top_reference_titles_and_paths(top_references)

        record = AIIdeaBench2025Record(
            benchmark_index=_record_index(row, fallback_index),
            topic=_summary_value(summary, "topic"),
            revised_topic=_summary_value(summary, "revised_topic"),
            target_paper=str(row.get("target_paper", "")).strip(),
            target_paper_path=str(row.get("target_paper_path", "")).strip(),
            motivation=_summary_value(summary, "motivation"),
            method_summary=_nested_value(summary, "method", "targeted_designs_summary"),
            reference_titles=reference_titles,
            reference_local_paths=reference_local_paths,
            raw_record=row,
        )
        records.append(record)

    return sorted(records, key=lambda record: record.benchmark_index)


def get_ai_idea_bench_2025_record(root: str | Path, benchmark_index: int) -> AIIdeaBench2025Record:
    records = load_ai_idea_bench_2025_records(root)
    for record in records:
        if record.benchmark_index == benchmark_index:
            return record
    if 0 <= benchmark_index < len(records):
        return records[benchmark_index]
    raise KeyError(
        f"Benchmark index {benchmark_index} was not found in {BENCHMARK_NAME}, "
        f"and it is also outside the available zero-based row range 0..{len(records) - 1}."
    )


def _resolve_local_paper_path(root: Path, raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/").strip()
    if not normalized:
        return ""

    prefixes = ("./papers_data/", "papers_data/")
    for prefix in prefixes:
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix):]
            return str((root / PAPERS_DIRNAME / suffix).resolve())

    if normalized.startswith("./"):
        normalized = normalized[2:]
    return str((root / normalized).resolve())


def ai_idea_bench_2025_instance_from_record(
    record: AIIdeaBench2025Record,
    *,
    benchmark_root: str | Path,
) -> ExperimentInstance:
    root = Path(benchmark_root)
    literature: list[str] = []
    resolved_paths: list[str] = []

    for idx, title in enumerate(record.reference_titles):
        raw_path = record.reference_local_paths[idx] if idx < len(record.reference_local_paths) else ""
        resolved_path = _resolve_local_paper_path(root, raw_path) if raw_path else ""
        resolved_paths.append(resolved_path)

        parts = [title]
        if resolved_path:
            parts.append(f"local_path={resolved_path}")
        literature.append(" | ".join(parts))

    topic = record.revised_topic or record.topic or record.target_paper

    return ExperimentInstance(
        name=f"ai-idea-bench-2025-{record.benchmark_index}",
        topic=topic,
        literature=literature,
        source_path=str(default_paths(root).metadata_path),
        metadata={
            "benchmark": BENCHMARK_NAME,
            "benchmark_index": record.benchmark_index,
            "target_paper": record.target_paper,
            "target_paper_path": _resolve_local_paper_path(root, record.target_paper_path)
            if record.target_paper_path
            else "",
            "motivation": record.motivation,
            "method_summary": record.method_summary,
            "reference_titles": record.reference_titles,
            "reference_local_paths": resolved_paths,
            "raw_record": record.raw_record,
        },
    )


def record_to_dict(record: AIIdeaBench2025Record) -> dict[str, Any]:
    return asdict(record)
