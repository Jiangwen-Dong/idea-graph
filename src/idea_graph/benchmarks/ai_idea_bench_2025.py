from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any
import zipfile
from difflib import SequenceMatcher

from ..instances import ExperimentInstance
from ..paper_grounding import (
    build_paper_snippet_from_file,
    build_paper_snippet_from_zip_entry,
    choose_best_title_match,
)
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
PAPERS_CONTAINER_DIRNAME = "Idea_bench_data"
PAPER_SNIPPETS_CACHE_DIRNAME = "paper_snippets_cache"


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


def _candidate_papers_dirs(root: Path) -> list[Path]:
    candidates = [
        root / PAPERS_DIRNAME,
        root / PAPERS_CONTAINER_DIRNAME / PAPERS_DIRNAME,
    ]
    return [path for path in candidates if path.exists()]


def _resolved_papers_dir(root: Path) -> Path:
    candidates = _candidate_papers_dirs(root)
    if candidates:
        return candidates[0]
    return root / PAPERS_DIRNAME


def _write_manifest(paths: AIIdeaBench2025Paths, *, include_papers: bool) -> None:
    resolved_papers_dir = _resolved_papers_dir(paths.root)
    payload = {
        "benchmark": BENCHMARK_NAME,
        "metadata_source": METADATA_URL,
        "papers_source": PAPERS_URL,
        "readme_source": README_URL,
        "metadata_path": str(paths.metadata_path),
        "papers_archive_path": str(paths.papers_archive_path) if include_papers else None,
        "papers_dir": str(resolved_papers_dir),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_papers": include_papers,
        "papers_extracted": resolved_papers_dir.exists(),
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

    resolved_existing_dir = _resolved_papers_dir(paths.root)
    if resolved_existing_dir.exists() and not force:
        return paths

    if force and resolved_existing_dir.exists():
        shutil.rmtree(resolved_existing_dir)

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
            for papers_dir in _candidate_papers_dirs(root):
                candidate = papers_dir / suffix
                if candidate.exists():
                    return str(candidate.resolve())
            fallback = _resolved_papers_dir(root) / suffix
            return str(fallback.resolve())

    if normalized.startswith("./"):
        normalized = normalized[2:]
    direct_candidate = root / normalized
    if direct_candidate.exists():
        return str(direct_candidate.resolve())
    return str(direct_candidate.resolve())


def _zip_pdf_entries(root: Path) -> list[str]:
    archive_path = default_paths(root).papers_archive_path
    if not archive_path.exists():
        return []
    with zipfile.ZipFile(archive_path, "r") as archive:
        return [info.filename for info in archive.infolist() if info.filename.lower().endswith(".pdf")]


def _raw_path_to_zip_entry(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/").strip()
    if not normalized:
        return ""
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith(f"{PAPERS_CONTAINER_DIRNAME}/"):
        return normalized
    if normalized.startswith(f"{PAPERS_DIRNAME}/"):
        return f"{PAPERS_CONTAINER_DIRNAME}/{normalized}"
    return normalized


def _resolve_zip_entry(root: Path, raw_path: str, *, fallback_title: str = "") -> str:
    archive_path = default_paths(root).papers_archive_path
    if not archive_path.exists():
        return ""

    desired = _raw_path_to_zip_entry(raw_path)
    entries = _zip_pdf_entries(root)
    if desired and desired in entries:
        return desired

    if desired:
        filename = Path(desired).name
        by_filename = [entry for entry in entries if Path(entry).name.casefold() == filename.casefold()]
        if by_filename:
            return by_filename[0]

    if fallback_title:
        match = choose_best_title_match(fallback_title, entries)
        if match:
            return match
    return ""


def _paper_snippets_cache_dir(root: Path) -> Path:
    return root / PAPER_SNIPPETS_CACHE_DIRNAME


def _normalize_title_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _title_tokens(text: str) -> list[str]:
    stopwords = {"the", "a", "an", "for", "of", "and", "in", "on", "to", "with", "from"}
    return [token for token in _normalize_title_key(text).split() if token and token not in stopwords]


def _archive_candidate_entries(root: Path, title: str, *, limit: int = 24) -> list[str]:
    tokens = _title_tokens(title)
    if not tokens:
        return []

    scored: list[tuple[float, str]] = []
    for entry in _zip_pdf_entries(root):
        stem = Path(entry).stem
        normalized_stem = _normalize_title_key(stem)
        entry_tokens = set(normalized_stem.split())
        overlap = sum(1 for token in tokens if token in entry_tokens)
        partial = sum(
            1
            for token in tokens
            if token not in entry_tokens and any(token in candidate or candidate in token for candidate in entry_tokens)
        )
        if overlap == 0 and partial == 0:
            continue
        score = (3.0 * overlap) + partial + SequenceMatcher(None, _normalize_title_key(title), normalized_stem).ratio()
        scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:limit]]


def _best_title_verified_zip_entry(root: Path, title: str) -> str:
    cache_dir = _paper_snippets_cache_dir(root)
    normalized_target = _normalize_title_key(title)
    if not normalized_target:
        return ""

    best_score = 0.0
    best_entry = ""
    for entry in _archive_candidate_entries(root, title):
        try:
            snippet = build_paper_snippet_from_zip_entry(
                default_paths(root).papers_archive_path,
                entry,
                fallback_title="",
                cache_dir=cache_dir,
            )
        except Exception:
            continue
        candidate_title = _normalize_title_key(snippet.raw_title or snippet.resolved_title)
        if not candidate_title:
            continue
        ratio = SequenceMatcher(None, normalized_target, candidate_title).ratio()
        token_overlap = len(set(_title_tokens(title)) & set(_title_tokens(snippet.raw_title or snippet.resolved_title)))
        score = ratio + (0.08 * token_overlap)
        if score > best_score:
            best_score = score
            best_entry = entry

    return best_entry if best_score >= 0.82 else ""


def _load_paper_snippet(root: Path, *, raw_path: str = "", fallback_title: str = "") -> dict[str, Any]:
    cache_dir = _paper_snippets_cache_dir(root)
    resolved_file_path = _resolve_local_paper_path(root, raw_path) if raw_path else ""
    if resolved_file_path and Path(resolved_file_path).exists():
        return build_paper_snippet_from_file(
            resolved_file_path,
            fallback_title=fallback_title,
            cache_dir=cache_dir,
        ).as_dict()

    zip_entry = _resolve_zip_entry(root, raw_path, fallback_title=fallback_title)
    if not zip_entry and fallback_title:
        zip_entry = _best_title_verified_zip_entry(root, fallback_title)
    if zip_entry:
        return build_paper_snippet_from_zip_entry(
            default_paths(root).papers_archive_path,
            zip_entry,
            fallback_title=fallback_title,
            cache_dir=cache_dir,
        ).as_dict()
    return {}


def _build_record_paper_grounding(record: AIIdeaBench2025Record, *, benchmark_root: Path) -> dict[str, Any]:
    target_snippet = _load_paper_snippet(
        benchmark_root,
        raw_path=record.target_paper_path,
        fallback_title=record.target_paper,
    )
    reference_snippets: list[dict[str, Any]] = []
    for idx, title in enumerate(record.reference_titles[:4]):
        raw_path = record.reference_local_paths[idx] if idx < len(record.reference_local_paths) else ""
        snippet = _load_paper_snippet(
            benchmark_root,
            raw_path=raw_path,
            fallback_title=title,
        )
        if snippet:
            reference_snippets.append(snippet)
    return {
        "target_paper_snippet": target_snippet,
        "reference_paper_snippets": reference_snippets,
        "cache_dir": str(_paper_snippets_cache_dir(benchmark_root)),
    }


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
    target_paper_path = _resolve_local_paper_path(root, record.target_paper_path) if record.target_paper_path else ""
    paper_grounding = _build_record_paper_grounding(record, benchmark_root=root)

    return ExperimentInstance(
        name=f"ai-idea-bench-2025-{record.benchmark_index}",
        topic=topic,
        literature=literature,
        source_path=str(default_paths(root).metadata_path),
        metadata={
            "benchmark": BENCHMARK_NAME,
            "benchmark_index": record.benchmark_index,
            "target_paper": record.target_paper,
            "target_paper_path": target_paper_path,
            "motivation": record.motivation,
            "method_summary": record.method_summary,
            "reference_titles": record.reference_titles,
            "reference_local_paths": resolved_paths,
            "paper_grounding": paper_grounding,
            "raw_record": record.raw_record,
        },
    )


def record_to_dict(record: AIIdeaBench2025Record) -> dict[str, Any]:
    return asdict(record)
