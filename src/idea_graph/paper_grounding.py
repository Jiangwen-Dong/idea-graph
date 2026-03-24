from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha1
import io
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

from pypdf import PdfReader

PAPER_SNIPPET_CACHE_VERSION = "v3"


@dataclass(frozen=True)
class PaperSnippet:
    source_kind: str
    source_path: str
    entry_name: str
    raw_title: str
    resolved_title: str
    abstract: str
    introduction: str
    method: str
    evaluation: str
    conclusion: str
    text_excerpt: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean_text(text).lower())


def _extract_text(reader: PdfReader, *, max_pages: int = 8, max_chars: int = 40000) -> str:
    chunks: list[str] = []
    total = 0
    for page in reader.pages[:max_pages]:
        page_text = _clean_text(page.extract_text() or "")
        if not page_text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(page_text) > remaining:
            page_text = page_text[:remaining]
        chunks.append(page_text)
        total += len(page_text)
    return "\n".join(chunks)


def _guess_title(text: str) -> str:
    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    for line in lines[:8]:
        if len(line) < 8:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if line.lower() in {"abstract", "introduction"}:
            continue
        if len(line.split()) > 2:
            return line
    return ""


def _resolve_title(raw_title: str, fallback_title: str, fallback_stem: str) -> str:
    raw_clean = _clean_text(raw_title)
    if (
        raw_clean
        and len(raw_clean) <= 180
        and len(raw_clean.split()) <= 28
        and "abstract" not in raw_clean.casefold()
        and "figure" not in raw_clean.casefold()
    ):
        return raw_clean

    fallback_clean = _clean_text(fallback_title)
    if fallback_clean:
        return fallback_clean
    return _clean_text(fallback_stem)


def _section_slice(text: str, headings: tuple[str, ...], stop_headings: tuple[str, ...], *, max_chars: int = 1600) -> str:
    lowered = text.lower()
    start_positions = []
    for heading in headings:
        idx = lowered.find(heading.lower())
        if idx != -1:
            start_positions.append(idx)
    if not start_positions:
        return ""
    start = min(start_positions)
    start_line_break = text.find("\n", start)
    if start_line_break != -1:
        start = start_line_break + 1
    end = len(text)
    for heading in stop_headings:
        idx = lowered.find(heading.lower(), start)
        if idx != -1:
            end = min(end, idx)
    snippet = _clean_text(text[start:end])
    return snippet[:max_chars].rstrip()


def _fallback_excerpt(text: str, *, max_chars: int = 1600) -> str:
    cleaned = _clean_text(text)
    return cleaned[:max_chars].rstrip()


def _cache_file(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{sha1(cache_key.encode('utf-8')).hexdigest()}.json"


def _read_cached(cache_dir: Path, cache_key: str) -> PaperSnippet | None:
    path = _cache_file(cache_dir, cache_key)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PaperSnippet(**payload)


def _write_cached(cache_dir: Path, cache_key: str, snippet: PaperSnippet) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_file(cache_dir, cache_key).write_text(
        json.dumps(snippet.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_paper_snippet_from_file(
    path: str | Path,
    *,
    fallback_title: str = "",
    cache_dir: str | Path | None = None,
) -> PaperSnippet:
    file_path = Path(path)
    cache_key = (
        f"{PAPER_SNIPPET_CACHE_VERSION}::file::{file_path.resolve()}::"
        f"{file_path.stat().st_mtime_ns}::{file_path.stat().st_size}"
    )
    cache_root = Path(cache_dir) if cache_dir is not None else None
    if cache_root is not None:
        cached = _read_cached(cache_root, cache_key)
        if cached is not None:
            return cached

    reader = PdfReader(str(file_path))
    text = _extract_text(reader)
    raw_title = _guess_title(text)
    snippet = PaperSnippet(
        source_kind="file",
        source_path=str(file_path.resolve()),
        entry_name=file_path.name,
        raw_title=raw_title,
        resolved_title=_resolve_title(raw_title, fallback_title, file_path.stem),
        abstract=_section_slice(
            text,
            headings=("abstract",),
            stop_headings=("introduction", "1 introduction", "keywords"),
        ),
        introduction=_section_slice(
            text,
            headings=("introduction", "1 introduction"),
            stop_headings=("related work", "2 related work", "background", "2 background", "method", "approach"),
        ),
        method=_section_slice(
            text,
            headings=("method", "approach", "model", "proposed method", "3 method", "2 method"),
            stop_headings=("experiment", "evaluation", "results", "4 experiments", "5 experiments"),
        ),
        evaluation=_section_slice(
            text,
            headings=("experiment", "evaluation", "results", "4 experiments", "5 experiments"),
            stop_headings=("conclusion", "discussion", "6 conclusion", "7 conclusion"),
        ),
        conclusion=_section_slice(
            text,
            headings=("conclusion", "discussion", "6 conclusion", "7 conclusion"),
            stop_headings=tuple(),
        ),
        text_excerpt=_fallback_excerpt(text, max_chars=2000),
    )
    if cache_root is not None:
        _write_cached(cache_root, cache_key, snippet)
    return snippet


def build_paper_snippet_from_zip_entry(
    archive_path: str | Path,
    entry_name: str,
    *,
    fallback_title: str = "",
    cache_dir: str | Path | None = None,
) -> PaperSnippet:
    zip_path = Path(archive_path)
    with ZipFile(zip_path, "r") as archive:
        info = archive.getinfo(entry_name)
        cache_key = (
            f"{PAPER_SNIPPET_CACHE_VERSION}::zip::{zip_path.resolve()}::"
            f"{zip_path.stat().st_mtime_ns}::{info.filename}::{info.file_size}"
        )
        cache_root = Path(cache_dir) if cache_dir is not None else None
        if cache_root is not None:
            cached = _read_cached(cache_root, cache_key)
            if cached is not None:
                return cached
        data = archive.read(entry_name)

    reader = PdfReader(io.BytesIO(data))
    text = _extract_text(reader)
    raw_title = _guess_title(text)
    snippet = PaperSnippet(
        source_kind="zip_entry",
        source_path=str(zip_path.resolve()),
        entry_name=entry_name,
        raw_title=raw_title,
        resolved_title=_resolve_title(raw_title, fallback_title, Path(entry_name).stem),
        abstract=_section_slice(
            text,
            headings=("abstract",),
            stop_headings=("introduction", "1 introduction", "keywords"),
        ),
        introduction=_section_slice(
            text,
            headings=("introduction", "1 introduction"),
            stop_headings=("related work", "2 related work", "background", "2 background", "method", "approach"),
        ),
        method=_section_slice(
            text,
            headings=("method", "approach", "model", "proposed method", "3 method", "2 method"),
            stop_headings=("experiment", "evaluation", "results", "4 experiments", "5 experiments"),
        ),
        evaluation=_section_slice(
            text,
            headings=("experiment", "evaluation", "results", "4 experiments", "5 experiments"),
            stop_headings=("conclusion", "discussion", "6 conclusion", "7 conclusion"),
        ),
        conclusion=_section_slice(
            text,
            headings=("conclusion", "discussion", "6 conclusion", "7 conclusion"),
            stop_headings=tuple(),
        ),
        text_excerpt=_fallback_excerpt(text, max_chars=2000),
    )
    if cache_root is not None:
        _write_cached(cache_root, cache_key, snippet)
    return snippet


def choose_best_title_match(title: str, candidates: list[str]) -> str:
    target_key = _normalize_key(title)
    if not target_key or not candidates:
        return ""
    exact = [candidate for candidate in candidates if _normalize_key(Path(candidate).stem) == target_key]
    if exact:
        return exact[0]
    contains = [candidate for candidate in candidates if target_key in _normalize_key(Path(candidate).stem)]
    if contains:
        return contains[0]
    reverse_contains = [candidate for candidate in candidates if _normalize_key(Path(candidate).stem) in target_key]
    if reverse_contains:
        return reverse_contains[0]
    return ""
