from __future__ import annotations

import os
from pathlib import Path


def _windows_safe_path(path: Path | str) -> str:
    raw = os.path.abspath(os.fspath(path))
    if os.name != "nt":
        return raw
    if raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw.lstrip("\\")
    return "\\\\?\\" + raw


def ensure_parent_dir(path: Path | str) -> None:
    parent = Path(path).parent
    if not str(parent):
        return
    os.makedirs(_windows_safe_path(parent), exist_ok=True)


def read_text_file(path: Path | str, *, encoding: str = "utf-8") -> str:
    with open(_windows_safe_path(path), "r", encoding=encoding) as handle:
        return handle.read()


def write_text_file(path: Path | str, content: str, *, encoding: str = "utf-8") -> None:
    ensure_parent_dir(path)
    with open(_windows_safe_path(path), "w", encoding=encoding) as handle:
        handle.write(content)
