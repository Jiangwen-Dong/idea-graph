from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_paper_batch_artifacts_main_table_methods_include_external_bridges() -> None:
    module = _load_script_module(
        "build_paper_batch_artifacts",
        "scripts/build_paper_batch_artifacts.py",
    )

    assert module.METHOD_ORDER == [
        "direct",
        "self-refine",
        "ai-researcher",
        "scipip",
        "virsci",
        "ours-eig",
    ]
    assert module.METHOD_DISPLAY_NAMES["ai-researcher"] == r"\textsc{AI-Researcher}"
    assert module.METHOD_DISPLAY_NAMES["scipip"] == r"\textsc{SciPIP}"
    assert module.METHOD_DISPLAY_NAMES["virsci"] == r"\textsc{VirSci}"


def test_build_paper_pilot_artifacts_display_names_cover_external_bridges() -> None:
    module = _load_script_module(
        "build_paper_pilot_artifacts",
        "scripts/build_paper_pilot_artifacts.py",
    )

    assert module.METHOD_DISPLAY_NAMES["ai-researcher"] == r"\textsc{AI-Researcher}"
    assert module.METHOD_DISPLAY_NAMES["scipip"] == r"\textsc{SciPIP}"
    assert module.METHOD_DISPLAY_NAMES["virsci"] == r"\textsc{VirSci}"
