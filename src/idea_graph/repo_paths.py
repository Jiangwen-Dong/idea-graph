from __future__ import annotations

from pathlib import Path


def shared_repo_root_from_worktree(root: Path) -> Path | None:
    candidate = Path(root)
    if candidate.parent.name != ".worktrees":
        return None
    return candidate.parent.parent


def effective_repo_root(root: Path) -> Path:
    shared_root = shared_repo_root_from_worktree(root)
    if shared_root is not None:
        return shared_root
    return Path(root)


def default_benchmark_root(root: Path) -> Path:
    return effective_repo_root(root) / "data" / "benchmarks"


def default_ai_benchmark_root(root: Path) -> Path:
    return default_benchmark_root(root) / "ai_idea_bench_2025"


def default_live_benchmark_root(root: Path) -> Path:
    return default_benchmark_root(root) / "liveideabench"
