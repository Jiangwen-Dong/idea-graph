from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _resolve_path(path_value: Any, *, base_dir: Path) -> Path:
    raw = _clean(path_value)
    if not raw:
        return Path(".")
    path = Path(raw)
    if path.is_absolute():
        return path
    for ancestor in (base_dir, *base_dir.parents):
        candidate = (ancestor / path).resolve()
        if candidate.exists():
            return candidate
    return (base_dir / path).resolve()


def _check_file(path: Path, issues: list[str], label: str) -> bool:
    if path.exists():
        return True
    issues.append(f"Missing {label}: {path}")
    return False


def _ai_researcher_status(entry: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    mode = _clean(entry.get("execution_mode")).lower() or "upstream"
    repo = _resolve_path(entry.get("repo_path"), base_dir=base_dir)
    ready = repo.exists()
    if not ready:
        issues.append(f"Missing AI-Researcher repo_path: {repo}")
    runner = repo / "ai_researcher"
    if ready and not runner.exists():
        ready = False
        issues.append(f"Missing AI-Researcher runner directory: {runner}")
    for script_name in ("grounded_idea_gen.py", "experiment_plan_gen.py", "tournament_ranking.py"):
        if ready:
            ready = _check_file(runner / "src" / script_name, issues, script_name) and ready
    adapter_status = "paper-faithful-adapter" if "bridge" in mode else "exact-upstream"
    return {
        "baseline": "ai-researcher",
        "enabled": bool(entry.get("enabled", True)),
        "ready": bool(ready),
        "execution_mode": mode,
        "adapter_status": adapter_status,
        "issues": issues,
    }


def _scipip_status(entry: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    repo = _resolve_path(entry.get("repo_path"), base_dir=base_dir)
    ready = repo.exists()
    if not ready:
        issues.append(f"Missing SciPIP repo_path: {repo}")
    if ready:
        ready = _check_file(repo / "src" / "generator.py", issues, "SciPIP generator.py") and ready
    config_path = _resolve_path(entry.get("config_path") or (repo / "configs" / "datasets.yaml"), base_dir=base_dir)
    if ready:
        ready = _check_file(config_path, issues, "SciPIP config_path") and ready
    return {
        "baseline": "scipip",
        "enabled": bool(entry.get("enabled", True)),
        "ready": bool(ready),
        "execution_mode": "upstream-generator",
        "adapter_status": "paper-faithful-adapter",
        "issues": issues,
    }


def _virsci_status(entry: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    repo = _resolve_path(entry.get("repo_path"), base_dir=base_dir)
    ready = repo.exists()
    if not ready:
        issues.append(f"Missing VirSci repo_path: {repo}")
    if ready:
        ready = _check_file(repo / "sci_platform" / "run.py", issues, "VirSci sci_platform/run.py") and ready
    issues.append("VirSci still requires a fixed-topic benchmark adapter before main-table eligibility.")
    return {
        "baseline": "virsci",
        "enabled": bool(entry.get("enabled", False)),
        "ready": False,
        "execution_mode": "upstream-multi-agent",
        "adapter_status": "exclude-until-fixed-topic-adapter",
        "issues": issues,
    }


def check_external_baseline_config(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    resolved_config_path = config_path.resolve()
    base_dir = resolved_config_path.parent
    baselines: list[dict[str, Any]] = []
    if isinstance(payload.get("ai-researcher"), dict):
        baselines.append(_ai_researcher_status(payload["ai-researcher"], base_dir=base_dir))
    if isinstance(payload.get("scipip"), dict):
        baselines.append(_scipip_status(payload["scipip"], base_dir=base_dir))
    if isinstance(payload.get("virsci"), dict):
        baselines.append(_virsci_status(payload["virsci"], base_dir=base_dir))
    return {
        "config_path": str(resolved_config_path),
        "baselines": baselines,
        "ready_baselines": [row["baseline"] for row in baselines if row["enabled"] and row["ready"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight external baselines for paper-faithful evaluation.")
    parser.add_argument("--external-baseline-config", type=Path, required=True)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    report = check_external_baseline_config(args.external_baseline_config)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
