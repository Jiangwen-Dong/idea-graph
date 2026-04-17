from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
from time import time
from typing import Any, Callable

from .literature_grounding import build_literature_grounding
from .models import FinalProposal, IdeaGraph
from .agent_backend import OpenAICompatibleCollaborationBackend
from .settings import OpenAICompatibleSettings

ADAPTER_STATUS_EXACT_UPSTREAM = "exact-upstream"
ADAPTER_STATUS_PAPER_FAITHFUL = "paper-faithful-adapter"
ADAPTER_STATUS_EXCLUDED = "exclude-until-fixed-topic-adapter"


def load_external_baseline_config(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"External baseline config {path} must contain a JSON object.")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            normalized[str(key)] = dict(value)
    return normalized


def run_external_baseline(
    graph: IdeaGraph,
    *,
    baseline_name: str,
    external_config: dict[str, dict[str, Any]] | None,
    progress_callback: Callable[[str], None] | None = None,
) -> FinalProposal:
    config_map = external_config or {}
    entry = config_map.get(baseline_name)
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"Baseline '{baseline_name}' requires --external-baseline-config with a matching '{baseline_name}' entry."
        )
    if not bool(entry.get("enabled", True)):
        raise RuntimeError(f"Baseline '{baseline_name}' is disabled in the external baseline config.")

    if baseline_name == "ai-researcher":
        return _run_ai_researcher(graph, entry, progress_callback)
    if baseline_name == "scipip":
        return _run_scipip(graph, entry, progress_callback)
    if baseline_name == "virsci":
        return _run_virsci(graph, entry, progress_callback)
    raise RuntimeError(f"No external baseline adapter is registered for '{baseline_name}'.")


def _emit(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "baseline-run"


def _safe_name(value: str, *, fallback_prefix: str, used: set[str]) -> str:
    base = _slugify(value) or fallback_prefix
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}-{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def _workspace_root(config: dict[str, Any]) -> Path:
    configured = _clean_text(config.get("workspace_root"))
    if configured:
        root = Path(configured)
    else:
        root = Path(".tmp-external-baseline-runs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_workspace(config: dict[str, Any], *, baseline_name: str, instance_name: str) -> Path:
    root = _workspace_root(config)
    workspace = root / f"{baseline_name}-{_slugify(instance_name)}-{int(time())}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _require_path(path_value: Any, *, label: str) -> Path:
    path = Path(_clean_text(path_value))
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env_overrides: dict[str, str] | None = None,
    timeout_seconds: int = 1800,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def _resolve_config_value(payload: Any) -> str:
    if isinstance(payload, dict):
        direct = _clean_text(payload.get("value"))
        if direct:
            return direct
        env_name = _clean_text(payload.get("env"))
        if env_name:
            return _clean_text(os.getenv(env_name))
        return ""
    text = _clean_text(payload)
    if text.startswith("env:"):
        return _clean_text(os.getenv(text[4:]))
    return text


def _resolve_env_overrides(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    resolved: dict[str, str] = {}
    for key, value in payload.items():
        env_name = _clean_text(key)
        if not env_name:
            continue
        resolved_value = _resolve_config_value(value)
        if resolved_value:
            resolved[env_name] = resolved_value
    return resolved


def _execution_mode(config: dict[str, Any], *, default: str) -> str:
    return _clean_text(config.get("execution_mode")).lower() or default


def _uses_bridge_mode(config: dict[str, Any], *, default: str) -> bool:
    return "bridge" in _execution_mode(config, default=default)


def _write_json_artifact(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_external_trace(graph: IdeaGraph, *, stage: str, trace: dict[str, object]) -> None:
    graph.metadata.setdefault("external_baseline_traces", []).append({"stage": stage, **trace})


def _bridge_packet(graph: IdeaGraph) -> dict[str, object]:
    return {
        "topic": _clean_text(graph.topic),
        "benchmark_packet": graph.metadata.get("benchmark_input_packet", {}),
        "benchmark_background": _build_benchmark_background(graph),
        "reference_packet": _reference_packet(graph),
    }


def _stamp_external_baseline_metadata(
    graph: IdeaGraph,
    *,
    execution_mode: str,
    adapter_status: str,
    proxy_fallback: bool = False,
    preserved_stages: list[str] | None = None,
) -> None:
    graph.metadata["external_baseline_execution_mode"] = execution_mode
    graph.metadata["external_baseline_adapter_status"] = adapter_status
    graph.metadata["external_baseline_proxy_fallback"] = proxy_fallback
    if preserved_stages is not None:
        graph.metadata["external_baseline_preserved_stages"] = list(preserved_stages)


@contextmanager
def _temporary_repo_keys(repo_root: Path, config: dict[str, Any]):
    keys_path = repo_root / "keys.json"
    keys_json_path = _clean_text(config.get("keys_json_path"))
    backup_text: str | None = None
    wrote_temp = False

    if keys_json_path:
        payload = json.loads(Path(keys_json_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"keys_json_path must point to a JSON object: {keys_json_path}")
    elif keys_path.exists():
        payload = None
    else:
        key_payload = config.get("keys", {})
        if not isinstance(key_payload, dict):
            key_payload = {}
        payload = {
            "api_key": _resolve_config_value(key_payload.get("api_key")),
            "organization_id": _resolve_config_value(key_payload.get("organization_id")),
            "s2_key": _resolve_config_value(key_payload.get("s2_key")),
            "anthropic_key": _resolve_config_value(key_payload.get("anthropic_key")),
        }
        if not any(payload.values()):
            raise RuntimeError(
                f"No usable keys.json source found for external baseline repo at {repo_root}. "
                "Set keys_json_path, provide a repo-local keys.json, or supply the 'keys' block in the external config."
            )

    if payload is None:
        yield keys_path
        return

    if keys_path.exists():
        backup_text = keys_path.read_text(encoding="utf-8")
    keys_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    wrote_temp = True
    try:
        yield keys_path
    finally:
        if not wrote_temp:
            return
        if backup_text is not None:
            keys_path.write_text(backup_text, encoding="utf-8")
        else:
            try:
                keys_path.unlink()
            except FileNotFoundError:
                pass


def _first_sentence(text: Any) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for separator in (". ", "! ", "? "):
        if separator in cleaned:
            return cleaned.split(separator, 1)[0].strip().rstrip(".!?") + "."
    return cleaned.rstrip(".!?") + "."


def _reference_packet(graph: IdeaGraph) -> list[dict[str, str]]:
    packet = graph.metadata.get("benchmark_input_packet", {})
    if isinstance(packet, dict):
        references = packet.get("reference_packet", [])
        if isinstance(references, list):
            normalized: list[dict[str, str]] = []
            for item in references:
                if not isinstance(item, dict):
                    continue
                title = _clean_text(item.get("title"))
                snippet = _clean_text(item.get("snippet"))
                if title or snippet:
                    normalized.append({"title": title, "snippet": snippet})
            if normalized:
                return normalized

    normalized = []
    for raw in graph.literature[:6]:
        text = _clean_text(raw)
        if not text:
            continue
        if "|" in text:
            title = _clean_text(text.split("|", 1)[0])
            snippet = _clean_text(text.split("|", 1)[1])
        else:
            title = text
            snippet = ""
        normalized.append({"title": title, "snippet": snippet})
    return normalized


def _ai_researcher_paper_cache(graph: IdeaGraph, *, limit: int) -> dict[str, object]:
    papers = []
    for index, item in enumerate(_reference_packet(graph)[:limit]):
        title = _clean_text(item.get("title"))
        abstract = _clean_text(item.get("snippet")) or title
        if not title:
            continue
        papers.append(
            {
                "paperId": f"benchmark-ref-{index}",
                "title": title,
                "abstract": abstract,
                "score": max(1, 10 - index),
            }
        )
    return {
        "topic_description": _clean_text(graph.topic),
        "paper_bank": papers,
    }


def _flatten_ai_researcher_seed_ideas(seed_payload: dict[str, Any]) -> dict[str, object]:
    topic_description = _clean_text(seed_payload.get("topic_description"))
    idea_groups = seed_payload.get("ideas", [])
    flattened: dict[str, Any] = {}
    used_names: set[str] = set()
    if isinstance(idea_groups, list):
        for group in idea_groups:
            if not isinstance(group, dict):
                continue
            for key, value in group.items():
                name = _safe_name(_clean_text(key) or "idea", fallback_prefix="idea", used=used_names)
                flattened[name] = value
    elif isinstance(idea_groups, dict):
        for key, value in idea_groups.items():
            name = _safe_name(_clean_text(key) or "idea", fallback_prefix="idea", used=used_names)
            flattened[name] = value
    return {
        "topic_description": topic_description,
        "ideas": flattened,
    }


def _flex_lookup(payload: dict[str, Any], *candidates: str) -> str:
    normalized = {_clean_text(key).casefold(): value for key, value in payload.items()}
    for candidate in candidates:
        value = normalized.get(candidate.casefold())
        if value:
            return _clean_text(value)
    return ""


def _proposal_from_ai_researcher_plan(plan_payload: dict[str, Any]) -> FinalProposal:
    full_plan = plan_payload.get("full_experiment_plan", {})
    if not isinstance(full_plan, dict):
        full_plan = {}
    raw_idea = plan_payload.get("raw_idea", {})
    if not isinstance(raw_idea, dict):
        raw_idea = {}

    title = (
        _flex_lookup(full_plan, "Title")
        or _clean_text(plan_payload.get("idea_name"))
        or _flex_lookup(raw_idea, "Title")
    )
    problem = _flex_lookup(full_plan, "Problem Statement", "Problem")
    existing_methods = _flex_lookup(raw_idea, "Existing Methods", "Existing Method")
    motivation = _flex_lookup(raw_idea, "Motivation") or _flex_lookup(full_plan, "Motivation")
    hypothesis = _flex_lookup(raw_idea, "Proposed Method", "Core Idea", "Hypothesis")
    method = _flex_lookup(full_plan, "Proposed Method")
    evaluation = _flex_lookup(
        full_plan,
        "Step-by-Step Experiment Plan",
        "Experiment Plan",
        "Experiments",
    )
    significance = _first_sentence(problem or motivation)
    caveats = _flex_lookup(full_plan, "Fallback Plan", "Risk", "Caveat")

    return FinalProposal(
        title=title,
        abstract="",
        problem=problem,
        existing_methods=existing_methods,
        motivation=motivation,
        hypothesis=hypothesis or _first_sentence(method),
        method=method,
        evaluation=evaluation,
        significance=significance,
        caveats=caveats,
    )


def _build_benchmark_background(graph: IdeaGraph) -> str:
    references = _reference_packet(graph)
    lines = [f"Research topic: {_clean_text(graph.topic)}"]
    if references:
        lines.append("Reference context:")
        for index, item in enumerate(references[:4], start=1):
            title = _clean_text(item.get("title"))
            snippet = _clean_text(item.get("snippet"))
            lines.append(f"{index}. {title}")
            if snippet:
                lines.append(f"   {snippet}")
    return "\n".join(lines).strip()


def _extract_markdown_section(text: str, headings: list[str]) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n")
    for heading in headings:
        pattern = re.compile(
            rf"(?:^|\n)\s*(?:[-*]\s*)?\*{{0,2}}{re.escape(heading)}\*{{0,2}}\s*:?\s*(.*?)(?=\n\s*(?:[-*]\s*)?\*{{0,2}}[A-Z][^\n]{{0,60}}\*{{0,2}}\s*:|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(normalized)
        if match:
            return _clean_text(match.group(1))
    return ""


def _proposal_from_scipip_text(graph: IdeaGraph, text: str) -> FinalProposal:
    grounding = build_literature_grounding(literature=graph.literature, metadata=graph.metadata)
    cleaned = _clean_text(text)
    title = _extract_markdown_section(cleaned, ["Title", "Idea 1", "Idea"]) or _first_sentence(cleaned)
    problem = _extract_markdown_section(cleaned, ["Research Problem", "Problem"]) or _clean_text(graph.topic)
    motivation = _extract_markdown_section(cleaned, ["Rationales", "Motivation"])
    method = _extract_markdown_section(cleaned, ["Method", "Implementation", "Approach", "Concept"]) or cleaned
    evaluation = _extract_markdown_section(cleaned, ["Experiment", "Experiments", "Evaluation"]) or grounding.experiment_plan_summary
    significance = _first_sentence(motivation or problem)
    return FinalProposal(
        title=title,
        abstract="",
        problem=problem,
        existing_methods=grounding.existing_methods_summary,
        motivation=motivation,
        hypothesis=_first_sentence(method),
        method=method,
        evaluation=evaluation,
        significance=significance,
        caveats="The external SciPIP wrapper output was mapped heuristically from the upstream text format.",
    )


def _proposal_from_virsci_payload(graph: IdeaGraph, payload: dict[str, Any]) -> FinalProposal:
    grounding = build_literature_grounding(literature=graph.literature, metadata=graph.metadata)
    idea = _clean_text(payload.get("idea"))
    abstract = _clean_text(payload.get("abstract"))
    topic = _clean_text(payload.get("topic")) or _clean_text(graph.topic)
    body = abstract or idea
    return FinalProposal(
        title=_first_sentence(topic),
        abstract="",
        problem=topic,
        existing_methods=grounding.existing_methods_summary,
        motivation=_first_sentence(idea),
        hypothesis=_first_sentence(idea),
        method=idea or body,
        evaluation=grounding.experiment_plan_summary,
        significance=_first_sentence(body),
        caveats="The external VirSci wrapper currently relies on the upstream team-info output and may not preserve benchmark-controlled topic input.",
    )


def _build_openai_compatible_bridge_backend(config: dict[str, Any]) -> OpenAICompatibleCollaborationBackend:
    llm_config_path = _clean_text(config.get("llm_config_path"))
    if llm_config_path:
        settings = OpenAICompatibleSettings.from_json_file(llm_config_path)
        return OpenAICompatibleCollaborationBackend(settings)

    nested = config.get("openai_compatible")
    if isinstance(nested, dict):
        settings = OpenAICompatibleSettings.from_mapping(nested)
        return OpenAICompatibleCollaborationBackend(settings)

    raise RuntimeError(
        "OpenAI-compatible bridge mode requires either 'llm_config_path' or an "
        "'openai_compatible' settings mapping in the external baseline config."
    )


def _build_ai_researcher_bridge_backend(config: dict[str, Any]) -> OpenAICompatibleCollaborationBackend:
    return _build_openai_compatible_bridge_backend(config)


def _run_ai_researcher_openai_compatible_bridge(
    graph: IdeaGraph,
    config: dict[str, Any],
    progress_callback: Callable[[str], None] | None,
) -> FinalProposal:
    from .baselines import BaselineSpec, _llm_ai_researcher_proxy_proposal

    _stamp_external_baseline_metadata(
        graph,
        execution_mode="openai-compatible-bridge",
        adapter_status=ADAPTER_STATUS_PAPER_FAITHFUL,
        proxy_fallback=False,
        preserved_stages=[
            "seed_generation",
            "proposal_expansion",
            "candidate_ranking",
        ],
    )
    backend = _build_ai_researcher_bridge_backend(config)
    workspace = _make_workspace(
        config,
        baseline_name="ai-researcher",
        instance_name=_clean_text(graph.metadata.get("instance_name")) or _clean_text(graph.topic) or "run",
    )
    graph.metadata["external_baseline_workspace"] = str(workspace)
    sanitized_backend = (
        backend.settings.sanitized_dict()
        if hasattr(backend.settings, "sanitized_dict")
        else {"model": getattr(backend.settings, "model", "")}
    )
    graph.metadata["external_baseline_backend"] = sanitized_backend

    baseline = BaselineSpec(
        name="ai-researcher",
        display_name="AI-Researcher",
        strategy="external",
        description=(
            "Repository-local AI-Researcher compatibility bridge that preserves the "
            "seed-generation, proposal-expansion, and ranking structure under an "
            "OpenAI-compatible backend."
        ),
        prompt_style="ai_researcher_proxy",
        candidate_count=max(2, int(config.get("ideas_n", 4) or 4)),
    )

    _emit(
        progress_callback,
        "AI-Researcher: running the OpenAI-compatible compatibility bridge with seed generation, proposal expansion, and ranking.",
    )
    proposal = _llm_ai_researcher_proxy_proposal(
        graph,
        baseline,
        backend,
        progress_callback=progress_callback,
    )
    selected_path = workspace / "selected_proposal.json"
    selected_path.write_text(json.dumps(asdict(proposal), indent=2, ensure_ascii=False), encoding="utf-8")
    graph.metadata["external_baseline_selected_file"] = str(selected_path)
    return proposal


def _run_ai_researcher(
    graph: IdeaGraph,
    config: dict[str, Any],
    progress_callback: Callable[[str], None] | None,
) -> FinalProposal:
    execution_mode = _clean_text(config.get("execution_mode")).lower()
    if execution_mode in {
        "openai-compatible-bridge",
        "openai_compatible_bridge",
        "openai-compatible",
        "bridge",
    } or _clean_text(config.get("llm_config_path")) or isinstance(config.get("openai_compatible"), dict):
        return _run_ai_researcher_openai_compatible_bridge(graph, config, progress_callback)

    _stamp_external_baseline_metadata(
        graph,
        execution_mode="upstream",
        adapter_status=ADAPTER_STATUS_EXACT_UPSTREAM,
        proxy_fallback=False,
        preserved_stages=[
            "grounded_idea_generation",
            "experiment_plan_generation",
            "tournament_ranking",
        ],
    )
    repo_root = _require_path(config.get("repo_path"), label="AI-Researcher repo_path")
    runner_root = repo_root / "ai_researcher"
    if not runner_root.exists():
        raise RuntimeError(f"AI-Researcher repo is missing the expected 'ai_researcher' directory: {repo_root}")

    workspace = _make_workspace(
        config,
        baseline_name="ai-researcher",
        instance_name=_clean_text(graph.metadata.get("instance_name")) or _clean_text(graph.topic) or "run",
    )
    graph.metadata["external_baseline_workspace"] = str(workspace)

    paper_cache_dir = workspace / "paper_cache"
    seed_ideas_dir = workspace / "seed_ideas"
    normalized_ideas_dir = workspace / "ideas_normalized"
    proposal_root = workspace / "experiment_plans"
    ranking_root = workspace / "ranking"
    for path in (paper_cache_dir, seed_ideas_dir, normalized_ideas_dir, proposal_root, ranking_root):
        path.mkdir(parents=True, exist_ok=True)

    cache_name = _slugify(_clean_text(graph.metadata.get("instance_name")) or _clean_text(graph.topic) or "benchmark")
    paper_cache_path = paper_cache_dir / f"{cache_name}.json"
    seed_ideas_path = seed_ideas_dir / f"{cache_name}.json"
    normalized_ideas_path = normalized_ideas_dir / f"{cache_name}.json"

    paper_cache = _ai_researcher_paper_cache(graph, limit=max(1, int(config.get("grounding_k", 4) or 4)))
    paper_cache_path.write_text(json.dumps(paper_cache, indent=2, ensure_ascii=False), encoding="utf-8")

    python_executable = _clean_text(config.get("python_executable")) or "python"
    engine = _clean_text(config.get("engine")) or "gpt-4o"
    method = _clean_text(config.get("method")) or "prompting"
    ideas_n = max(1, int(config.get("ideas_n", 4) or 4))
    ranking_rounds = max(1, int(config.get("ranking_rounds", 3) or 3))
    timeout_seconds = max(60, int(config.get("timeout_seconds", 1800) or 1800))
    env_overrides = _resolve_env_overrides(config.get("env", {}))

    with _temporary_repo_keys(repo_root, config):
        _emit(progress_callback, "AI-Researcher: generating grounded seed ideas via upstream script.")
        command = [
            python_executable,
            "src/grounded_idea_gen.py",
            "--engine",
            engine,
            "--paper_cache",
            str(paper_cache_path),
            "--idea_cache",
            str(seed_ideas_path),
            "--grounding_k",
            str(max(1, int(config.get("grounding_k", 4) or 4))),
            "--method",
            method,
            "--ideas_n",
            str(ideas_n),
            "--seed",
            str(int(config.get("seed", 2024) or 2024)),
            "--RAG",
            "True" if paper_cache["paper_bank"] else "False",
            "--append_existing_ideas",
            "False",
        ]
        result = _run_command(command, cwd=runner_root, env_overrides=env_overrides, timeout_seconds=timeout_seconds)
        graph.metadata.setdefault("external_baseline_commands", []).append(
            {"stage": "seed_generation", "command": command, "stdout": result.stdout, "stderr": result.stderr}
        )
        if result.returncode != 0:
            raise RuntimeError(f"AI-Researcher seed generation failed:\n{result.stderr or result.stdout}")

        seed_payload = json.loads(seed_ideas_path.read_text(encoding="utf-8"))
        normalized_ideas = _flatten_ai_researcher_seed_ideas(seed_payload)
        normalized_ideas_path.write_text(
            json.dumps(normalized_ideas, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not normalized_ideas.get("ideas"):
            raise RuntimeError("AI-Researcher did not produce any normalized seed ideas.")

        _emit(progress_callback, "AI-Researcher: expanding seed ideas into full proposals via upstream script.")
        command = [
            python_executable,
            "src/experiment_plan_gen.py",
            "--engine",
            engine,
            "--idea_cache_dir",
            str(normalized_ideas_dir) + os.sep,
            "--cache_name",
            cache_name,
            "--experiment_plan_cache_dir",
            str(proposal_root) + os.sep,
            "--idea_name",
            "all",
            "--seed",
            str(int(config.get("seed", 2024) or 2024)),
            "--method",
            method,
        ]
        result = _run_command(command, cwd=runner_root, env_overrides=env_overrides, timeout_seconds=timeout_seconds)
        graph.metadata.setdefault("external_baseline_commands", []).append(
            {"stage": "proposal_generation", "command": command, "stdout": result.stdout, "stderr": result.stderr}
        )
        if result.returncode != 0:
            raise RuntimeError(f"AI-Researcher proposal generation failed:\n{result.stderr or result.stdout}")

        _emit(progress_callback, "AI-Researcher: ranking proposals via upstream tournament script.")
        command = [
            python_executable,
            "src/tournament_ranking.py",
            "--engine",
            engine,
            "--experiment_plan_cache_dir",
            str(proposal_root) + os.sep,
            "--cache_name",
            cache_name,
            "--ranking_score_dir",
            str(ranking_root),
            "--max_round",
            str(ranking_rounds),
        ]
        result = _run_command(command, cwd=runner_root, env_overrides=env_overrides, timeout_seconds=timeout_seconds)
        graph.metadata.setdefault("external_baseline_commands", []).append(
            {"stage": "proposal_ranking", "command": command, "stdout": result.stdout, "stderr": result.stderr}
        )
        if result.returncode != 0:
            raise RuntimeError(f"AI-Researcher proposal ranking failed:\n{result.stderr or result.stdout}")

    top_ideas_path = ranking_root / cache_name / "top_ideas.json"
    if not top_ideas_path.exists():
        raise RuntimeError(f"AI-Researcher ranking output was not found: {top_ideas_path}")
    top_ideas_payload = json.loads(top_ideas_path.read_text(encoding="utf-8"))
    if not isinstance(top_ideas_payload, dict) or not top_ideas_payload:
        raise RuntimeError("AI-Researcher ranking output is empty.")
    best_filename = next(iter(top_ideas_payload))
    proposal_path = proposal_root / cache_name / best_filename
    if not proposal_path.exists():
        raise RuntimeError(f"AI-Researcher selected proposal file is missing: {proposal_path}")

    graph.metadata["external_baseline_selected_file"] = str(proposal_path)
    graph.metadata["external_baseline_selected_score"] = top_ideas_payload[best_filename].get("ai_ranking_score")
    return _proposal_from_ai_researcher_plan(json.loads(proposal_path.read_text(encoding="utf-8")))


def _run_scipip(
    graph: IdeaGraph,
    config: dict[str, Any],
    progress_callback: Callable[[str], None] | None,
) -> FinalProposal:
    if _uses_bridge_mode(config, default="upstream-generator"):
        return _run_scipip_openai_compatible_bridge(graph, config, progress_callback)

    _stamp_external_baseline_metadata(
        graph,
        execution_mode="upstream-generator",
        adapter_status=ADAPTER_STATUS_PAPER_FAITHFUL,
        proxy_fallback=False,
        preserved_stages=[
            "background_conditioned_generation",
            "retrieval_or_inspiration",
            "idea_filtering_or_expansion",
        ],
    )
    repo_root = _require_path(config.get("repo_path"), label="SciPIP repo_path")
    generator_path = repo_root / "src" / "generator.py"
    if not generator_path.exists():
        raise RuntimeError(f"SciPIP repo is missing src/generator.py: {repo_root}")

    workspace = _make_workspace(
        config,
        baseline_name="scipip",
        instance_name=_clean_text(graph.metadata.get("instance_name")) or _clean_text(graph.topic) or "run",
    )
    graph.metadata["external_baseline_workspace"] = str(workspace)
    ids_path = workspace / "backgrounds.json"
    output_dir = workspace / "output"
    output_file = "scipip_output.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(
        json.dumps([{"background": _build_benchmark_background(graph)}], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    python_executable = _clean_text(config.get("python_executable")) or "python"
    config_path = _clean_text(config.get("config_path")) or str(repo_root / "configs" / "datasets.yaml")
    command = [
        python_executable,
        "src/generator.py",
        "new-idea",
        "--config-path",
        config_path,
        "--ids-path",
        str(ids_path),
        "--out-path",
        str(output_dir),
        "--out-file",
        output_file,
        "--retriever-name",
        _clean_text(config.get("retriever_name")) or "SNKG",
        "--brainstorm-mode",
        _clean_text(config.get("brainstorm_mode")) or "mode_c",
        "--use-inspiration",
        str(bool(config.get("use_inspiration", True))),
        "--num",
        "1",
    ]
    _emit(progress_callback, "SciPIP: generating ideas via upstream script.")
    result = _run_command(
        command,
        cwd=repo_root,
        env_overrides=_resolve_env_overrides(config.get("env", {})),
        timeout_seconds=max(60, int(config.get("timeout_seconds", 1800) or 1800)),
    )
    graph.metadata.setdefault("external_baseline_commands", []).append(
        {"stage": "idea_generation", "command": command, "stdout": result.stdout, "stderr": result.stderr}
    )
    if result.returncode != 0:
        raise RuntimeError(f"SciPIP generation failed:\n{result.stderr or result.stdout}")

    output_path = output_dir / output_file
    if not output_path.exists():
        raise RuntimeError(f"SciPIP output file was not found: {output_path}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("SciPIP output did not contain any idea entries.")
    item = payload[0]
    if not isinstance(item, dict):
        raise RuntimeError("SciPIP output entry is malformed.")

    candidates = item.get("expanded_final_ideas") or item.get("filtered_ideas") or item.get("initial_ideas") or []
    if isinstance(candidates, list) and candidates:
        selected_text = _clean_text(candidates[0])
    else:
        selected_text = _clean_text(item.get("all_idea")) or _clean_text(item.get("problem"))
    if not selected_text:
        raise RuntimeError("SciPIP output did not contain a parseable idea text.")

    graph.metadata["external_baseline_selected_file"] = str(output_path)
    return _proposal_from_scipip_text(graph, selected_text)


def _run_scipip_openai_compatible_bridge(
    graph: IdeaGraph,
    config: dict[str, Any],
    progress_callback: Callable[[str], None] | None,
) -> FinalProposal:
    from .baselines import _baseline_postprocess_proposal, _llm_json_object, _proposal_from_payload, get_baseline_spec

    _stamp_external_baseline_metadata(
        graph,
        execution_mode="openai-compatible-bridge",
        adapter_status=ADAPTER_STATUS_PAPER_FAITHFUL,
        proxy_fallback=False,
        preserved_stages=[
            "problem_decomposition",
            "reference_inspiration",
            "idea_synthesis",
        ],
    )
    repo_root = _require_path(config.get("repo_path"), label="SciPIP repo_path")
    workspace = _make_workspace(
        config,
        baseline_name="scipip",
        instance_name=_clean_text(graph.metadata.get("instance_name")) or _clean_text(graph.topic) or "run",
    )
    graph.metadata["external_baseline_workspace"] = str(workspace)

    backend = _build_openai_compatible_bridge_backend(config)
    graph.metadata["external_baseline_backend_settings"] = backend.settings.sanitized_dict()
    graph.metadata["external_baseline_source_repo"] = str(repo_root)

    packet = _bridge_packet(graph)
    decomposition_path = workspace / "scipip_bridge_decomposition.json"
    proposal_path = workspace / "scipip_bridge_proposal.json"

    _emit(progress_callback, "SciPIP: decomposing benchmark background into a structured research problem.")
    decomposition_payload, decomposition_trace = _llm_json_object(
        backend,
        role="SciPIPDecomposition",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are emulating the decomposition stage of SciPIP under a fixed benchmark packet. "
                    "Preserve background-conditioned problem formulation, visible-reference inspiration mining, "
                    "and a concise integrated direction. Use only the provided topic and visible references. "
                    "Return one strict JSON object only. "
                    'Schema: {"research_problem":"...","rationales":["..."],'
                    '"reference_inspirations":[{"title":"...","inspiration":"..."}],'
                    '"integrated_direction":"...","experiment_axes":["..."],"candidate_title":"..."}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(packet, ensure_ascii=False, indent=2),
            },
        ],
        temperature=0.2,
        max_tokens=1600,
    )
    _append_external_trace(graph, stage="scipip_problem_decomposition", trace=decomposition_trace)
    _write_json_artifact(decomposition_path, decomposition_payload)

    _emit(progress_callback, "SciPIP: synthesizing the final benchmark-faithful proposal from the decomposition notes.")
    proposal_payload, proposal_trace = _llm_json_object(
        backend,
        role="SciPIPIdeaSynthesis",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are emulating SciPIP's idea-synthesis stage for scientific paper ideation. "
                    "Expand the structured decomposition into one strong proposal without changing the benchmark topic. "
                    "Use only visible references and decomposition notes. Return one strict JSON object only with the "
                    'fields {"title","problem","existing_methods","motivation","hypothesis","method","evaluation","significance","caveats"}.'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        **packet,
                        "decomposition": decomposition_payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        temperature=0.3,
        max_tokens=1800,
    )
    _append_external_trace(graph, stage="scipip_idea_synthesis", trace=proposal_trace)
    _write_json_artifact(proposal_path, proposal_payload)

    graph.metadata["external_baseline_selected_file"] = str(proposal_path)
    graph.metadata["external_baseline_decomposition_file"] = str(decomposition_path)
    proposal = _proposal_from_payload(proposal_payload)
    return _baseline_postprocess_proposal(graph, get_baseline_spec("scipip"), proposal)


def _run_virsci(
    graph: IdeaGraph,
    config: dict[str, Any],
    progress_callback: Callable[[str], None] | None,
) -> FinalProposal:
    if _uses_bridge_mode(config, default="upstream-multi-agent"):
        return _run_virsci_fixed_topic_bridge(graph, config, progress_callback)

    _stamp_external_baseline_metadata(
        graph,
        execution_mode="upstream-multi-agent",
        adapter_status=ADAPTER_STATUS_EXCLUDED,
        proxy_fallback=False,
        preserved_stages=[
            "multi_agent_discussion",
            "team_synthesis",
        ],
    )
    if _clean_text(graph.metadata.get("benchmark")):
        raise RuntimeError(
            "The upstream Virtual-Scientists repository does not expose a fixed-topic benchmark entrypoint, "
            "so an exact benchmark-faithful integration is not currently possible without patching the upstream system."
        )

    repo_root = _require_path(config.get("repo_path"), label="VirSci repo_path")
    run_root = repo_root / "sci_platform"
    if not (run_root / "run.py").exists():
        raise RuntimeError(f"Virtual-Scientists repo is missing sci_platform/run.py: {repo_root}")

    workspace = _make_workspace(
        config,
        baseline_name="virsci",
        instance_name=_clean_text(graph.metadata.get("instance_name")) or _clean_text(graph.topic) or "run",
    )
    graph.metadata["external_baseline_workspace"] = str(workspace)
    env = _resolve_env_overrides(config.get("env", {}))
    env.setdefault("PYTHONPATH", str(repo_root / "agentscope-main" / "src"))

    command = [
        _clean_text(config.get("python_executable")) or "python",
        "run.py",
        "--runs",
        str(max(1, int(config.get("runs", 1) or 1))),
        "--team_limit",
        str(max(1, int(config.get("team_limit", 1) or 1))),
        "--max_discuss_iteration",
        str(max(1, int(config.get("max_discuss_iteration", 2) or 2))),
        "--max_team_member",
        str(max(2, int(config.get("max_team_member", 2) or 2))),
        "--epochs",
        str(max(1, int(config.get("epochs", 6) or 6))),
    ]
    _emit(progress_callback, "VirSci: launching upstream multi-agent run.")
    result = _run_command(
        command,
        cwd=run_root,
        env_overrides=env,
        timeout_seconds=max(60, int(config.get("timeout_seconds", 3600) or 3600)),
    )
    graph.metadata.setdefault("external_baseline_commands", []).append(
        {"stage": "virsci_run", "command": command, "stdout": result.stdout, "stderr": result.stderr}
    )
    if result.returncode != 0:
        raise RuntimeError(f"Virtual-Scientists run failed:\n{result.stderr or result.stdout}")

    team_info_root = run_root / f"team_info/{max(1, int(config.get('max_discuss_iteration', 2) or 2))}_itrs_{max(2, int(config.get('max_team_member', 2) or 2))}_members"
    team_files = sorted(team_info_root.glob("*_dialogue.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not team_files:
        raise RuntimeError(f"Virtual-Scientists did not produce any team info JSON files in {team_info_root}")
    payload = json.loads(team_files[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Virtual-Scientists team info payload is malformed.")
    graph.metadata["external_baseline_selected_file"] = str(team_files[0])
    return _proposal_from_virsci_payload(graph, payload)


def _run_virsci_fixed_topic_bridge(
    graph: IdeaGraph,
    config: dict[str, Any],
    progress_callback: Callable[[str], None] | None,
) -> FinalProposal:
    from .baselines import _baseline_postprocess_proposal, _llm_json_object, _proposal_from_payload, get_baseline_spec

    _stamp_external_baseline_metadata(
        graph,
        execution_mode="benchmark-fixed-topic-bridge",
        adapter_status=ADAPTER_STATUS_PAPER_FAITHFUL,
        proxy_fallback=False,
        preserved_stages=[
            "multi_agent_discussion",
            "team_synthesis",
        ],
    )
    repo_root = _require_path(config.get("repo_path"), label="VirSci repo_path")
    run_root = repo_root / "sci_platform"
    if not (run_root / "run.py").exists():
        raise RuntimeError(f"Virtual-Scientists repo is missing sci_platform/run.py: {repo_root}")

    workspace = _make_workspace(
        config,
        baseline_name="virsci",
        instance_name=_clean_text(graph.metadata.get("instance_name")) or _clean_text(graph.topic) or "run",
    )
    graph.metadata["external_baseline_workspace"] = str(workspace)

    backend = _build_openai_compatible_bridge_backend(config)
    graph.metadata["external_baseline_backend_settings"] = backend.settings.sanitized_dict()
    graph.metadata["external_baseline_source_repo"] = str(repo_root)

    packet = _bridge_packet(graph)
    discussion_turns = max(2, int(config.get("discussion_turns", 3) or 3))
    personas = [
        ("ScientistAlpha", "frame why this topic matters now and surface the most important bottleneck."),
        ("ScientistBeta", "propose the core mechanism and explain how it differs from nearby work."),
        ("ScientistGamma", "stress-test feasibility, risks, and the evaluation design."),
    ][:discussion_turns]
    discussion_rows: list[dict[str, object]] = []

    for scientist_name, focus in personas:
        _emit(progress_callback, f"VirSci: collecting discussion turn from {scientist_name}.")
        discussion_payload, discussion_trace = _llm_json_object(
            backend,
            role="VirSciDiscussion",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {scientist_name} in a VirSci-style multi-agent scientific discussion. "
                        f"Your focus is: {focus} "
                        "The benchmark topic is fixed and may not be changed. "
                        "Respond with one strict JSON object only. "
                        'Schema: {"scientist":"...", "stance":"...", "topic_commitment":"...", '
                        '"mechanism":"...", "novelty_argument":"...", "risk":"...", "experiment":"..."}'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            **packet,
                            "prior_discussion": discussion_rows,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            temperature=0.35,
            max_tokens=1400,
        )
        discussion_rows.append(discussion_payload)
        _append_external_trace(
            graph,
            stage=f"virsci_discussion_{scientist_name.lower()}",
            trace=discussion_trace,
        )

    transcript_path = workspace / "virsci_bridge_discussion.json"
    _write_json_artifact(transcript_path, {"discussion": discussion_rows})

    _emit(progress_callback, "VirSci: synthesizing the panel discussion into one final proposal.")
    proposal_payload, proposal_trace = _llm_json_object(
        backend,
        role="VirSciSynthesis",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the team leader in a VirSci-style scientific collaboration. "
                    "Synthesize the fixed-topic panel discussion into one coherent proposal without changing the benchmark topic. "
                    "Use only the benchmark packet and recorded discussion. Return one strict JSON object only with the "
                    'fields {"title","problem","existing_methods","motivation","hypothesis","method","evaluation","significance","caveats"}.'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        **packet,
                        "discussion": discussion_rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        temperature=0.25,
        max_tokens=1900,
    )
    _append_external_trace(graph, stage="virsci_team_synthesis", trace=proposal_trace)

    proposal_path = workspace / "virsci_bridge_proposal.json"
    _write_json_artifact(proposal_path, proposal_payload)

    graph.metadata["external_baseline_discussion_file"] = str(transcript_path)
    graph.metadata["external_baseline_selected_file"] = str(proposal_path)
    graph.metadata["external_baseline_discussion_turns"] = len(discussion_rows)
    proposal = _proposal_from_payload(proposal_payload)
    return _baseline_postprocess_proposal(graph, get_baseline_spec("virsci"), proposal)
