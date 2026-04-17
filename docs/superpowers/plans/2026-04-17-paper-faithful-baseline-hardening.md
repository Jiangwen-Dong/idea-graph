# Paper-Faithful Baseline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden `ai-researcher`, `scipip`, and `virsci` with fast paper-faithful go/no-go gates so the paper main table can be launched without spending excessive reproduction time.

**Architecture:** Add a lightweight reproduction registry and validation packet around the existing external baseline adapters. Prefer existing upstream wrappers, document adapter status, and fail fast when a baseline cannot satisfy the shared benchmark protocol.

**Tech Stack:** Python 3, existing `run_pipeline.py`, `run_quality_batch.py`, `external_baselines.py`, JSON/Markdown reports, pytest.

---

## File Structure

- Create: `docs/baseline_reproduction_matrix.md`
  Purpose: human-readable paper-facing matrix of exact, paper-faithful, appendix-only, or excluded baseline status.
- Create: `configs/external_baselines.paper_faithful.example.json`
  Purpose: example config for true/paper-faithful external baseline runs without secrets.
- Create: `scripts/check_external_baselines.py`
  Purpose: cheap preflight that validates external baseline config, repo paths, required scripts, and expected execution mode.
- Modify: `src/idea_graph/external_baselines.py`
  Purpose: record adapter status metadata and make VirSci fixed-topic feasibility explicit rather than hidden.
- Modify: `tests/test_benchmark_mode_and_baselines.py`
  Purpose: lock metadata and no-proxy fallback behavior.
- Create: `tests/test_external_baseline_preflight.py`
  Purpose: test the preflight report without needing upstream repos.
- Create output root during execution: `outputs/baseline_hardening/`
  Purpose: store B0/B1 smoke artifacts and go/no-go summaries.

## Task 1: Add Reproduction Matrix And Paper-Faithful Config

**Files:**
- Create: `docs/baseline_reproduction_matrix.md`
- Create: `configs/external_baselines.paper_faithful.example.json`

- [ ] **Step 1: Add the reproduction matrix**

Create `docs/baseline_reproduction_matrix.md` with:

```markdown
# Baseline Reproduction Matrix

This matrix records which baselines are valid for headline paper evaluation
under the shared benchmark I/O contract.

## Eligibility Labels

- `controlled-local`: simple local baseline with transparent protocol
- `exact-upstream`: upstream repository scripts run the method logic
- `paper-faithful-adapter`: original stages are preserved with thin benchmark I/O adapters
- `appendix-only`: useful diagnostic but not headline evidence
- `exclude`: not benchmark-faithful enough for the main table

## Current Matrix

| Baseline | Target Label | Preserved Method Structure | Adapter Scope | Main-Table Gate |
| --- | --- | --- | --- | --- |
| `direct` | `controlled-local` | one-pass single-agent idea generation | shared benchmark packet and output schema | eligible |
| `self-refine` | `controlled-local` | draft, critique, revision | shared benchmark packet and output schema | eligible |
| `ai-researcher` | `paper-faithful-adapter` | seed generation, proposal expansion, candidate ranking | paper-cache construction, provider config, output normalization | B0 then B1 |
| `scipip` | `paper-faithful-adapter` | upstream `generator.py new-idea` with retrieval/decomposition | benchmark background JSON, config path, output normalization | B0 then B1 |
| `virsci` | `exclude` until fixed-topic adapter passes | multi-agent team discussion and synthesis | fixed-topic packet injection if feasible | feasibility audit first |
| `ai-researcher-proxy` | `appendix-only` | local approximation of seed/expand/rank | implemented inside this repo | not headline |
| `scipip-proxy` | `appendix-only` | local structured decomposition approximation | implemented inside this repo | not headline |
| `virsci-proxy` | `appendix-only` | local discussion-style approximation | implemented inside this repo | not headline |

## Paper Wording

Use:

> We implemented benchmark-faithful reproductions of prior baselines under a
> unified evaluation interface, preserving each method's core stages while
> adding thin adapters for benchmark packet ingestion and output normalization.

Avoid claiming bit-for-bit upstream reproduction unless the exact-upstream path
is actually used.
```

- [ ] **Step 2: Add a paper-faithful config template**

Create `configs/external_baselines.paper_faithful.example.json` with:

```json
{
  "ai-researcher": {
    "enabled": true,
    "execution_mode": "upstream",
    "repo_path": "C:/path/to/AI-Researcher",
    "python_executable": "python",
    "engine": "gpt-4o",
    "method": "prompting",
    "ideas_n": 4,
    "ranking_rounds": 3,
    "grounding_k": 4,
    "timeout_seconds": 1800,
    "keys": {
      "api_key": {
        "env": "OPENAI_API_KEY"
      }
    },
    "env": {
      "OPENAI_API_KEY": {
        "env": "OPENAI_API_KEY"
      }
    }
  },
  "scipip": {
    "enabled": true,
    "repo_path": "C:/path/to/SciPIP",
    "python_executable": "python",
    "config_path": "C:/path/to/SciPIP/configs/datasets.yaml",
    "workspace_root": ".tmp-external-baseline-runs",
    "retriever_name": "SNKG",
    "brainstorm_mode": "mode_c",
    "use_inspiration": true,
    "timeout_seconds": 3600,
    "env": {
      "OPENAI_API_KEY": {
        "env": "OPENAI_API_KEY"
      }
    }
  },
  "virsci": {
    "enabled": false,
    "repo_path": "C:/path/to/Virtual-Scientists",
    "python_executable": "python",
    "workspace_root": ".tmp-external-baseline-runs",
    "runs": 1,
    "team_limit": 1,
    "max_discuss_iteration": 2,
    "max_team_member": 2,
    "epochs": 6,
    "timeout_seconds": 7200,
    "env": {
      "OPENAI_API_KEY": {
        "env": "OPENAI_API_KEY"
      }
    }
  }
}
```

- [ ] **Step 3: Verify formatting**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Commit**

Run:

```powershell
git add docs/baseline_reproduction_matrix.md configs/external_baselines.paper_faithful.example.json
git commit -m "docs: define paper-faithful baseline matrix"
git push origin main
```

## Task 2: Add External Baseline Preflight

**Files:**
- Create: `scripts/check_external_baselines.py`
- Create: `tests/test_external_baseline_preflight.py`

- [ ] **Step 1: Write the failing preflight tests**

Create `tests/test_external_baseline_preflight.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.check_external_baselines import check_external_baseline_config


class ExternalBaselinePreflightTests(unittest.TestCase):
    def test_preflight_marks_missing_repos_as_not_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "external.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ai-researcher": {
                            "enabled": True,
                            "execution_mode": "upstream",
                            "repo_path": str(Path(tmp) / "missing-ai-researcher"),
                        },
                        "scipip": {
                            "enabled": True,
                            "repo_path": str(Path(tmp) / "missing-scipip"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = check_external_baseline_config(config_path)

        by_name = {row["baseline"]: row for row in report["baselines"]}
        self.assertFalse(by_name["ai-researcher"]["ready"])
        self.assertFalse(by_name["scipip"]["ready"])
        self.assertIn("missing", " ".join(by_name["ai-researcher"]["issues"]).lower())

    def test_preflight_accepts_minimal_fake_upstream_layouts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ai_repo = root / "AI-Researcher"
            ai_runner = ai_repo / "ai_researcher" / "src"
            ai_runner.mkdir(parents=True)
            for name in ("grounded_idea_gen.py", "experiment_plan_gen.py", "tournament_ranking.py"):
                (ai_runner / name).write_text("print('ok')\n", encoding="utf-8")

            scipip_repo = root / "SciPIP"
            (scipip_repo / "src").mkdir(parents=True)
            (scipip_repo / "src" / "generator.py").write_text("print('ok')\n", encoding="utf-8")
            config_file = scipip_repo / "configs" / "datasets.yaml"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("datasets: []\n", encoding="utf-8")

            config_path = root / "external.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ai-researcher": {
                            "enabled": True,
                            "execution_mode": "upstream",
                            "repo_path": str(ai_repo),
                        },
                        "scipip": {
                            "enabled": True,
                            "repo_path": str(scipip_repo),
                            "config_path": str(config_file),
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = check_external_baseline_config(config_path)

        by_name = {row["baseline"]: row for row in report["baselines"]}
        self.assertTrue(by_name["ai-researcher"]["ready"])
        self.assertEqual(by_name["ai-researcher"]["adapter_status"], "exact-upstream")
        self.assertTrue(by_name["scipip"]["ready"])
        self.assertEqual(by_name["scipip"]["adapter_status"], "paper-faithful-adapter")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest tests/test_external_baseline_preflight.py -q
```

Expected: fails because `scripts/check_external_baselines.py` does not exist.

- [ ] **Step 3: Implement the preflight script**

Create `scripts/check_external_baselines.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _check_file(path: Path, issues: list[str], label: str) -> bool:
    if path.exists():
        return True
    issues.append(f"Missing {label}: {path}")
    return False


def _ai_researcher_status(entry: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    mode = _clean(entry.get("execution_mode")).lower() or "upstream"
    repo = Path(_clean(entry.get("repo_path")) or ".")
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


def _scipip_status(entry: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    repo = Path(_clean(entry.get("repo_path")) or ".")
    ready = repo.exists()
    if not ready:
        issues.append(f"Missing SciPIP repo_path: {repo}")
    if ready:
        ready = _check_file(repo / "src" / "generator.py", issues, "SciPIP generator.py") and ready
    config_path = Path(_clean(entry.get("config_path")) or str(repo / "configs" / "datasets.yaml"))
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


def _virsci_status(entry: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    repo = Path(_clean(entry.get("repo_path")) or ".")
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
    baselines: list[dict[str, Any]] = []
    if isinstance(payload.get("ai-researcher"), dict):
        baselines.append(_ai_researcher_status(payload["ai-researcher"]))
    if isinstance(payload.get("scipip"), dict):
        baselines.append(_scipip_status(payload["scipip"]))
    if isinstance(payload.get("virsci"), dict):
        baselines.append(_virsci_status(payload["virsci"]))
    return {
        "config_path": str(config_path),
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
```

- [ ] **Step 4: Run GREEN**

Run:

```powershell
python -m pytest tests/test_external_baseline_preflight.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts/check_external_baselines.py tests/test_external_baseline_preflight.py
git commit -m "feat: preflight paper-faithful baselines"
git push origin main
```

## Task 3: Record Adapter Status In Baseline Metadata

**Files:**
- Modify: `src/idea_graph/external_baselines.py`
- Modify: `tests/test_benchmark_mode_and_baselines.py`

- [ ] **Step 1: Add metadata assertions**

In `tests/test_benchmark_mode_and_baselines.py`, update the AI-Researcher bridge test to assert:

```python
self.assertEqual(graph.metadata["external_baseline_adapter_status"], "paper-faithful-adapter")
self.assertFalse(graph.metadata.get("external_baseline_proxy_fallback", True))
```

Add a test for the VirSci benchmark-mode no-go:

```python
def test_virsci_benchmark_mode_records_no_go_reason(self) -> None:
    instance = attach_baseline_metadata(
        self._ai_idea_bench_instance(),
        baseline_name="virsci",
        io_mode="auto",
    )
    with self.assertRaises(RuntimeError) as context:
        run_baseline_experiment(
            instance,
            baseline_name="virsci",
            external_baseline_config={"virsci": {"enabled": True, "repo_path": "C:/missing/Virtual-Scientists"}},
        )
    self.assertIn("fixed-topic benchmark", str(context.exception).lower())
```

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest tests/test_benchmark_mode_and_baselines.py::BenchmarkModeAndBaselineTests::test_ai_researcher_external_bridge_runs_with_openai_compatible_backend -q
```

Expected: fails because metadata keys are not recorded yet.

- [ ] **Step 3: Add adapter metadata**

In `src/idea_graph/external_baselines.py`, inside `_run_ai_researcher_openai_compatible_bridge`, after setting `external_baseline_execution_mode`, add:

```python
graph.metadata["external_baseline_adapter_status"] = "paper-faithful-adapter"
graph.metadata["external_baseline_proxy_fallback"] = False
graph.metadata["external_baseline_preserved_stages"] = [
    "seed_generation",
    "proposal_expansion",
    "candidate_ranking",
]
```

Inside the upstream `_run_ai_researcher` path, after setting `external_baseline_workspace`, add:

```python
graph.metadata["external_baseline_execution_mode"] = "upstream"
graph.metadata["external_baseline_adapter_status"] = "exact-upstream"
graph.metadata["external_baseline_proxy_fallback"] = False
graph.metadata["external_baseline_preserved_stages"] = [
    "grounded_idea_generation",
    "experiment_plan_generation",
    "tournament_ranking",
]
```

Inside `_run_scipip`, after setting `external_baseline_workspace`, add:

```python
graph.metadata["external_baseline_execution_mode"] = "upstream-generator"
graph.metadata["external_baseline_adapter_status"] = "paper-faithful-adapter"
graph.metadata["external_baseline_proxy_fallback"] = False
graph.metadata["external_baseline_preserved_stages"] = [
    "background_conditioned_generation",
    "retrieval_or_inspiration",
    "idea_filtering_or_expansion",
]
```

Inside `_run_virsci`, before raising the benchmark-mode error, set:

```python
graph.metadata["external_baseline_execution_mode"] = "upstream-multi-agent"
graph.metadata["external_baseline_adapter_status"] = "exclude-until-fixed-topic-adapter"
graph.metadata["external_baseline_proxy_fallback"] = False
```

- [ ] **Step 4: Run GREEN**

Run:

```powershell
python -m pytest tests/test_benchmark_mode_and_baselines.py -q
```

Expected: all baseline tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/idea_graph/external_baselines.py tests/test_benchmark_mode_and_baselines.py
git commit -m "feat: record paper-faithful baseline metadata"
git push origin main
```

## Task 4: Run B0 Feasibility Smokes

**Files:**
- Output: `outputs/baseline_hardening/b0_preflight_report.json`
- Output: `outputs/baseline_hardening/b0_smoke/`

- [ ] **Step 1: Preflight local external config**

Use the local config file that contains real paths and no tracked secrets:

```powershell
python scripts/check_external_baselines.py --external-baseline-config configs/external_baselines.qwen.json --output-path outputs/baseline_hardening/b0_preflight_report.json
```

Expected:

- `direct` and `self-refine` do not need preflight.
- `ai-researcher` is ready through either upstream or bridge mode.
- `scipip` is ready only if the real upstream repo/config exists.
- `virsci` remains not ready until fixed-topic adapter work is explicitly done.

- [ ] **Step 2: Run cheap B0 smoke for always-ready systems**

Run:

```powershell
python scripts/run_quality_batch.py --llm-config configs/openai_compatible.example.json --external-baseline-config configs/external_baselines.qwen.json --ai-indices 13 --live-row-indices 0 --baselines direct self-refine ai-researcher ours-eig --batch-name baseline-b0-core --native-eval
```

Expected:

- all four systems finish two runs each
- `ai-researcher` artifacts record `external_baseline_adapter_status`
- `ours-eig` artifacts record `runtime_protocol=parallel_graph_v2`

- [ ] **Step 3: Add SciPIP to B0 only if preflight says ready**

Run only when `b0_preflight_report.json` lists `scipip` under `ready_baselines`:

```powershell
python scripts/run_quality_batch.py --llm-config configs/openai_compatible.example.json --external-baseline-config configs/external_baselines.qwen.json --ai-indices 13 --live-row-indices 0 --baselines scipip --batch-name baseline-b0-scipip --native-eval
```

Expected:

- `scipip` finishes both runs or produces actionable upstream config errors
- no fallback to `scipip-proxy`

- [ ] **Step 4: Do not block on VirSci**

Run the preflight/audit only:

```powershell
python scripts/check_external_baselines.py --external-baseline-config configs/external_baselines.qwen.json --output-path outputs/baseline_hardening/virsci_preflight_report.json
```

Expected:

- `virsci` is marked not main-table-ready unless a fixed-topic adapter has been separately implemented.

## Task 5: Decide B1 Candidate Set And Launch Small Main-Table Smoke

**Files:**
- Output: `outputs/baseline_hardening/b1_smoke/`
- Modify: `docs/baseline_reproduction_matrix.md`

- [ ] **Step 1: Freeze B1 candidate list**

Use:

```text
direct
self-refine
ai-researcher
ours-eig
```

Add `scipip` only if B0 SciPIP passed. Add `virsci` only if fixed-topic adapter passed.

- [ ] **Step 2: Run B1 smoke on eight cases**

Run the core B1 command:

```powershell
python scripts/run_quality_batch.py --llm-config configs/openai_compatible.example.json --external-baseline-config configs/external_baselines.qwen.json --ai-indices 13 15 18 21 --live-row-indices 0 23 47 70 --baselines direct self-refine ai-researcher ours-eig --batch-name baseline-b1-core --native-eval
```

If SciPIP passed B0, run:

```powershell
python scripts/run_quality_batch.py --llm-config configs/openai_compatible.example.json --external-baseline-config configs/external_baselines.qwen.json --ai-indices 13 15 18 21 --live-row-indices 0 23 47 70 --baselines scipip --batch-name baseline-b1-scipip --native-eval
```

- [ ] **Step 3: Update the matrix with gate results**

Edit `docs/baseline_reproduction_matrix.md`:

- set `ai-researcher` gate to `B1 passed` or `appendix-only`
- set `scipip` gate to `B1 passed`, `B0 failed`, or `appendix-only`
- set `virsci` gate to `exclude` unless fixed-topic adapter passed

- [ ] **Step 4: Commit matrix update**

Run:

```powershell
git add docs/baseline_reproduction_matrix.md
git commit -m "docs: record baseline smoke gate results"
git push origin main
```

## Task 6: Main Paper-Eval Readiness Decision

**Files:**
- Output: `outputs/baseline_hardening/main_table_decision.json`
- Output: `outputs/baseline_hardening/main_table_decision.md`

- [ ] **Step 1: Write decision artifacts**

Create `outputs/baseline_hardening/main_table_decision.md` with:

```markdown
# Main-Table Baseline Decision

## Included

- `direct`: controlled local lower bound.
- `self-refine`: controlled local iterative baseline.
- `ai-researcher`: include if B1 passed under exact-upstream or paper-faithful adapter status.
- `scipip`: include only if B1 passed.
- `ours-eig`: active parallel-v2 method.

## Excluded Or Appendix

- `virsci`: exclude from headline table unless fixed-topic adapter passed B1.
- `*-proxy`: appendix-only diagnostics.

## Paper Wording

We implemented benchmark-faithful reproductions of prior baselines under a
unified evaluation interface, preserving each method's core stages while adding
thin adapters for benchmark packet ingestion and output normalization. Baselines
that could not satisfy the shared benchmark protocol without substantial method
changes were excluded from the headline comparison rather than replaced by
local proxy implementations.
```

- [ ] **Step 2: Do not commit output artifacts**

The decision artifacts remain under ignored `outputs/`. Commit only docs/code
changes that describe the protocol.

## Plan Self-Review

- Spec coverage:
  - baseline taxonomy: Task 1
  - per-baseline hardening: Tasks 2, 3, 4, and 5
  - fast go/no-go gates: Tasks 4 and 5
  - main-table decision: Task 6
- Placeholder scan:
  - no placeholder requirements remain
  - VirSci is intentionally a feasibility gate, not a blocked implementation promise
- Type consistency:
  - adapter labels are consistently `exact-upstream`, `paper-faithful-adapter`, `appendix-only`, and `exclude-until-fixed-topic-adapter`
  - output roots consistently use `outputs/baseline_hardening/`
