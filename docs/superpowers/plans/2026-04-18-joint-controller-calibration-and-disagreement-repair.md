# Joint Controller Calibration And Disagreement Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frozen-dev joint calibration path for both edit overrides and commit decisions, then build the minimum disagreement-repair tooling needed to refresh the two-head critic with stronger supervision.

**Architecture:** Keep the current `parallel_graph_v2` runtime and two-head critic intact. Add a small calibration module that learns controller thresholds from frozen development traces, wire the runtime/baseline loader to consume a calibration artifact, and add a focused disagreement-extraction utility so we can relabel only the highest-value edit states rather than rebuilding the full dataset. The implementation should stay reviewer-safe: no adaptive paper-eval tuning, only frozen development-set calibration and auditable artifacts.

**Tech Stack:** Python 3, pytest, existing `idea_graph` runtime controller stack, JSON/JSONL artifacts, two-head runtime bundle loading, local quality/controller packet artifacts.

---

## File Map

- Create: `src/idea_graph/joint_controller_calibration.py`
  Responsibility: schema, fitting, validation, and loading logic for joint edit+commit calibration artifacts.
- Create: `scripts/calibrate_joint_controller.py`
  Responsibility: build a frozen-dev calibration artifact from controller packet runs.
- Create: `scripts/extract_edit_disagreement_subset.py`
  Responsibility: extract disagreement-heavy edit states from saved heuristic/critic packet runs for targeted relabeling.
- Create: `tests/test_joint_controller_calibration.py`
  Responsibility: regression tests for artifact fitting, guardrails, and runtime override application.
- Create: `tests/test_extract_edit_disagreement_subset.py`
  Responsibility: regression tests for packet alignment and disagreement subset extraction.
- Modify: `src/idea_graph/baselines.py`
  Responsibility: load joint controller calibration artifacts into runtime metadata for the two-head baseline.
- Modify: `src/idea_graph/relation_graph_two_head_runtime_critic.py`
  Responsibility: expose calibration-friendly score access and artifact loading hooks without changing existing runtime scoring behavior.
- Modify: `docs/evaluation.md`
  Responsibility: rename paper-facing wording from commit-only calibration to frozen-dev joint controller calibration.

## Task 1: Joint Calibration Artifact

**Files:**
- Create: `src/idea_graph/joint_controller_calibration.py`
- Create: `tests/test_joint_controller_calibration.py`
- Modify: `src/idea_graph/baselines.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from idea_graph.joint_controller_calibration import (
    JointControllerCalibration,
    JointControllerCalibrationError,
    apply_joint_controller_calibration,
    fit_joint_controller_calibration,
)


def test_fit_joint_controller_calibration_picks_joint_thresholds() -> None:
    calibration = fit_joint_controller_calibration(
        edit_examples=[
            {"override_margin": 0.02, "label": 0},
            {"override_margin": 0.09, "label": 1},
            {"override_margin": 0.12, "label": 1},
        ],
        commit_examples=[
            {"commit_probability": 0.45, "round_index": 2, "label": 0},
            {"commit_probability": 0.77, "round_index": 3, "label": 1},
            {"commit_probability": 0.81, "round_index": 4, "label": 1},
        ],
    )
    assert calibration.tau_override >= 0.05
    assert calibration.gamma_commit >= 0.70
    assert calibration.min_commit_round >= 2


def test_fit_joint_controller_calibration_rejects_single_class_commit_labels() -> None:
    try:
        fit_joint_controller_calibration(
            edit_examples=[{"override_margin": 0.10, "label": 1}],
            commit_examples=[{"commit_probability": 0.80, "round_index": 3, "label": 1}],
        )
    except JointControllerCalibrationError as exc:
        assert "both positive and negative" in str(exc)
    else:
        raise AssertionError("Expected JointControllerCalibrationError")


def test_apply_joint_controller_calibration_overrides_runtime_metadata(tmp_path: Path) -> None:
    calibration = JointControllerCalibration(
        tau_override=0.11,
        tau_commit=0.08,
        gamma_commit=0.73,
        min_commit_round=3,
        guard_support_threshold=0.72,
        source="critic_dev",
        version="joint_controller_calibration_v1",
    )
    metadata = {
        "runtime_controller_tau_override": 0.05,
        "runtime_controller_tau_commit": 0.08,
        "runtime_controller_gamma_commit": 0.60,
        "runtime_controller_min_commit_round": 2,
        "runtime_controller_guard_support_threshold": 0.66,
    }
    applied = apply_joint_controller_calibration(metadata, calibration)
    assert applied["runtime_controller_tau_override"] == 0.11
    assert applied["runtime_controller_gamma_commit"] == 0.73
    assert applied["runtime_controller_min_commit_round"] == 3
    assert applied["runtime_controller_guard_support_threshold"] == 0.72
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_joint_controller_calibration.py -q`
Expected: FAIL with `ModuleNotFoundError` for `idea_graph.joint_controller_calibration`.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


class JointControllerCalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class JointControllerCalibration:
    tau_override: float
    tau_commit: float
    gamma_commit: float
    min_commit_round: int
    guard_support_threshold: float
    source: str
    version: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_joint_controller_calibration(
    *,
    edit_examples: Sequence[Mapping[str, object]],
    commit_examples: Sequence[Mapping[str, object]],
) -> JointControllerCalibration:
    commit_labels = {int(example["label"]) for example in commit_examples}
    if commit_labels != {0, 1}:
        raise JointControllerCalibrationError(
            "Commit calibration requires both positive and negative labels."
        )
    tau_override = max(float(example["override_margin"]) for example in edit_examples if int(example["label"]) == 1)
    gamma_commit = min(float(example["commit_probability"]) for example in commit_examples if int(example["label"]) == 1)
    min_commit_round = min(int(example["round_index"]) for example in commit_examples if int(example["label"]) == 1)
    return JointControllerCalibration(
        tau_override=round(tau_override, 4),
        tau_commit=0.08,
        gamma_commit=round(gamma_commit, 4),
        min_commit_round=min_commit_round,
        guard_support_threshold=0.66,
        source="critic_dev",
        version="joint_controller_calibration_v1",
    )


def apply_joint_controller_calibration(
    metadata: Mapping[str, object],
    calibration: JointControllerCalibration,
) -> dict[str, object]:
    updated = dict(metadata)
    updated["runtime_controller_tau_override"] = float(calibration.tau_override)
    updated["runtime_controller_tau_commit"] = float(calibration.tau_commit)
    updated["runtime_controller_gamma_commit"] = float(calibration.gamma_commit)
    updated["runtime_controller_min_commit_round"] = int(calibration.min_commit_round)
    updated["runtime_controller_guard_support_threshold"] = float(calibration.guard_support_threshold)
    updated["runtime_controller_calibration_version"] = calibration.version
    updated["runtime_controller_calibration_source"] = calibration.source
    return updated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_joint_controller_calibration.py -q`
Expected: PASS.

- [ ] **Step 5: Wire the baseline loader to apply artifacts when present**

```python
from .joint_controller_calibration import (
    JointControllerCalibration,
    apply_joint_controller_calibration,
    load_joint_controller_calibration,
)


calibration_path = model_dir / "joint_controller_calibration.json"
if calibration_path.exists():
    calibration = load_joint_controller_calibration(calibration_path)
    metadata = apply_joint_controller_calibration(metadata, calibration)
```

- [ ] **Step 6: Run focused regression tests**

Run: `python -m pytest tests/test_joint_controller_calibration.py tests/test_benchmark_mode_and_baselines.py tests/test_relation_graph_two_head_runtime_critic.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/idea_graph/joint_controller_calibration.py tests/test_joint_controller_calibration.py src/idea_graph/baselines.py
git commit -m "feat: add joint controller calibration artifact"
```

## Task 2: Calibration CLI Over Frozen Dev Packets

**Files:**
- Create: `scripts/calibrate_joint_controller.py`
- Modify: `src/idea_graph/joint_controller_calibration.py`
- Test: `tests/test_joint_controller_calibration.py`

- [ ] **Step 1: Write the failing CLI test**

```python
def test_calibration_cli_writes_joint_artifact(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    (packet_dir / "edit_examples.jsonl").write_text(
        '{"override_margin": 0.03, "label": 0}\n'
        '{"override_margin": 0.10, "label": 1}\n',
        encoding="utf-8",
    )
    (packet_dir / "commit_examples.jsonl").write_text(
        '{"commit_probability": 0.48, "round_index": 2, "label": 0}\n'
        '{"commit_probability": 0.79, "round_index": 3, "label": 1}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "joint_controller_calibration.json"
    code = subprocess.call(
        [
            sys.executable,
            "scripts/calibrate_joint_controller.py",
            "--edit-examples",
            str(packet_dir / "edit_examples.jsonl"),
            "--commit-examples",
            str(packet_dir / "commit_examples.jsonl"),
            "--output-path",
            str(output_path),
        ]
    )
    assert code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["version"] == "joint_controller_calibration_v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_joint_controller_calibration.py::test_calibration_cli_writes_joint_artifact -q`
Expected: FAIL because `scripts/calibrate_joint_controller.py` does not exist.

- [ ] **Step 3: Implement the minimal CLI**

```python
parser.add_argument("--edit-examples", required=True)
parser.add_argument("--commit-examples", required=True)
parser.add_argument("--output-path", required=True)

edit_examples = _load_jsonl(args.edit_examples)
commit_examples = _load_jsonl(args.commit_examples)
calibration = fit_joint_controller_calibration(
    edit_examples=edit_examples,
    commit_examples=commit_examples,
)
Path(args.output_path).write_text(
    json.dumps(calibration.as_dict(), indent=2),
    encoding="utf-8",
)
```

- [ ] **Step 4: Run CLI tests**

Run: `python -m pytest tests/test_joint_controller_calibration.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_joint_controller.py tests/test_joint_controller_calibration.py src/idea_graph/joint_controller_calibration.py
git commit -m "feat: add joint controller calibration cli"
```

## Task 3: Edit Disagreement Extraction

**Files:**
- Create: `scripts/extract_edit_disagreement_subset.py`
- Create: `tests/test_extract_edit_disagreement_subset.py`
- Modify: `docs/evaluation.md`

- [ ] **Step 1: Write the failing disagreement extraction test**

```python
def test_extract_edit_disagreement_subset_aligns_paired_runs(tmp_path: Path) -> None:
    run_manifest = tmp_path / "run_manifest.jsonl"
    run_manifest.write_text(
        '{"group_id":"g1","baseline_name":"ours-eig","run_dir":"heuristic"}\n'
        '{"group_id":"g1","baseline_name":"ours-eig-critic-graph-twohead","run_dir":"critic"}\n',
        encoding="utf-8",
    )
    heuristic_graph = {"metadata": {"runtime_controller_log": []}}
    critic_graph = {
        "metadata": {
            "runtime_controller_log": [
                {
                    "round_name": "Round2",
                    "role": "MethodDesigner",
                    "selected_candidate": {"candidate_id": "critic-a"},
                    "heuristic_candidate": {"candidate_id": "heur-a"},
                    "controller_decision": {"selected_source": "critic", "override_margin": 0.12},
                }
            ]
        }
    }
    ...
    assert rows[0]["group_id"] == "g1"
    assert rows[0]["override_margin"] == 0.12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract_edit_disagreement_subset.py -q`
Expected: FAIL because `scripts/extract_edit_disagreement_subset.py` does not exist.

- [ ] **Step 3: Implement the minimal extractor**

```python
for group_id, runs in grouped_rows.items():
    heuristic = runs.get("ours-eig")
    critic = runs.get("ours-eig-critic-graph-twohead")
    if not heuristic or not critic:
        continue
    critic_log = critic_graph["metadata"].get("runtime_controller_log", [])
    for row in critic_log:
        decision = row.get("controller_decision", {})
        if str(decision.get("selected_source", "")).strip() != "critic":
            continue
        disagreements.append(
            {
                "group_id": group_id,
                "round_name": row.get("round_name"),
                "role": row.get("role"),
                "override_margin": decision.get("override_margin"),
                "critic_candidate_id": row.get("selected_candidate", {}).get("candidate_id"),
                "heuristic_candidate_id": row.get("heuristic_candidate", {}).get("candidate_id"),
            }
        )
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_extract_edit_disagreement_subset.py -q`
Expected: PASS.

- [ ] **Step 5: Update paper-facing wording**

```markdown
- calibrated learned two-head parallel EIG
  learned edit/action selection plus frozen-dev joint controller calibration
```

- [ ] **Step 6: Run the focused suite**

Run: `python -m pytest tests/test_extract_edit_disagreement_subset.py tests/test_joint_controller_calibration.py tests/test_parallel_runtime.py tests/test_benchmark_mode_and_baselines.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/extract_edit_disagreement_subset.py tests/test_extract_edit_disagreement_subset.py docs/evaluation.md
git commit -m "feat: add disagreement extraction for controller repair"
```

## Self-Review

- Spec coverage:
  - joint edit+commit calibration: Task 1 and Task 2
  - disagreement-driven edit repair subset: Task 3
  - reviewer-safe frozen-dev wording: Task 3
- Placeholder scan:
  - no `TODO` / `TBD` placeholders remain
  - all file paths and commands are explicit
- Type consistency:
  - artifact name remains `joint_controller_calibration.json`
  - baseline name remains `ours-eig-critic-graph-twohead`
  - runtime metadata keys remain `runtime_controller_*`
