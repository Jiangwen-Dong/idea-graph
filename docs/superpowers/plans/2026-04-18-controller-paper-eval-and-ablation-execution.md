# Controller Paper-Eval And Ablation Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the parallel EIG controller experiments launchable with stable method labels, auditable calibration settings, and executable `no-commit` / `no-edit` ablations before running held-out paper-eval.

**Architecture:** Keep all paper-facing method definitions in `src/idea_graph/experiment_plans.py`, while runtime behavior stays in `baselines.py` and `parallel_runtime.py`. `no-commit` disables the commit head but keeps learned edit selection. `no-edit` keeps learned post-round commit scoring but bypasses learned role-local edit reranking.

**Implementation note:** The landed implementation now treats `ours-eig-critic-calibrated`, `ours-eig-critic-no-commit`, and `ours-eig-critic-no-edit` as directly launchable baseline names. The frozen-dev calibration artifact lives at `data/splits/parallel_v2/frozen_dev_joint_controller_calibration.json`, while the raw `ours-eig-critic-graph-twohead` path disables calibration explicitly.

**Tech Stack:** Python, pytest, dataclasses, existing `parallel_graph_v2` runtime, existing controller packet runner and quality-batch runner.

---

## File Structure

- Modify `src/idea_graph/experiment_plans.py`
  Add method-plan entries for `ours-eig-critic-calibrated`, `ours-eig-critic-no-commit`, and `ours-eig-critic-no-edit`. Ensure plan metadata records stable paper-facing variants and safe runtime-controller overrides.
- Modify `src/idea_graph/baselines.py`
  Add `runtime_controller_use_edit` to runtime metadata, defaults, and loaded controller metadata. Keep raw two-head controller reproducible even if a model-dir calibration file exists.
- Modify `src/idea_graph/relation_graph_runtime_critic.py`
  Extend `RelationGraphRuntimeConfig` with `use_edit: bool = True`.
- Modify `src/idea_graph/parallel_runtime.py`
  Bypass `_maybe_apply_runtime_controller(...)` edit reranking when `config.use_edit` is false, while preserving commit-head use in `_runtime_commit_check(...)`.
- Modify `tests/test_experiment_plans.py`
  Define expected metadata for the new method plans.
- Modify `tests/test_benchmark_mode_and_baselines.py`
  Define runtime-controller default metadata, including `runtime_controller_use_edit`.
- Modify `tests/test_parallel_runtime.py`
  Add a runtime-level test proving no-edit bypasses edit selection but still allows learned commit.

## Task 1: Method Plans For Controller Variants

**Files:**
- Modify: `tests/test_experiment_plans.py`
- Modify: `src/idea_graph/experiment_plans.py`

- [ ] **Step 1: Write the failing test for calibrated, no-commit, and no-edit plans**

Add this test to `tests/test_experiment_plans.py` after `test_main_method_plan_includes_two_head_graph_controller`:

```python
    def test_ablation_method_plan_includes_controller_variants(self) -> None:
        expected = {
            "ours-eig-critic-calibrated",
            "ours-eig-critic-no-commit",
            "ours-eig-critic-no-edit",
        }
        self.assertTrue(expected.issubset(ABLATION_METHOD_PLANS))

        calibrated = prepare_instance_for_method_plan(
            self._instance(),
            plan=ABLATION_METHOD_PLANS["ours-eig-critic-calibrated"],
        )
        self.assertEqual(calibrated.metadata["method_name"], "ours-eig-critic-calibrated")
        self.assertEqual(calibrated.metadata["runner_baseline_name"], "ours-eig-critic-graph-twohead")
        self.assertEqual(calibrated.metadata["runtime_protocol"], "parallel_graph_v2")
        self.assertEqual(calibrated.metadata["runtime_controller_kind"], "relation_graph_two_head_critic")
        self.assertTrue(calibrated.metadata["runtime_controller_use_edit"])
        self.assertTrue(calibrated.metadata["runtime_controller_use_commit"])
        self.assertFalse(calibrated.metadata.get("runtime_controller_disable_calibration", False))

        no_commit = prepare_instance_for_method_plan(
            self._instance(),
            plan=ABLATION_METHOD_PLANS["ours-eig-critic-no-commit"],
        )
        self.assertEqual(no_commit.metadata["method_name"], "ours-eig-critic-no-commit")
        self.assertTrue(no_commit.metadata["runtime_controller_use_edit"])
        self.assertFalse(no_commit.metadata["runtime_controller_use_commit"])
        self.assertTrue(no_commit.metadata["runtime_controller_disable_calibration"])

        no_edit = prepare_instance_for_method_plan(
            self._instance(),
            plan=ABLATION_METHOD_PLANS["ours-eig-critic-no-edit"],
        )
        self.assertEqual(no_edit.metadata["method_name"], "ours-eig-critic-no-edit")
        self.assertFalse(no_edit.metadata["runtime_controller_use_edit"])
        self.assertTrue(no_edit.metadata["runtime_controller_use_commit"])
        self.assertFalse(no_edit.metadata.get("runtime_controller_disable_calibration", False))
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
python -m pytest tests/test_experiment_plans.py::ExperimentPlanTests::test_ablation_method_plan_includes_controller_variants -q
```

Expected: FAIL because the three method-plan keys do not exist.

- [ ] **Step 3: Add the minimal method-plan entries**

In `src/idea_graph/experiment_plans.py`, add:

```python
    "ours-eig-critic-calibrated": ExperimentMethodPlan(
        name="ours-eig-critic-calibrated",
        baseline_name="ours-eig-critic-graph-twohead",
        restarts=1,
        max_rounds=5,
        stop_when_mature=True,
        runtime_protocol="parallel_graph_v2",
        rationale="Parallel EIG with the two-head graph critic and frozen-dev controller calibration.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_twohead_calibrated",
            "runtime_controller_use_edit": True,
            "runtime_controller_use_commit": True,
        },
    ),
    "ours-eig-critic-no-commit": ExperimentMethodPlan(
        name="ours-eig-critic-no-commit",
        baseline_name="ours-eig-critic-graph-twohead",
        restarts=1,
        max_rounds=5,
        stop_when_mature=False,
        runtime_protocol="parallel_graph_v2",
        rationale="Parallel EIG where the graph critic selects edits, but stopping is fixed-horizon.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_twohead_no_commit",
            "runtime_controller_disable_calibration": True,
            "runtime_controller_use_edit": True,
            "runtime_controller_use_commit": False,
        },
    ),
    "ours-eig-critic-no-edit": ExperimentMethodPlan(
        name="ours-eig-critic-no-edit",
        baseline_name="ours-eig-critic-graph-twohead",
        restarts=1,
        max_rounds=5,
        stop_when_mature=True,
        runtime_protocol="parallel_graph_v2",
        rationale="Parallel EIG where heuristic role-local edits are retained and the learned commit head controls stopping.",
        metadata_overrides={
            "idea_graph_protocol_variant": "eig_parallel_v2_twohead_no_edit",
            "runtime_controller_use_edit": False,
            "runtime_controller_use_commit": True,
        },
    ),
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```powershell
python -m pytest tests/test_experiment_plans.py::ExperimentPlanTests::test_ablation_method_plan_includes_controller_variants -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/idea_graph/experiment_plans.py tests/test_experiment_plans.py docs/superpowers/plans/2026-04-18-controller-paper-eval-and-ablation-execution.md
git commit -m "feat: add controller ablation method plans"
```

## Task 2: Runtime Metadata For Edit-Head Enablement

**Files:**
- Modify: `tests/test_benchmark_mode_and_baselines.py`
- Modify: `tests/test_joint_controller_calibration.py`
- Modify: `src/idea_graph/baselines.py`
- Modify: `src/idea_graph/relation_graph_runtime_critic.py`

- [ ] **Step 1: Write failing metadata tests**

Update the two-head baseline tests so they assert:

```python
self.assertTrue(instance.metadata["runtime_controller_use_edit"])
```

Update the calibration-disable tests so reset defaults include:

```python
self.assertTrue(updated.metadata["runtime_controller_use_edit"])
```

and runtime config checks include:

```python
self.assertTrue(config.use_edit)
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
python -m pytest tests/test_benchmark_mode_and_baselines.py::BenchmarkModeAndBaselinesTests::test_attach_baseline_metadata_enables_two_head_relation_graph_runtime_defaults tests/test_joint_controller_calibration.py::JointControllerCalibrationTests::test_runtime_controller_loader_can_disable_joint_calibration -q
```

Expected: FAIL because `runtime_controller_use_edit` and `config.use_edit` do not exist.

- [ ] **Step 3: Add runtime metadata and config support**

In `src/idea_graph/baselines.py`:

```python
"runtime_controller_use_edit",
```

Add it to `TWO_HEAD_RUNTIME_CONTROLLER_DEFAULTS`:

```python
"runtime_controller_use_edit": True,
```

Set it for text and graph controllers:

```python
metadata["runtime_controller_use_edit"] = True
```

When building `RelationGraphRuntimeConfig`, pass:

```python
use_edit=bool(graph.metadata.get("runtime_controller_use_edit", True)),
```

In `src/idea_graph/relation_graph_runtime_critic.py`, update:

```python
@dataclass(frozen=True)
class RelationGraphRuntimeConfig:
    tau_override: float = 0.05
    tau_commit: float = 0.08
    gamma_commit: float = 0.60
    min_commit_round: int = 2
    use_edit: bool = True
    use_commit: bool = False
    guard_support_threshold: float = 0.66
    guard_support_gain_floor: float = 0.10
    guard_requires_contradiction_progress: bool = False
```

- [ ] **Step 4: Run the metadata tests**

Run:

```powershell
python -m pytest tests/test_benchmark_mode_and_baselines.py::BenchmarkModeAndBaselinesTests::test_attach_baseline_metadata_enables_two_head_relation_graph_runtime_defaults tests/test_joint_controller_calibration.py::JointControllerCalibrationTests::test_runtime_controller_loader_can_disable_joint_calibration -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/idea_graph/baselines.py src/idea_graph/relation_graph_runtime_critic.py tests/test_benchmark_mode_and_baselines.py tests/test_joint_controller_calibration.py
git commit -m "feat: track runtime edit-controller enablement"
```

## Task 3: No-Edit Runtime Behavior

**Files:**
- Modify: `tests/test_parallel_runtime.py`
- Modify: `src/idea_graph/parallel_runtime.py`

- [ ] **Step 1: Write the failing no-edit runtime test**

Add a test that constructs a fake runtime controller whose edit scorer would choose `skip`, sets `RelationGraphRuntimeConfig(use_edit=False, use_commit=True, gamma_commit=0.5, min_commit_round=1)`, runs one parallel role round, and asserts:

```python
assert not graph.metadata.get("runtime_controller_log")
assert result.post_round_commit.source in {"runtime_controller_commit", "runtime_controller_continue"}
```

The test must also assert that at least one non-skip heuristic decision is materialized, proving edit control was bypassed rather than selecting the critic's `skip`.

- [ ] **Step 2: Run the failing test**

Run:

```powershell
python -m pytest tests/test_parallel_runtime.py::test_parallel_runtime_no_edit_uses_heuristic_edits_but_learned_commit -q
```

Expected: FAIL because `_maybe_apply_runtime_controller` currently runs edit reranking whenever a controller is loaded.

- [ ] **Step 3: Implement no-edit bypass**

At the top of `_maybe_apply_runtime_controller(...)` in `src/idea_graph/parallel_runtime.py`, after loading `controller_config`, add:

```python
    if not bool(getattr(controller_config, "use_edit", True)):
        return raw_decisions, False
```

Leave `_runtime_commit_check(...)` unchanged because it already depends on `config.use_commit`.

- [ ] **Step 4: Run the no-edit test**

Run:

```powershell
python -m pytest tests/test_parallel_runtime.py::test_parallel_runtime_no_edit_uses_heuristic_edits_but_learned_commit -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/idea_graph/parallel_runtime.py tests/test_parallel_runtime.py
git commit -m "feat: support commit-only controller ablation"
```

## Task 4: Held-Out Experiment Execution Plan

**Files:**
- Create: `docs/superpowers/plans/2026-04-18-controller-heldout-experiment-launch.md`

- [ ] **Step 1: Write the launch plan**

Create a launch document with these exact batches:

```markdown
# Controller Held-Out Experiment Launch Plan

## Dev Freeze Gate

- Split: `critic_dev`, excluding the 12 groups used by `C:\eig_p2v2_calib_dev12_refit`
- Size: 48 groups
- Balance: 24 AI Idea Bench 2025, 24 LiveIdeaBench
- Methods:
  - `Ours-Heuristic-Parallel`
  - `Ours-Critic-Graph`
  - `Ours-Critic-Calibrated`
- Decision gate:
  - promote calibrated only if held-out dev mean score beats raw critic by at least `0.05`
  - otherwise keep calibrated as ablation

## Paper Eval Main Table

- Split: `data/splits/parallel_v2/paper_eval_v2_registry.jsonl`
- Size: 256 groups
- Shards: 4 shards of 64 groups
- Methods:
  - `direct`
  - `self-refine`
  - `ai-researcher`
  - `scipip`
  - `virsci`
  - `Ours-Heuristic-Parallel`
  - selected learned controller from the dev freeze gate

## Controller Ablation

- Split: deterministic 128-group subset from paper eval
- Balance: 64 AI Idea Bench 2025, 64 LiveIdeaBench
- Methods:
  - `Ours-Heuristic-Parallel`
  - `Ours-Critic-Graph`
  - `Ours-Critic-Calibrated`
  - `Ours-Critic-No-Commit`
  - `Ours-Critic-No-Edit`
```

- [ ] **Step 2: Commit**

Run:

```powershell
git add docs/superpowers/plans/2026-04-18-controller-heldout-experiment-launch.md
git commit -m "docs: plan controller heldout evaluation"
```

## Task 5: Final Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run focused controller tests**

Run:

```powershell
python -m pytest tests/test_experiment_plans.py tests/test_benchmark_mode_and_baselines.py tests/test_joint_controller_calibration.py tests/test_parallel_runtime.py -q
```

Expected: PASS.

- [ ] **Step 2: Run diff hygiene**

Run:

```powershell
git diff --check
git status --short --branch
```

Expected: no whitespace errors and a clean branch after commits.

- [ ] **Step 3: Push**

Run:

```powershell
git push
```

Expected: branch `feature/joint-controller-calibration` updated on origin.

## Self-Review

- Spec coverage:
  - paper-eval launch setting is covered by Task 4
  - ablation method definitions are covered by Task 1
  - `no-commit` behavior is represented by metadata and fixed horizon
  - `no-edit` behavior is covered by Task 2 and Task 3
  - verification and push are covered by Task 5
- Placeholder scan:
  - no `TODO` or `TBD` placeholders remain
  - all commands use concrete file paths or repo-relative paths
- Type consistency:
  - method names use `ours-eig-critic-*`
  - runtime metadata uses `runtime_controller_*`
  - config flag is `use_edit`, matching `use_commit`
