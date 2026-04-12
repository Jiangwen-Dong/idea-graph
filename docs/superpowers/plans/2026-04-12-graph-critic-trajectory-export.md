# Graph Critic Trajectory Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export a graph-critic training dataset from saved run artifacts, including transition-level state snapshots and reviewer-facing profiling statistics.

**Architecture:** Add a new `trajectory_dataset` library that discovers run directories, extracts run metadata and trace usage, reconstructs timestamp-sliced pre-action graph states, and writes JSONL/JSON snapshot artifacts. Keep the CLI thin so later G2/G3 dataset builders can reuse the same parsing and profiling code.

**Tech Stack:** Python 3, stdlib `json`/`pathlib`/`datetime`, existing repo IO helpers, `unittest`, `pytest`

---

### Task 1: Add failing tests for trajectory discovery, traces, and profiling

**Files:**
- Create: `tests/test_trajectory_dataset.py`
- Check: `tests/test_fs_utils.py`

- [ ] **Step 1: Write the failing test file**

```python
class TrajectoryDatasetTests(unittest.TestCase):
    def test_discover_run_dirs_requires_summary_and_graph(self) -> None:
        ...

    def test_extract_trace_stats_and_costs(self) -> None:
        ...

    def test_build_dataset_profile_aggregates_counts(self) -> None:
        ...
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_trajectory_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing-symbol errors for `idea_graph.trajectory_dataset`

### Task 2: Add failing tests for timestamp reconstruction and contradiction resolution

**Files:**
- Modify: `tests/test_trajectory_dataset.py`

- [ ] **Step 1: Add state-reconstruction tests**

```python
    def test_reconstruct_state_before_action_filters_nodes_and_edges_by_timestamp(self) -> None:
        ...

    def test_reconstruct_state_infers_contradiction_resolution_from_later_repair(self) -> None:
        ...

    def test_build_transition_rows_only_exports_action_runs(self) -> None:
        ...
```

- [ ] **Step 2: Run the tests to verify the intended failures**

Run: `python -m pytest tests/test_trajectory_dataset.py -q`
Expected: FAIL on missing functions such as `reconstruct_state_before_action` and `build_transition_rows`

### Task 3: Implement the trajectory dataset library

**Files:**
- Create: `src/idea_graph/trajectory_dataset.py`
- Check: `src/idea_graph/fs_utils.py`

- [ ] **Step 1: Implement discovery and JSON helpers**

```python
def discover_run_dirs(input_roots: Sequence[Path]) -> list[Path]:
    ...

def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    ...
```

- [ ] **Step 2: Implement trace usage, runtime, and manifest extraction**

```python
def extract_trace_stats(graph_payload: Mapping[str, Any], *, pricing: PricingConfig | None = None) -> TraceStats:
    ...

def build_run_manifest_row(run_dir: Path, summary_payload: Mapping[str, Any], graph_payload: Mapping[str, Any], *, pricing: PricingConfig | None = None) -> dict[str, Any]:
    ...
```

- [ ] **Step 3: Implement snapshot reconstruction and transition export**

```python
def reconstruct_state_before_action(graph_payload: Mapping[str, Any], action_index: int) -> dict[str, Any]:
    ...

def build_transition_rows(... ) -> list[dict[str, Any]]:
    ...
```

- [ ] **Step 4: Implement dataset profile aggregation and file writing**

```python
def aggregate_dataset_profile(... ) -> dict[str, Any]:
    ...

def export_graph_critic_dataset(... ) -> ExportResult:
    ...
```

- [ ] **Step 5: Run the targeted tests**

Run: `python -m pytest tests/test_trajectory_dataset.py -q`
Expected: PASS

### Task 4: Add the thin exporter CLI and smoke-test it

**Files:**
- Create: `scripts/export_graph_critic_dataset.py`

- [ ] **Step 1: Add CLI argument parsing and call into the library**

```python
def build_parser() -> argparse.ArgumentParser:
    ...

def main() -> None:
    ...
```

- [ ] **Step 2: Run the test file again after wiring the CLI**

Run: `python -m pytest tests/test_trajectory_dataset.py -q`
Expected: PASS

- [ ] **Step 3: Run a smoke export on saved pilot artifacts**

Run: `python scripts/export_graph_critic_dataset.py --input-root outputs/m2_aiib_r009_diagnosis_safe_grounding --output-dir outputs/graph_critic_datasets --dataset-name smoke_r009_safe_grounding --limit-runs 4`
Expected: exit code `0` and files `run_manifest.jsonl`, `trajectory_examples.jsonl`, `dataset_profile.json`, `README.md`, and `state_snapshots/` under `outputs/graph_critic_datasets/smoke_r009_safe_grounding`

### Task 5: Update active experiment records

**Files:**
- Modify: `docs/paper_experiment_tracker.md`
- Modify: `docs/experiment_execution_log.md`

- [ ] **Step 1: Record what G1 now exports**

```md
- trajectory export implemented
- manifest / transition / profile files verified
- current usable run and transition counts recorded from smoke export
```

- [ ] **Step 2: Record next-step dependencies for G2 dataset construction**

```md
- split policy by benchmark instance
- weak-label vs native-label separation
- later overhead profiling table fields
```

- [ ] **Step 3: Re-run focused verification before reporting completion**

Run: `python -m pytest tests/test_trajectory_dataset.py -q`
Expected: PASS

Run: `git status --short`
Expected: shows the new plan, tests, module, script, and doc updates without reverting unrelated work
