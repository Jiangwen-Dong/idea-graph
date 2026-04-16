# Parallel Runtime V2 Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a feature-flagged `parallel_graph_v2` runtime inside `idea-graph` that preserves the current benchmark-facing input/output contract while switching same-round execution to frozen-snapshot parallel proposals with replay-grade tracing.

**Architecture:** Keep the current sequential runtime intact as `sequential_v1`, and add a separate parallel coordinator path that freezes the round snapshot, gathers one slate per active role, selects at most one action per role, materializes the selected actions in deterministic order, and records per-round trace artifacts. Reuse the existing action schema, final proposal pipeline, and graph mutation logic where possible so the external protocol stays stable.

**Tech Stack:** Python 3.10+, stdlib `dataclasses`, `concurrent.futures`, pytest, existing `idea_graph` runtime modules

---

## File Structure

Runtime-facing files:

- `src/idea_graph/engine.py`
  Keep `run_experiment(...)` as the public entrypoint, add a `runtime_protocol` switch, and delegate to either sequential or parallel round execution.
- `src/idea_graph/models.py`
  Add lightweight dataclasses for parallel round decisions and replay-grade round traces.
- `src/idea_graph/parallel_runtime.py`
  New frozen-round coordinator for `parallel_graph_v2`.
- `src/idea_graph/parallel_role_executor.py`
  New helper for active-role proposal collection and deterministic fallback handling.
- `src/idea_graph/role_activation.py`
  New heuristic role-activation helper.
- `src/idea_graph/parallel_replay.py`
  New serializer for round trace payloads stored in `graph.metadata`.

Tests:

- `tests/test_engine.py`
  Add end-to-end protocol-switch and round-order behavior tests.
- `tests/test_parallel_runtime.py`
  Add focused tests for frozen-snapshot semantics, explicit `skip`, and deterministic materialization.
- `tests/test_parallel_replay.py`
  Add trace-shape tests for per-round replay artifacts.

## Task 1: Add Runtime Protocol Configuration

**Files:**
- Modify: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/engine.py`
- Test: `.worktrees/parallel-runtime-v2-exec/tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_engine.py`:

```python
    def test_run_experiment_accepts_parallel_runtime_protocol(self) -> None:
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            metadata={"runtime_protocol": "parallel_graph_v2"},
            max_rounds=1,
            stop_when_mature=False,
        )

        self.assertEqual(graph.metadata.get("runtime_protocol"), "parallel_graph_v2")
        self.assertEqual(graph.metadata.get("executed_round_count"), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_engine.py::EngineTests::test_run_experiment_accepts_parallel_runtime_protocol -q
```

Expected: FAIL because `run_experiment(...)` does not yet persist or honor the new runtime protocol.

- [ ] **Step 3: Write minimal implementation**

Add protocol metadata handling in `src/idea_graph/engine.py` immediately after `graph` is created:

```python
runtime_protocol = (
    str(graph.metadata.get("runtime_protocol", "sequential_v1")).strip()
    or "sequential_v1"
)
graph.metadata["runtime_protocol"] = runtime_protocol
```

Do not delegate to the parallel runtime yet in this task. The test only locks
the public protocol metadata so later tasks can safely branch on it.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_engine.py::EngineTests::test_run_experiment_accepts_parallel_runtime_protocol -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_engine.py src/idea_graph/engine.py
git commit -m "feat: add runtime protocol switch"
```

## Task 2: Add Frozen Parallel Round Data Structures

**Files:**
- Modify: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/models.py`
- Create: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/parallel_runtime.py`
- Test: `.worktrees/parallel-runtime-v2-exec/tests/test_parallel_runtime.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_parallel_runtime.py`:

```python
from idea_graph.models import IdeaGraph, GraphAction, Branch, Node
from idea_graph.parallel_runtime import ParallelRoleRoundResult


def test_parallel_round_result_tracks_selected_actions_and_skips() -> None:
    result = ParallelRoleRoundResult(
        round_name="Round1",
        active_roles=("MechanismProposer", "EvaluationDesigner"),
        skipped_roles=("EvaluationDesigner",),
        selected_actions=(),
        termination_reason="continue",
    )

    assert result.round_name == "Round1"
    assert result.skipped_roles == ("EvaluationDesigner",)
    assert result.termination_reason == "continue"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_parallel_runtime.py::test_parallel_round_result_tracks_selected_actions_and_skips -q
```

Expected: FAIL with import or attribute errors because the new dataclass does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add to `src/idea_graph/models.py`:

```python
@dataclass(frozen=True)
class ParallelRoleRoundResult:
    round_name: str
    active_roles: tuple[str, ...]
    skipped_roles: tuple[str, ...]
    selected_actions: tuple[GraphAction, ...]
    termination_reason: str
```

Add the first explicit export in `src/idea_graph/parallel_runtime.py`:

```python
from .models import ParallelRoleRoundResult

__all__ = ["ParallelRoleRoundResult"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_parallel_runtime.py::test_parallel_round_result_tracks_selected_actions_and_skips -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_parallel_runtime.py src/idea_graph/models.py src/idea_graph/parallel_runtime.py
git commit -m "feat: add parallel round result model"
```

## Task 3: Implement Frozen-Snapshot Parallel Proposal Execution

**Files:**
- Create: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/parallel_role_executor.py`
- Create: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/role_activation.py`
- Modify: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/parallel_runtime.py`
- Test: `.worktrees/parallel-runtime-v2-exec/tests/test_parallel_runtime.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_parallel_runtime.py`:

```python
from idea_graph.models import IdeaGraph
from idea_graph.parallel_runtime import execute_parallel_role_round


def test_parallel_role_round_uses_frozen_snapshot_for_all_roles() -> None:
    observed_node_counts = []

    class FrozenSnapshotBackend:
        name = "frozen-snapshot-test"

        def generate_seed(self, graph, role):
            raise RuntimeError("seed generation not used")

        def choose_action(self, graph, round_name, role):
            observed_node_counts.append((role, len(graph.active_nodes())))
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": next(branch.id for branch in graph.branches.values() if branch.role == role)},
                rationale="skip-like no-op for frozen snapshot test",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=FrozenSnapshotBackend(),
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    counts = {count for _, count in observed_node_counts}
    assert len(counts) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_parallel_runtime.py::test_parallel_role_round_uses_frozen_snapshot_for_all_roles -q
```

Expected: FAIL because `execute_parallel_role_round(...)` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create a role activation helper in `src/idea_graph/role_activation.py`:

```python
from .schema import ROLE_NAMES


def active_roles_for_round(graph, round_name: str) -> tuple[str, ...]:
    del graph, round_name
    return tuple(ROLE_NAMES)
```

Create a parallel executor in `src/idea_graph/parallel_role_executor.py`:

```python
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor


def collect_parallel_role_decisions(graph, round_name, collaboration_backend, roles):
    snapshot = deepcopy(graph)

    def _run(role: str):
        return role, collaboration_backend.choose_action(snapshot, round_name, role)

    with ThreadPoolExecutor(max_workers=max(1, len(roles))) as pool:
        return list(pool.map(_run, roles))
```

Then add a minimal frozen-round coordinator in `src/idea_graph/parallel_runtime.py`:

```python
from .parallel_role_executor import collect_parallel_role_decisions
from .role_activation import active_roles_for_round


def execute_parallel_role_round(
    graph,
    *,
    round_name,
    collaboration_backend,
    runtime_controller,
    runtime_controller_metadata,
    progress_callback,
):
    del runtime_controller, runtime_controller_metadata, progress_callback
    roles = active_roles_for_round(graph, round_name)
    return collect_parallel_role_decisions(graph, round_name, collaboration_backend, roles)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_parallel_runtime.py::test_parallel_role_round_uses_frozen_snapshot_for_all_roles -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_parallel_runtime.py src/idea_graph/parallel_runtime.py src/idea_graph/parallel_role_executor.py src/idea_graph/role_activation.py
git commit -m "feat: add frozen parallel role execution"
```

## Task 4: Add Explicit Skip And Deterministic Materialization

**Files:**
- Modify: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/parallel_runtime.py`
- Modify: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/agent_backend.py`
- Modify: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/collaboration_protocol.py`
- Test: `.worktrees/parallel-runtime-v2-exec/tests/test_parallel_runtime.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_parallel_runtime.py`:

```python
def test_parallel_round_materializes_non_skip_actions_in_role_order() -> None:
    class OrderedBackend:
        name = "ordered-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("not used")

        def choose_action(self, graph, round_name, role):
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            if role == "EvaluationDesigner":
                return ActionDecision(kind="skip", target_ids=[], payload={"branch_id": branch_id}, rationale="skip")
            return ActionDecision(kind="freeze_branch", target_ids=[], payload={"branch_id": branch_id}, rationale=role)

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=OrderedBackend(),
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    assert "EvaluationDesigner" in result.skipped_roles
    assert [action.role for action in result.selected_actions] == sorted(action.role for action in result.selected_actions)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_parallel_runtime.py::test_parallel_round_materializes_non_skip_actions_in_role_order -q
```

Expected: FAIL because skip handling and deterministic selected-action output are not implemented yet.

- [ ] **Step 3: Write minimal implementation**

Add `skip` to the action hints in `src/idea_graph/collaboration_protocol.py` and allow the backend prompt layer to surface it.

Then update `src/idea_graph/parallel_runtime.py`:

```python
from .engine import action_from_decision, apply_action


def execute_parallel_role_round(
    graph,
    *,
    round_name,
    collaboration_backend,
    runtime_controller,
    runtime_controller_metadata,
    progress_callback,
):
    del runtime_controller, runtime_controller_metadata, progress_callback
    roles = active_roles_for_round(graph, round_name)
    raw_decisions = collect_parallel_role_decisions(
        graph,
        round_name,
        collaboration_backend,
        roles,
    )
    selected_actions = []
    skipped_roles = []
    for role, decision in sorted(raw_decisions, key=lambda item: item[0]):
        if str(decision.kind).strip() == "skip":
            skipped_roles.append(role)
            continue
        action = action_from_decision(graph, round_name=round_name, role=role, decision=decision)
        action.source = "parallel_llm"
        apply_action(graph, action)
        selected_actions.append(action)
    return ParallelRoleRoundResult(
        round_name=round_name,
        active_roles=tuple(role for role, _ in raw_decisions),
        skipped_roles=tuple(skipped_roles),
        selected_actions=tuple(selected_actions),
        termination_reason="continue",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_parallel_runtime.py::test_parallel_round_materializes_non_skip_actions_in_role_order -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_parallel_runtime.py src/idea_graph/parallel_runtime.py src/idea_graph/agent_backend.py src/idea_graph/collaboration_protocol.py
git commit -m "feat: add skip and deterministic materialization"
```

## Task 5: Persist Replay-Grade Round Traces

**Files:**
- Create: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/parallel_replay.py`
- Modify: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/parallel_runtime.py`
- Modify: `.worktrees/parallel-runtime-v2-exec/src/idea_graph/engine.py`
- Test: `.worktrees/parallel-runtime-v2-exec/tests/test_parallel_replay.py`
- Test: `.worktrees/parallel-runtime-v2-exec/tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_parallel_replay.py`:

```python
from idea_graph.parallel_replay import append_parallel_round_trace


def test_append_parallel_round_trace_persists_round_payload_in_metadata() -> None:
    metadata = {}
    append_parallel_round_trace(
        metadata,
        {
            "round": "Round1",
            "active_roles": ["MechanismProposer"],
            "inactive_roles": ["EvaluationDesigner"],
            "selected_actions": [],
            "skipped_roles": ["MechanismProposer"],
        },
    )

    traces = metadata.get("parallel_round_traces")
    assert isinstance(traces, list)
    assert traces[0]["round"] == "Round1"
```

Append to `tests/test_engine.py`:

```python
    def test_parallel_runtime_records_round_traces(self) -> None:
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            metadata={"runtime_protocol": "parallel_graph_v2"},
            max_rounds=1,
            stop_when_mature=False,
        )

        traces = graph.metadata.get("parallel_round_traces")
        self.assertTrue(traces)
        self.assertEqual(traces[0]["round"], "Round1")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_parallel_replay.py tests/test_engine.py::EngineTests::test_parallel_runtime_records_round_traces -q
```

Expected: FAIL because the replay helper and trace persistence do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `src/idea_graph/parallel_replay.py`:

```python
from __future__ import annotations


def append_parallel_round_trace(metadata: dict[str, object], payload: dict[str, object]) -> None:
    traces = metadata.setdefault("parallel_round_traces", [])
    if isinstance(traces, list):
        traces.append(dict(payload))
```

Update `src/idea_graph/parallel_runtime.py` to call it after each round:

```python
append_parallel_round_trace(
    graph.metadata,
    {
        "round": result.round_name,
        "active_roles": list(result.active_roles),
        "inactive_roles": [role for role in ROLE_NAMES if role not in result.active_roles],
        "selected_actions": [action.id for action in result.selected_actions],
        "skipped_roles": list(result.skipped_roles),
        "termination_reason": result.termination_reason,
    },
)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_parallel_replay.py tests/test_engine.py::EngineTests::test_parallel_runtime_records_round_traces -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_parallel_replay.py tests/test_engine.py src/idea_graph/parallel_replay.py src/idea_graph/parallel_runtime.py src/idea_graph/engine.py
git commit -m "feat: trace parallel runtime rounds"
```

## Task 6: Verify Phase 1 Slice

**Files:**
- Test: `.worktrees/parallel-runtime-v2-exec/tests/test_engine.py`
- Test: `.worktrees/parallel-runtime-v2-exec/tests/test_parallel_runtime.py`
- Test: `.worktrees/parallel-runtime-v2-exec/tests/test_parallel_replay.py`

- [ ] **Step 1: Run focused verification**

Run:

```bash
python -m pytest tests/test_engine.py tests/test_parallel_runtime.py tests/test_parallel_replay.py tests/test_relation_graph_runtime_critic.py -q
```

Expected: PASS

- [ ] **Step 2: Run the previously green baseline slice**

Run:

```bash
python -m pytest tests/test_engine.py tests/test_relation_graph_runtime_critic.py -q
```

Expected: PASS

- [ ] **Step 3: Commit verification-only metadata if needed**

```bash
git status --short
```

Expected: no uncommitted changes
