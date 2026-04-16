# Parallel Runtime V2 Two-Head Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fully implement the approved `parallel_graph_v2` pipeline inside `idea-graph`, including post-round-only commit semantics, replay artifacts for high-quality supervision, and a shared-encoder two-head graph critic training path.

**Architecture:** Extend the already-landed frozen-snapshot parallel runtime rather than replacing it. First upgrade runtime objects and traces so the repo records `selected_role_decisions`, `edit_patches`, `materialized_graph_actions`, and post-round commit supervision cleanly. Then add a separate two-head data/model/training path that reuses the existing relation-graph encoding logic where possible while keeping the old single-head path intact for compatibility and ablation.

**Tech Stack:** Python 3, pytest, JSONL dataset exports, PyTorch, existing `idea_graph` runtime/controller infrastructure.

---

## File Structure

- Modify: `src/idea_graph/models.py`
  Purpose: add explicit runtime record types for selected role decisions, edit patches, and post-round commit checks.
- Modify: `src/idea_graph/parallel_runtime.py`
  Purpose: return the richer runtime records, emit empty patches for `skip`, and evaluate a post-round commit decision only after merge.
- Modify: `src/idea_graph/parallel_replay.py`
  Purpose: write richer round traces plus separate edit-row and post-round commit-row payloads.
- Modify: `src/idea_graph/engine.py`
  Purpose: store upgraded trace objects in `graph.metadata`, stop on post-round commit only, and preserve benchmark-facing behavior.
- Modify: `src/idea_graph/trajectory_dataset.py`
  Purpose: export `post_round_commit_examples.jsonl`, emit a quality report, and keep compatibility exports for existing consumers.
- Modify: `src/idea_graph/candidate_slate_dataset.py`
  Purpose: keep the existing compatibility adapter while adding a parallel two-head dataset builder with leak-safe grouped splits.
- Create: `src/idea_graph/relation_graph_two_head_data.py`
  Purpose: load edit rows and post-round commit rows into shared graph examples for training.
- Create: `src/idea_graph/relation_graph_two_head_model.py`
  Purpose: shared encoder plus `EditHead` and `CommitHead`.
- Create: `src/idea_graph/relation_graph_two_head_train.py`
  Purpose: train edit ranking and commit classification jointly and write metrics/checkpoints.
- Create: `scripts/build_parallel_two_head_dataset.py`
  Purpose: build the new edit/commit dataset package from exported replay.
- Create: `scripts/train_relation_graph_two_head_critic.py`
  Purpose: train the new two-head critic from the curated dataset package.
- Modify: `tests/test_parallel_runtime.py`
  Purpose: lock the new runtime object semantics.
- Modify: `tests/test_engine.py`
  Purpose: lock upgraded metadata traces and post-round commit behavior.
- Modify: `tests/test_trajectory_dataset.py`
  Purpose: verify export of post-round commit rows and the quality report.
- Modify: `tests/test_candidate_slate_dataset.py`
  Purpose: verify the two-head dataset builder and grouped split behavior.
- Create: `tests/test_relation_graph_two_head_data.py`
  Purpose: verify edit/commit dataset loading and snapshot reuse.
- Create: `tests/test_relation_graph_two_head_model.py`
  Purpose: verify encoder/head tensor shapes and separate forward paths.
- Create: `tests/test_relation_graph_two_head_train.py`
  Purpose: verify a minimal training smoke and artifact writing.

## Task 1: Upgrade Parallel Runtime Records

**Files:**
- Modify: `src/idea_graph/models.py`
- Modify: `src/idea_graph/parallel_runtime.py`
- Modify: `src/idea_graph/parallel_replay.py`
- Modify: `src/idea_graph/engine.py`
- Test: `tests/test_parallel_runtime.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing runtime record tests**

```python
def test_parallel_round_result_tracks_decisions_patches_and_commit_check() -> None:
    result = ParallelRoleRoundResult(
        round_name="Round1",
        active_roles=("MechanismProposer",),
        skipped_roles=(),
        selected_role_decisions=(
            ParallelRoleDecisionRecord(
                role="MechanismProposer",
                kind="freeze_branch",
                target_ids=(),
                payload={"branch_id": "B001"},
                rationale="freeze",
            ),
        ),
        edit_patches=(
            ParallelEditPatchRecord(
                role="MechanismProposer",
                kind="freeze_branch",
                target_ids=(),
                payload={"branch_id": "B001"},
                is_empty=False,
            ),
        ),
        materialized_graph_actions=(),
        post_round_commit=ParallelCommitCheckRecord(
            round_name="Round1",
            state_kind="parallel_post_round",
            should_commit=False,
            source="maturity_snapshot",
        ),
    )

    assert result.selected_role_decisions[0].role == "MechanismProposer"
    assert result.edit_patches[0].is_empty is False
    assert result.post_round_commit.should_commit is False
```

```python
def test_parallel_runtime_records_selected_role_decisions_and_post_round_commit() -> None:
    graph = run_experiment(
        topic="graph-based scientific ideation",
        literature=["paper a", "paper b", "paper c", "paper d"],
        metadata={"runtime_protocol": "parallel_graph_v2"},
        max_rounds=1,
        stop_when_mature=False,
    )

    trace = graph.metadata["parallel_round_traces"][0]
    assert "selected_role_decisions" in trace
    assert "edit_patches" in trace
    assert "materialized_graph_actions" in trace
    assert "post_round_commit" in trace
    assert "selected_actions" not in trace
```

- [ ] **Step 2: Run the targeted tests to verify RED**

Run: `python -m pytest tests/test_parallel_runtime.py tests/test_engine.py -q`
Expected: FAIL because the current models and engine traces still use `selected_actions` and `termination_reason`.

- [ ] **Step 3: Implement the runtime record upgrade**

```python
@dataclass(frozen=True)
class ParallelRoleDecisionRecord:
    role: str
    kind: str
    target_ids: tuple[str, ...]
    payload: dict[str, object]
    rationale: str


@dataclass(frozen=True)
class ParallelEditPatchRecord:
    role: str
    kind: str
    target_ids: tuple[str, ...]
    payload: dict[str, object]
    is_empty: bool


@dataclass(frozen=True)
class ParallelCommitCheckRecord:
    round_name: str
    state_kind: str
    should_commit: bool
    source: str
    support_coverage: float = 0.0
    unresolved_contradiction_ratio: float = 0.0
    utility: float = 0.0
```

```python
selected_role_decisions = []
edit_patches = []
materialized_graph_actions = []

for role, decision in sorted(raw_decisions, key=lambda item: item[0]):
    selected_role_decisions.append(_decision_record(role, decision))
    if str(decision.kind).strip() == "skip":
        skipped_roles.append(role)
        edit_patches.append(_empty_patch_record(role, decision))
        continue
    edit_patches.append(_patch_record(role, decision))
    action = action_from_decision(...)
    apply_action(graph, action)
    materialized_graph_actions.append(action)

snapshot = maturity_snapshot(graph)
post_round_commit = ParallelCommitCheckRecord(
    round_name=round_name,
    state_kind="parallel_post_round",
    should_commit=bool(snapshot.is_mature),
    source="maturity_snapshot",
    support_coverage=float(snapshot.support_coverage),
    unresolved_contradiction_ratio=float(snapshot.unresolved_contradiction_ratio),
    utility=float(snapshot.utility),
)
```

- [ ] **Step 4: Re-run the targeted tests to verify GREEN**

Run: `python -m pytest tests/test_parallel_runtime.py tests/test_engine.py -q`
Expected: PASS

- [ ] **Step 5: Commit the runtime upgrade**

```bash
git add src/idea_graph/models.py src/idea_graph/parallel_runtime.py src/idea_graph/parallel_replay.py src/idea_graph/engine.py tests/test_parallel_runtime.py tests/test_engine.py
git commit -m "feat: record parallel decisions and post-round commit"
```

## Task 2: Export High-Quality Replay Labels

**Files:**
- Modify: `src/idea_graph/trajectory_dataset.py`
- Modify: `src/idea_graph/parallel_replay.py`
- Test: `tests/test_trajectory_dataset.py`

- [ ] **Step 1: Write the failing export tests**

```python
def test_export_graph_critic_dataset_writes_post_round_commit_examples() -> None:
    result = export_graph_critic_dataset(
        input_roots=[fixture_run_dir],
        output_dir=tmp_path,
        dataset_name="parallel-export",
    )

    rows = [
        json.loads(line)
        for line in (result.dataset_dir / "post_round_commit_examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[0]["state_kind"] == "parallel_post_round"
    assert rows[0]["commit_supervision"]["available"] is True
```

```python
def test_parallel_quality_report_counts_missing_or_ambiguous_labels() -> None:
    profile = aggregate_parallel_edit_profile(
        [
            {"state_id": "s1", "candidate_kind": "skip", "is_logged_selected": True, "role": "MechanismProposer"},
            {"state_id": "s1", "candidate_kind": "freeze_branch", "is_logged_selected": False, "role": "MechanismProposer"},
        ]
    )
    assert "selected_skip_fraction" in profile
```

- [ ] **Step 2: Run the targeted tests to verify RED**

Run: `python -m pytest tests/test_trajectory_dataset.py -q`
Expected: FAIL because `post_round_commit_examples.jsonl` is not exported yet.

- [ ] **Step 3: Implement replay export and quality rules**

```python
def build_post_round_commit_rows(
    run_dir: Path,
    summary_payload: Mapping[str, Any],
    graph_payload: Mapping[str, Any],
    *,
    snapshot_dir: Path,
) -> list[dict[str, Any]]:
    traces = graph_payload.get("metadata", {}).get("parallel_round_traces", [])
    rows: list[dict[str, Any]] = []
    for round_index, trace in enumerate(traces):
        post_round_commit = _as_object_dict(trace.get("post_round_commit"))
        if not post_round_commit:
            continue
        rows.append(
            {
                "state_id": f"{run_dir.resolve()}::post-round::{round_index:03d}",
                "round_name": str(trace.get("round", "")).strip(),
                "state_kind": "parallel_post_round",
                "commit_supervision": {
                    "available": True,
                    "label": 1 if bool(post_round_commit.get("should_commit")) else 0,
                    "source": str(post_round_commit.get("source", "")).strip() or "parallel_post_round",
                },
            }
        )
    return rows
```

```python
write_text_file(dataset_dir / "post_round_commit_examples.jsonl", _jsonl_lines(post_round_commit_rows))
write_text_file(
    dataset_dir / "parallel_label_quality.json",
    json.dumps(
        {
            "edit_state_count": len({row["state_id"] for row in parallel_edit_rows}),
            "post_round_commit_state_count": len(post_round_commit_rows),
            "selected_skip_count": selected_skip_count,
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
)
```

- [ ] **Step 4: Re-run the targeted tests to verify GREEN**

Run: `python -m pytest tests/test_trajectory_dataset.py -q`
Expected: PASS

- [ ] **Step 5: Commit the replay export upgrade**

```bash
git add src/idea_graph/trajectory_dataset.py src/idea_graph/parallel_replay.py tests/test_trajectory_dataset.py
git commit -m "feat: export post-round commit supervision"
```

## Task 3: Build The Parallel Two-Head Dataset

**Files:**
- Modify: `src/idea_graph/candidate_slate_dataset.py`
- Create: `src/idea_graph/relation_graph_two_head_data.py`
- Create: `scripts/build_parallel_two_head_dataset.py`
- Test: `tests/test_candidate_slate_dataset.py`
- Test: `tests/test_relation_graph_two_head_data.py`

- [ ] **Step 1: Write the failing dataset-builder tests**

```python
def test_build_parallel_two_head_dataset_from_export_writes_edit_and_commit_rows(tmp_path: Path) -> None:
    result = build_parallel_two_head_dataset_from_export(
        g1_dataset_dir=g1_dir,
        output_dir=tmp_path,
        dataset_name="parallel-two-head",
    )

    edit_rows = [json.loads(line) for line in (result.dataset_dir / "edit_head_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    commit_rows = [json.loads(line) for line in (result.dataset_dir / "commit_head_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert edit_rows
    assert commit_rows
    assert all(row["state_kind"] == "parallel_pre_action" for row in edit_rows)
    assert all(row["state_kind"] == "parallel_post_round" for row in commit_rows)
```

```python
def test_two_head_dataset_loader_builds_train_and_dev_examples() -> None:
    dataset = build_relation_graph_two_head_dataset(
        dataset_dir=two_head_dataset_dir,
        g1_dataset_dir=g1_dir,
        partition_manifest_path=partition_manifest,
        text_backend=HashTextEmbeddingBackend(dim=8),
    )

    assert dataset.edit_train_examples
    assert dataset.commit_train_examples
    assert dataset.edit_dev_examples
    assert dataset.commit_dev_examples
```

- [ ] **Step 2: Run the targeted tests to verify RED**

Run: `python -m pytest tests/test_candidate_slate_dataset.py tests/test_relation_graph_two_head_data.py -q`
Expected: FAIL because the two-head dataset builder and loader do not exist yet.

- [ ] **Step 3: Implement the dataset package and loader**

```python
def build_parallel_two_head_dataset_from_export(
    *,
    g1_dataset_dir: Path,
    output_dir: Path,
    dataset_name: str,
    validation_fraction: float = 0.2,
) -> ParallelTwoHeadDatasetBuildResult:
    edit_rows, edit_manifest = build_parallel_candidate_dataset_rows(...)
    commit_rows, commit_manifest = build_parallel_commit_dataset_rows(...)
    split_rows = assign_group_splits(...)
    write_text_file(dataset_dir / "edit_head_rows.jsonl", _jsonl_lines(edit_rows))
    write_text_file(dataset_dir / "commit_head_rows.jsonl", _jsonl_lines(commit_rows))
    write_text_file(dataset_dir / "split_manifest.jsonl", _jsonl_lines(split_rows))
    return ParallelTwoHeadDatasetBuildResult(...)
```

```python
@dataclass(frozen=True)
class RelationGraphTwoHeadDataset:
    edit_train_examples: list[RelationGraphCandidateExample]
    edit_dev_examples: list[RelationGraphCandidateExample]
    commit_train_examples: list[RelationGraphCommitExample]
    commit_dev_examples: list[RelationGraphCommitExample]
    metadata: dict[str, Any]
```

- [ ] **Step 4: Re-run the targeted tests to verify GREEN**

Run: `python -m pytest tests/test_candidate_slate_dataset.py tests/test_relation_graph_two_head_data.py -q`
Expected: PASS

- [ ] **Step 5: Commit the two-head dataset builder**

```bash
git add src/idea_graph/candidate_slate_dataset.py src/idea_graph/relation_graph_two_head_data.py scripts/build_parallel_two_head_dataset.py tests/test_candidate_slate_dataset.py tests/test_relation_graph_two_head_data.py
git commit -m "feat: add parallel two-head critic dataset builder"
```

## Task 4: Implement Shared Encoder Plus Two Heads

**Files:**
- Create: `src/idea_graph/relation_graph_two_head_model.py`
- Create: `src/idea_graph/relation_graph_two_head_train.py`
- Create: `scripts/train_relation_graph_two_head_critic.py`
- Test: `tests/test_relation_graph_two_head_model.py`
- Test: `tests/test_relation_graph_two_head_train.py`

- [ ] **Step 1: Write the failing model and train tests**

```python
def test_two_head_model_returns_edit_scores_and_commit_logits() -> None:
    model = RelationGraphTwoHeadCritic(
        text_dim=8,
        hidden_dim=16,
        node_type_count=4,
        role_count=4,
        edge_type_count=4,
        candidate_kind_count=4,
    )
    edit_scores = model.score_edit_batch(edit_batch)
    commit_logits = model.score_commit_batch(commit_batch)
    assert edit_scores.shape[0] == edit_batch.labels.shape[0]
    assert commit_logits.shape[0] == commit_batch.labels.shape[0]
```

```python
def test_train_relation_graph_two_head_critic_writes_metrics(tmp_path: Path) -> None:
    artifacts = train_relation_graph_two_head_critic(
        dataset=dataset,
        output_dir=tmp_path,
        hidden_dim=16,
        batch_size=1,
        epochs=1,
        learning_rate=1e-3,
        text_backend_name="hash",
    )
    assert (tmp_path / "edit_metrics.json").exists()
    assert (tmp_path / "commit_metrics.json").exists()
    assert artifacts.model_path is not None
```

- [ ] **Step 2: Run the targeted tests to verify RED**

Run: `python -m pytest tests/test_relation_graph_two_head_model.py tests/test_relation_graph_two_head_train.py -q`
Expected: FAIL because the two-head model and trainer do not exist yet.

- [ ] **Step 3: Implement the shared encoder and training loop**

```python
class RelationGraphSharedEncoder(nn.Module):
    def encode_graph(self, batch: RelationGraphBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


class RelationGraphTwoHeadCritic(nn.Module):
    def __init__(...):
        self.encoder = RelationGraphSharedEncoder(...)
        self.edit_head = nn.Sequential(...)
        self.commit_head = nn.Sequential(...)

    def score_edit_batch(self, batch: RelationGraphBatch) -> torch.Tensor:
        graph_summary, target_summary, neighbor_summary, node_states = self.encoder.encode_graph(batch)
        ...

    def score_commit_batch(self, batch: RelationGraphCommitBatch) -> torch.Tensor:
        graph_summary, _, _, _ = self.encoder.encode_commit_graph(batch)
        return self.commit_head(graph_summary).squeeze(-1)
```

```python
edit_loss = compute_state_ranking_loss(edit_scores, edit_batch.labels, edit_batch.candidate_state_index)
commit_loss = nn.functional.binary_cross_entropy_with_logits(commit_logits, commit_batch.labels)
loss = edit_loss + commit_loss_weight * commit_loss
```

- [ ] **Step 4: Re-run the targeted tests to verify GREEN**

Run: `python -m pytest tests/test_relation_graph_two_head_model.py tests/test_relation_graph_two_head_train.py -q`
Expected: PASS

- [ ] **Step 5: Commit the two-head model**

```bash
git add src/idea_graph/relation_graph_two_head_model.py src/idea_graph/relation_graph_two_head_train.py scripts/train_relation_graph_two_head_critic.py tests/test_relation_graph_two_head_model.py tests/test_relation_graph_two_head_train.py
git commit -m "feat: add shared-encoder two-head graph critic"
```

## Task 5: Run Verification And Label-Curation Smoke

**Files:**
- Modify: `docs/superpowers/specs/2026-04-16-idea-graph-parallel-runtime-and-two-head-critic-design.md` (only if implementation reveals a spec mismatch)
- Output: `outputs/graph_critic_datasets/<dataset-name>/...`

- [ ] **Step 1: Run the focused verification suite**

Run: `python -m pytest tests/test_parallel_runtime.py tests/test_engine.py tests/test_trajectory_dataset.py tests/test_candidate_slate_dataset.py tests/test_relation_graph_two_head_data.py tests/test_relation_graph_two_head_model.py tests/test_relation_graph_two_head_train.py -q`
Expected: PASS

- [ ] **Step 2: Build a smoke two-head dataset from existing parallel exports**

Run:

```bash
python scripts/build_parallel_two_head_dataset.py --g1-dataset-dir outputs/graph_critic_datasets/<parallel-export> --output-dir outputs/graph_critic_datasets --dataset-name parallel-two-head-smoke
```

Expected: writes `edit_head_rows.jsonl`, `commit_head_rows.jsonl`, `split_manifest.jsonl`, and a quality report.

- [ ] **Step 3: Train a one-epoch smoke two-head critic**

Run:

```bash
python scripts/train_relation_graph_two_head_critic.py --dataset-dir outputs/graph_critic_datasets/parallel-two-head-smoke --g1-dataset-dir outputs/graph_critic_datasets/<parallel-export> --partition-manifest data/critic/partition_manifest.jsonl --output-dir outputs/graph_critic_datasets/parallel-two-head-smoke-model --text-backend hash --embedding-dim 64 --hidden-dim 64 --batch-size 4 --epochs 1 --lr 1e-3
```

Expected: writes a checkpoint plus edit and commit metrics JSON files.

- [ ] **Step 4: Inspect the smoke outputs for label quality**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
dataset_dir = Path("outputs/graph_critic_datasets/parallel-two-head-smoke")
print(json.dumps(json.loads((dataset_dir / "dataset_stats.json").read_text(encoding="utf-8")), indent=2))
print(json.dumps(json.loads((dataset_dir / "parallel_label_quality.json").read_text(encoding="utf-8")), indent=2))
PY
```

Expected: non-zero edit rows, non-zero commit rows, grouped splits, and skip/commit counts that look plausible.

- [ ] **Step 5: Commit the verification-safe implementation**

```bash
git add docs/superpowers/plans/2026-04-16-parallel-runtime-v2-two-head-critic.md
git commit -m "docs: add parallel two-head implementation plan"
```

## Plan Self-Review

- Spec coverage:
  - runtime protocol with post-round-only commit: covered by Task 1
  - high-quality replay labels and export semantics: covered by Task 2
  - separate edit/commit datasets: covered by Task 3
  - shared encoder with two heads and training: covered by Task 4
  - smoke curation and verification: covered by Task 5
- Placeholder scan:
  - no `TODO`, `TBD`, or “implement later” placeholders remain
- Type consistency:
  - plan consistently uses `selected_role_decisions`, `edit_patches`, `materialized_graph_actions`, and `post_round_commit`
