# Split Registry And Paper-Eval Candidate Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the current graph-critic pool as development-only, add a canonical split registry artifact, and define the first untouched `paper_eval` candidate list so critic training and final benchmark evaluation are clearly separated.

**Architecture:** Reuse the existing partition-manifest machinery instead of inventing another split layer. Add one thin registry layer that consolidates benchmark instance role, pool name, and allowed usage into a single auditable artifact. Separately, define a human-readable `paper_eval` candidate manifest drawn from benchmark instances not present in the current development pool. Keep the current `critic_train / critic_dev` split stable.

**Tech Stack:** Python 3.10+, existing partition-manifest code, JSON/JSONL artifacts, Markdown docs, `pytest`

---

## File Map

### New Files

- Create: `src/idea_graph/critic_split_registry.py`
  - registry builder
  - allowed-usage derivation
- Create: `scripts/build_critic_split_registry.py`
  - CLI for registry generation
- Create: `tests/test_critic_split_registry.py`
- Create: `docs/critic_pools.md`
  - canonical human-readable note for development vs final pools
- Create: `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions/split_registry.jsonl`
  - generated artifact
- Create: `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions/split_registry_stats.json`
  - generated artifact
- Create: `outputs/graph_critic_datasets/paper_eval_candidate_pool_v1/README.md`
  - manually curated candidate list note
- Create: `outputs/graph_critic_datasets/paper_eval_candidate_pool_v1/candidate_instances.json`
  - first untouched candidate manifest

### Files To Extend

- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`

## Task A: Implement The Split Registry Layer

**Files:**
- Create: `src/idea_graph/critic_split_registry.py`
- Create: `tests/test_critic_split_registry.py`
- Create: `scripts/build_critic_split_registry.py`

- [ ] **Step A1: Write the failing tests**

```python
def test_build_split_registry_marks_development_pool_roles() -> None:
    rows = build_split_registry(partition_rows, pool_name="development_pool_v1")
    by_group = {row["group_id"]: row for row in rows}
    assert by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-13"]["pool_name"] == "development_pool_v1"
    assert "train_offline_critic" in by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-13"]["allowed_usages"]
    assert "paper_final_eval" not in by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-13"]["allowed_usages"]


def test_build_split_registry_marks_paper_eval_as_final_eval_only() -> None:
    rows = build_split_registry(paper_eval_partition_rows, pool_name="paper_eval_v1")
    by_group = {row["group_id"]: row for row in rows}
    assert by_group["AI_Idea_Bench_2025::heldout"]["partition_role"] == "paper_eval"
    assert by_group["AI_Idea_Bench_2025::heldout"]["allowed_usages"] == ["paper_final_eval"]
```

- [ ] **Step A2: Run the tests to verify failure**

Run:
`python -m pytest tests/test_critic_split_registry.py -q`

Expected:
- import failure because `critic_split_registry.py` does not exist yet

- [ ] **Step A3: Implement the minimal registry builder**

Required fields per row:
- `group_id`
- `benchmark`
- `instance_name`
- `pool_name`
- `partition_role`
- `source_split`
- `allowed_usages`
- `notes`

Rules:
- `critic_train` in `development_pool_v1`:
  - `train_offline_critic`
  - `train_online_critic`
  - `development_analysis`
- `critic_dev` in `development_pool_v1`:
  - `select_checkpoint`
  - `calibrate_threshold`
  - `development_analysis`
- `paper_eval` in `paper_eval_v1`:
  - `paper_final_eval`

- [ ] **Step A4: Add the CLI builder**

Run:
`python scripts/build_critic_split_registry.py --partition-manifest outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl --pool-name development_pool_v1 --output-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions`

- [ ] **Step A5: Re-run the registry tests**

Run:
`python -m pytest tests/test_critic_split_registry.py -q`

Expected:
- all split-registry tests pass

## Task B: Build The Current Development Registry Artifact

**Files:**
- Create: generated registry artifacts under `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions`

- [ ] **Step B1: Generate the current registry**

Command:
`python scripts/build_critic_split_registry.py --partition-manifest outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl --pool-name development_pool_v1 --output-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions`

- [ ] **Step B2: Verify the generated artifact**

Checks:
- all 11 current groups appear exactly once
- all rows are tagged `development_pool_v1`
- no row includes `paper_final_eval`

## Task C: Define The First Untouched `paper_eval` Candidate Manifest

**Files:**
- Create: `outputs/graph_critic_datasets/paper_eval_candidate_pool_v1/candidate_instances.json`
- Create: `outputs/graph_critic_datasets/paper_eval_candidate_pool_v1/README.md`
- Create: `docs/critic_pools.md`

- [ ] **Step C1: Select candidate instances conservatively**

Selection rules:
- do not reuse any instance already present in the 11-group development pool
- include both benchmarks
- prefer instances that are easy to identify and reproducible by benchmark index/name
- keep the list modest and explicit

Recommended initial target:
- `AI_Idea_Bench_2025`: 4-8 untouched candidate instances
- `LiveIdeaBench`: 4 untouched candidate instances

- [ ] **Step C2: Write the candidate manifest**

Required fields:
- `benchmark`
- `instance_name`
- `status`
- `intended_role`
- `notes`

Use:
- `status = proposed`
- `intended_role = paper_eval`

- [ ] **Step C3: Write the human-readable pool note**

`docs/critic_pools.md` should explain:
- current `development_pool_v1`
- future `paper_eval_candidate_pool_v1`
- what each pool may and may not be used for

## Task D: Verify And Record

**Files:**
- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`

- [ ] **Step D1: Run the targeted tests**

Run:
`python -m pytest tests/test_critic_split_registry.py -q`

- [ ] **Step D2: Run the broader graph-critic packet**

Run:
`python -m pytest tests/test_critic_partitions.py tests/test_critic_split_registry.py tests/test_online_text_critic.py tests/test_critic_replay.py tests/test_critic_policy.py -q`

- [ ] **Step D3: Update docs**

Record:
- current 11-group pool frozen as `development_pool_v1`
- split registry generated
- first untouched `paper_eval` candidate list defined
- next step is real train-group episode collection, not more split reshuffling
