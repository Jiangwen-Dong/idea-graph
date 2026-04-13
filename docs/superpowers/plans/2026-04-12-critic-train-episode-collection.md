# Critic-Train Episode Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin, split-safe runner that selects only frozen `development_pool_v1 / critic_train` benchmark instances, emits a launch manifest, and optionally collects new `ours-eig` runs into a dedicated training-episode root for later online critic adaptation.

**Architecture:** Reuse the existing benchmark and generation stack instead of inventing a second batch framework. The new layer sits above `split_registry.jsonl` and `run_pipeline.py`: it filters allowed training groups, normalizes benchmark selectors, writes a human-auditable manifest plus collection metadata, and optionally launches per-instance runs into `outputs/graph_critic_online_episodes/<collection>/runs`. This keeps the protocol reviewer-safe because `critic_dev` and future `paper_eval` rows are blocked before any generation starts.

**Tech Stack:** Python 3.10+, existing `idea_graph` benchmark loaders and run pipeline, JSONL manifests, `pytest`

---

## File Map

### New Files

- Create: `src/idea_graph/critic_episode_collection.py`
  - split-registry loading
  - `critic_train` filtering
  - benchmark-selector parsing
  - launch-manifest construction
  - optional subprocess execution helpers
- Create: `scripts/collect_critic_train_episodes.py`
  - CLI for dry-run manifest generation and optional execution
- Create: `tests/test_critic_episode_collection.py`
  - unit and CLI dry-run coverage

### Files To Extend

- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/critic_pools.md`

## Task A: Lock Selection Rules And Manifest Schema

**Files:**
- Create: `tests/test_critic_episode_collection.py`
- Create: `src/idea_graph/critic_episode_collection.py`

- [ ] **Step A1: Write failing tests for split-registry filtering**

Test the required rules:
- only rows from `development_pool_v1`
- only rows with `partition_role=critic_train`
- only rows whose `allowed_usages` include `train_online_critic`
- optional filtering by `group_id` and `limit`

- [ ] **Step A2: Write failing tests for benchmark-selector parsing**

Cover:
- `AI_Idea_Bench_2025::ai-idea-bench-2025-13` -> benchmark `ai_idea_bench_2025`, index `13`
- `liveideabench::liveideabench-meteorology-0` -> benchmark `liveideabench`, index `0`

- [ ] **Step A3: Run the new test file and verify failure**

Run:
`python -m pytest tests/test_critic_episode_collection.py -q`

- [ ] **Step A4: Implement the minimal manifest builder**

Required manifest fields:
- `group_id`
- `benchmark`
- `instance_name`
- `pool_name`
- `partition_role`
- `benchmark_cli_name`
- `benchmark_index`
- `baseline_name`
- `max_rounds`
- `native_eval`

## Task B: Add The Thin Collection CLI

**Files:**
- Create: `scripts/collect_critic_train_episodes.py`
- Modify: `tests/test_critic_episode_collection.py`

- [ ] **Step B1: Add a failing CLI dry-run test**

The CLI should:
- read `split_registry.jsonl`
- build a manifest
- write collection metadata under a dedicated output root
- avoid launching subprocesses unless `--execute` is passed

- [ ] **Step B2: Implement the CLI**

Required inputs:
- `--split-registry`
- `--output-dir`

Recommended optional inputs:
- `--collection-name`
- `--pool-name`
- `--group-id`
- `--limit`
- `--baseline`
- `--max-rounds`
- `--native-eval`
- `--llm-config`
- `--benchmark-root`
- `--execute`

Required outputs:
- `launch_manifest.jsonl`
- `collection_config.json`
- `collection_summary.json`

Recommended execution outputs:
- `execution_results.jsonl`
- `logs/`
- `runs/`

## Task C: Keep Execution Separate, Safe, And Profiled

**Files:**
- Create: `src/idea_graph/critic_episode_collection.py`
- Modify: `tests/test_critic_episode_collection.py`

- [ ] **Step C1: Add execution helpers guarded by `--execute`**

Rules:
- never launch `critic_dev`
- never launch `paper_eval`
- write per-run stdout/stderr logs
- record exit code and resolved `run_dir`

- [ ] **Step C2: Profile successful runs using saved artifacts**

For each successful collected episode, record when available:
- prompt tokens
- completion tokens
- total tokens
- estimated cost
- local final score
- native final score

This supports later reviewer-facing overhead accounting.

## Task D: Verify And Update Active Docs

**Files:**
- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/critic_pools.md`

- [ ] **Step D1: Run the targeted collection tests**

Run:
`python -m pytest tests/test_critic_episode_collection.py -q`

- [ ] **Step D2: Run the broader critic packet**

Run:
`python -m pytest tests/test_critic_split_registry.py tests/test_critic_replay.py tests/test_online_text_critic.py tests/test_critic_episode_collection.py -q`

- [ ] **Step D3: Run one dry-run collection smoke**

Use the current frozen registry to verify:
- selected count matches expected `critic_train` rows
- manifest layout is stable
- execution remains opt-in

- [ ] **Step D4: Update docs**

Record:
- the new collection runner
- the exact frozen `critic_train` scope
- the next step after dry-run verification:
  a real `critic_train` collection slice followed by replay export and online adaptation
