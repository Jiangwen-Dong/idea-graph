# Task 4 Tiny Online Adaptation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first tiny `critic_train`-only online adaptation runner for the text critic, using the existing warm-start stack plus the new replay/policy layer, and evaluate the adapted scorer on `critic_dev` without touching future `paper_eval`.

**Architecture:** Keep the learned controller lightweight. The online stage does not introduce a second model family. Instead, it reuses the current text-critic path and performs batched mixed-buffer retraining over two sources: frozen offline `critic_train` candidate rows and new replay rows shaped like candidate-slate examples. The runner remains conservative: it accepts only `critic_train` replay rows, evaluates only on `critic_dev`, and writes artifacts that make the update provenance explicit.

**Tech Stack:** Python 3.10+, existing `idea_graph` warm-start code, JSONL replay artifacts, `scikit-learn`, `pytest`

---

## Core Refinement Before Coding

The replay layer must preserve **candidate-slate rows**, not only a chosen action summary. The online critic is a next-action ranker, so each online training state needs:

- `state_text`
- `candidate_text`
- `candidate_id`
- `is_logged_selected`
- `is_commit`
- `is_commit_positive_state`
- `targets`
- `group_id`
- `partition_role`

Without those fields, online adaptation would not have state-local negative candidates and would not be compatible with the current `CandidateExample` training path.

## File Map

### New Files

- Create: `scripts/run_online_text_critic_adaptation.py`
  - loads warm-start artifact, offline candidate dataset, partition manifest, and online replay rows
  - runs one mixed-buffer adaptation pass
  - writes artifacts and metrics

### Files To Extend

- Modify: `src/idea_graph/critic_replay.py`
  - document and validate candidate-slate-shaped replay rows
- Modify: `src/idea_graph/online_text_critic.py`
  - online-row filtering
  - online-row to `CandidateExample` conversion
  - mixed-buffer adaptation training/evaluation helpers
- Modify: `tests/test_critic_replay.py`
  - candidate-row shape checks if needed
- Modify: `tests/test_online_text_critic.py`
  - adaptation tests
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/eig_graph_critic_plan.md`

## Task A: Lock The Replay Row Contract To Candidate-Slate Shape

**Files:**
- Modify: `src/idea_graph/critic_replay.py`
- Modify: `tests/test_critic_replay.py`

- [ ] **Step A1: Add a failing replay test for candidate-slate row export**

```python
def test_episode_training_rows_preserve_candidate_text_fields() -> None:
    episode = make_candidate_slate_episode()
    rows = episode_to_training_rows(episode)
    assert rows[0]["state_text"]
    assert rows[0]["candidate_text"]
    assert "is_logged_selected" in rows[0]
    assert "is_commit" in rows[0]
```

- [ ] **Step A2: Run the replay test to verify failure if the contract is incomplete**

Run:
`python -m pytest tests/test_critic_replay.py -q`

- [ ] **Step A3: Implement minimal replay-row validation**

Required behavior:
- reject replay rows missing the core candidate-slate fields
- keep the replay contract narrow and text-critic-compatible

- [ ] **Step A4: Re-run the replay tests**

Run:
`python -m pytest tests/test_critic_replay.py -q`

## Task B: Add Online Adaptation Helpers In `online_text_critic.py`

**Files:**
- Modify: `src/idea_graph/online_text_critic.py`
- Modify: `tests/test_online_text_critic.py`

- [ ] **Step B1: Write failing tests for online-row filtering and conversion**

```python
def test_partition_rows_for_role_filters_candidate_rows_by_partition() -> None:
    rows = partition_rows_for_role(candidate_rows, partition_lookup, partition_role="critic_train")
    assert {row["group_id"] for row in rows} == {"g-train"}


def test_build_online_adaptation_examples_rejects_non_train_rows() -> None:
    with pytest.raises(ValueError, match="critic_train"):
        build_online_adaptation_examples(rows_with_dev_partition)
```

- [ ] **Step B2: Add a failing test for mixed-buffer retraining**

```python
def test_train_online_text_critic_adaptation_reports_offline_and_online_counts() -> None:
    result = train_online_text_critic_adaptation(...)
    assert result.metrics["validation_example_count"] == 4
    assert result.metadata["offline_example_count"] > 0
    assert result.metadata["online_example_count"] > 0
```

- [ ] **Step B3: Run the adaptation tests to verify failure**

Run:
`python -m pytest tests/test_online_text_critic.py -q`

- [ ] **Step B4: Implement the helpers**

Add:

```python
def partition_rows_for_role(...): ...
def build_online_adaptation_examples(...): ...
def train_online_text_critic_adaptation(...): ...
```

Design rules:
- online rows must all be `critic_train`
- dev evaluation stays `critic_dev`
- mixed training rows come from:
  - offline `critic_train` candidate rows
  - online replay rows
- retraining can refit the lightweight head from the mixed buffer; no new model family

- [ ] **Step B5: Re-run the adaptation tests**

Run:
`python -m pytest tests/test_online_text_critic.py -q`

## Task C: Add The Tiny Runner Script

**Files:**
- Create: `scripts/run_online_text_critic_adaptation.py`

- [ ] **Step C1: Implement the CLI**

Required inputs:
- `--candidate-dataset-dir`
- `--partition-manifest`
- `--online-buffer`
- `--output-dir`
- optional:
  - `--offline-fraction`
  - `--max-train-examples`
  - `--commit-positive-weight`

- [ ] **Step C2: Write explicit output artifacts**

Required outputs:
- `metrics.json`
- `metadata.json`
- `adaptation_config.json`
- `model.pkl`

Recommended outputs:
- `online_buffer_audit.json`
- `checkpoint_metrics.jsonl`

- [ ] **Step C3: Keep this runner evaluation-only on `critic_dev`**

Rules:
- never train on `critic_dev`
- never train on `paper_eval`
- fail loudly if the online buffer contains forbidden partition roles

## Task D: Verify And Smoke Check

**Files:**
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/eig_graph_critic_plan.md`

- [ ] **Step D1: Run the targeted adaptation packet**

Run:
`python -m pytest tests/test_critic_replay.py tests/test_online_text_critic.py tests/test_critic_policy.py -q`

- [ ] **Step D2: Run the broader graph-critic packet**

Run:
`python -m pytest tests/test_trajectory_dataset.py tests/test_critic_dataset.py tests/test_candidate_slate_dataset.py tests/test_critic_partitions.py tests/test_text_critic.py tests/test_online_text_critic.py tests/test_critic_replay.py tests/test_critic_policy.py -q`

- [ ] **Step D3: If feasible, run a small plumbing smoke check**

Allowed for engineering verification only:
- use a tiny replay JSONL built from existing `critic_train` rows
- mark it clearly as a smoke-only buffer, not paper evidence

- [ ] **Step D4: Update docs**

Record:
- the refined replay contract
- the tiny adaptation runner
- whether the smoke adaptation changed `critic_dev` metrics
- whether the next step is true episode collection or further runner cleanup

## Decision Gate After This Slice

Proceed to real train-group episode collection only if:

- online adaptation accepts only `critic_train` replay rows
- the runner writes a clean mixed-buffer artifact trail
- dev evaluation remains split-safe
- the graph-critic packet still passes

If those hold, the next stage is no longer training-plumbing work. It is collecting the first real train-group online episodes and comparing `offline` vs `offline + online`.
