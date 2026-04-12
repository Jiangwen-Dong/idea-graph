# Online Warm-Start Critic And Batched Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a scalable two-stage learned controller for `EIG`: first train a lightweight critic offline from saved graph trajectories, then improve it with train-group-only batched online updates before freezing it for held-out evaluation.

**Architecture:** Reuse the existing `G1 -> G2 -> G2.5` dataset stack and keep the learned component small: frozen state/candidate features plus a trainable scorer head. The method has two stages: `offline warm start` from saved trajectories, followed by `episodic batched online adaptation` on train benchmark groups only. Validation groups are used for checkpoint selection and commit calibration. Final critic-controlled evaluation is always frozen and performed on untouched held-out benchmark instances.

**Tech Stack:** Python 3.10+, existing JSONL dataset builders, `scikit-learn` for the text pilot baseline, existing benchmark runners, existing benchmark-native evaluation, `pytest`

---

## Problem Framing

This plan resolves the main supervision concern in the current graph-critic track:

- logged selected actions come from the heuristic controller, so they are valid only as weak teacher labels
- the current positive `commit` signal must come from outcome-derived labels rather than from hand-designed rules alone
- the learned controller must remain cheap, stable, and adaptive enough to satisfy reviewer concerns about scalability

The method therefore uses four supervision namespaces:

1. `teacher_logged`
   heuristic-selected action within a candidate slate
2. `terminal_commit`
   final exported graph state where `commit` is positive
3. `hindsight_commit`
   sampled intermediate state where immediate synthesis is compared against continued evolution
4. `trajectory_return`
   final weak/local/native return attached to all state-action rows

Only `teacher_logged` and `terminal_commit` are required for the first implementation slice. `hindsight_commit` is added in the next dataset-enrichment pass. `trajectory_return` is auxiliary, not the sole optimization target.

## Canonical Data Splits

The learned-controller line must keep three clean partitions:

- `critic_train_groups`
  benchmark-instance groups used for offline warm start and online adaptation
- `critic_dev_groups`
  benchmark-instance groups used for checkpoint selection, commit calibration, and adaptation stopping
- `paper_eval_groups`
  untouched benchmark instances used for critic-controlled generation and final paper evidence

Current pilot reality:

- the saved dataset is still small at the group level
- the existing `G2` split can support `train` and `validation`
- until a richer pool exists, `paper_eval_groups` should come from newly launched benchmark instances not used in the saved critic dataset

No online updates are allowed on `critic_dev_groups` or `paper_eval_groups`.

## Stable Controller Interface

The controller output remains:

- input:
  graph state plus feasible candidate slate, including `commit`
- output:
  scalar score for each candidate
- decision:
  choose one next action

This interface stays fixed across:

- `G3` text-only critic
- `G4` graph critic
- `G4.5` batched online adaptation

The adaptive method is therefore not a different controller family. It is the same scorer interface with stronger supervision and online refinement.

## File Structure

### Existing Files To Extend

- Modify: `src/idea_graph/trajectory_dataset.py`
  - terminal-state export
  - hindsight-label export hooks
- Modify: `src/idea_graph/critic_dataset.py`
  - reusable label packaging
  - group partition manifest helpers
- Modify: `src/idea_graph/candidate_slate_dataset.py`
  - commit-positive state rows
  - hindsight commit labels when available
- Modify: `src/idea_graph/text_critic.py`
  - warm-start training API
  - weighted updates
  - replay-compatible metadata
- Modify: `scripts/export_graph_critic_dataset.py`
- Modify: `scripts/build_graph_critic_dataset.py`
- Modify: `scripts/build_graph_critic_candidate_dataset.py`

### New Files To Add

- Create: `src/idea_graph/critic_partitions.py`
  - build stable `train / dev / paper_eval` group manifests
- Create: `src/idea_graph/critic_replay.py`
  - replay-buffer schema and batch mixing utilities
- Create: `src/idea_graph/critic_policy.py`
  - safe critic-vs-heuristic decision logic
- Create: `src/idea_graph/online_text_critic.py`
  - batched online update wrapper over the lightweight scorer head
- Create: `scripts/build_critic_partition_manifest.py`
- Create: `scripts/train_text_critic_warmstart.py`
- Create: `scripts/run_online_text_critic_adaptation.py`
- Create: `tests/test_critic_partitions.py`
- Create: `tests/test_critic_replay.py`
- Create: `tests/test_critic_policy.py`
- Create: `tests/test_online_text_critic.py`

### Active Docs To Update After Implementation

- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/paper_experiment_plan.md`
- Modify: `docs/experiment_execution_log.md`

---

### Task 0: Sync The Current Graph-Critic Worktree Into The Canonical Branch

**Files:**
- Modify: main-repo copies of the already reviewed graph-critic files
- Verify: no divergence remains between `main` and `feature/g2-critic-dataset` for active graph-critic code/docs

- [ ] **Step 1: Review the current worktree-only graph-critic diff**

Check:
- `src/idea_graph/trajectory_dataset.py`
- `src/idea_graph/critic_dataset.py`
- `src/idea_graph/candidate_slate_dataset.py`
- `src/idea_graph/text_critic.py`
- related tests and docs

Expected:
- worktree contains the latest reviewed implementation
- `main` is still missing part of this stack

- [ ] **Step 2: Merge or cherry-pick the reviewed worktree commits back to the canonical branch**

Goal:
- make `main` the canonical source of truth again before adding new online-adaptation code

Expected outcome:
- no active graph-critic file exists only in the worktree

- [ ] **Step 3: Run the focused critic packet on the canonical branch**

Run:
`python -m pytest tests/test_engine.py tests/test_candidate_slate_dataset.py tests/test_text_critic.py tests/test_critic_dataset.py tests/test_trajectory_dataset.py -q`

Expected:
- pass on the canonical branch before any new online-adaptation work starts

---

### Task 1: Finalize The Enriched Offline Dataset Contract (`G3.5`)

**Files:**
- Modify: `src/idea_graph/trajectory_dataset.py`
- Modify: `src/idea_graph/critic_dataset.py`
- Modify: `src/idea_graph/candidate_slate_dataset.py`
- Modify: `tests/test_trajectory_dataset.py`
- Modify: `tests/test_candidate_slate_dataset.py`
- Create: `tests/test_critic_partitions.py`
- Create: `src/idea_graph/critic_partitions.py`
- Create: `scripts/build_critic_partition_manifest.py`

- [ ] **Step 1: Extend the dataset stack to include explicit label provenance**

Add row-level fields:
- `state_kind`
- `label_namespace`
- `commit_supervision`
- `label_provenance`
- `partition_role`

The allowed `label_namespace` values for the first pass:
- `teacher_logged`
- `terminal_commit`

The allowed `partition_role` values:
- `critic_train`
- `critic_dev`
- `paper_eval`

- [ ] **Step 2: Add failing tests for partition construction**

Test requirements:
- partitions are assigned by benchmark instance group, never by state row
- no overlap across `critic_train`, `critic_dev`, and `paper_eval`
- small-data mode is supported:
  - if group count is insufficient for a true `paper_eval` split, the script must emit only `critic_train` and `critic_dev` and document that `paper_eval` must come from new benchmark runs

- [ ] **Step 3: Implement `critic_partitions.py`**

Responsibilities:
- build partition rows from group manifests
- preserve benchmark balance when possible
- emit deterministic assignments
- expose a simple helper:
  - `build_partition_manifest(split_rows, *, holdout_groups=None) -> list[dict[str, object]]`

- [ ] **Step 4: Add a CLI builder for the partition manifest**

Run:
`python scripts/build_critic_partition_manifest.py --g2-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g2_commit_enriched --output-dir outputs/graph_critic_datasets --dataset-name current_benchmarked_ours_eig_full_g3p5_partitions`

Expected outputs:
- `partition_manifest.jsonl`
- `partition_stats.json`
- `README.md`

- [ ] **Step 5: Rebuild the enriched dataset with the current saved run pool and explain coverage**

Required outputs:
- `current_benchmarked_ours_eig_full_g1_commit_enriched`
- `current_benchmarked_ours_eig_full_g2_commit_enriched`
- `current_benchmarked_ours_eig_full_g25_commit_enriched`

Required checks:
- record the recovered run count
- explain any gap relative to the earlier frozen `60`-run dataset
- if coverage is still incomplete, mark dataset recovery as partial and queue additional train-group generation

- [ ] **Step 6: Verify the dataset packet**

Run:
`python -m pytest tests/test_trajectory_dataset.py tests/test_critic_dataset.py tests/test_candidate_slate_dataset.py tests/test_critic_partitions.py -q`

Expected:
- all dataset-enrichment tests pass

---

### Task 2: Implement The Offline Warm-Start Trainer

**Files:**
- Create: `src/idea_graph/online_text_critic.py`
- Modify: `src/idea_graph/text_critic.py`
- Create: `scripts/train_text_critic_warmstart.py`
- Create: `tests/test_online_text_critic.py`

- [ ] **Step 1: Define the warm-start supervision contract**

Inputs:
- candidate slate rows from `G2.5`
- partition manifest from `G3.5`

Primary objectives:
- `teacher_logged` ranking loss within each candidate slate
- `terminal_commit` binary loss on the `commit` candidate

Optional auxiliary objective:
- weak regression or weighting by `trajectory_return`

Do not require `hindsight_commit` yet for the first warm-start trainer.

- [ ] **Step 2: Add failing tests for weighted warm-start training**

Test requirements:
- training accepts:
  - offline candidate rows
  - partition manifest
  - namespace filters
- model metadata records:
  - train/dev group counts
  - commit positive counts
  - namespace counts
- validation scoring fails loudly when there are no positive labels for a reported namespace

- [ ] **Step 3: Implement the lightweight warm-start trainer**

Recommended design:
- keep the current text feature path
- keep the encoder frozen
- update only the lightweight scorer head
- expose:
  - `train_warmstart_text_critic(...)`
  - `evaluate_warmstart_text_critic(...)`

Required outputs:
- `model.pkl`
- `metrics.json`
- `metadata.json`
- `training_config.json`

- [ ] **Step 4: Train the first warm-start pilot on the enriched dataset**

Run:
`python scripts/train_text_critic_warmstart.py --candidate-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g25_commit_enriched --partition-manifest outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g3p5_partitions/partition_manifest.jsonl --output-dir outputs/graph_critic_models/current_benchmarked_ours_eig_full_g4_text_warmstart`

Required report fields:
- top-1 action accuracy
- MRR
- train/dev commit positive counts
- namespace counts

- [ ] **Step 5: Verify the warm-start trainer**

Run:
`python -m pytest tests/test_text_critic.py tests/test_online_text_critic.py -q`

Expected:
- warm-start tests pass

---

### Task 3: Implement The Replay Buffer And Batched Online Update Loop

**Files:**
- Create: `src/idea_graph/critic_replay.py`
- Create: `src/idea_graph/critic_policy.py`
- Modify: `src/idea_graph/online_text_critic.py`
- Create: `scripts/run_online_text_critic_adaptation.py`
- Create: `tests/test_critic_replay.py`
- Create: `tests/test_critic_policy.py`

- [ ] **Step 1: Define the online episode contract**

One online adaptation episode should log:
- benchmark instance group
- partition role
- run directory
- per-state candidate slate
- critic scores
- heuristic scores or chosen heuristic fallback
- chosen action
- whether heuristic override happened
- whether commit was allowed
- final local/native return
- sampled hindsight labels when available

Store these under an adaptation directory, not mixed into the older offline dataset directories.

- [ ] **Step 2: Define the safe critic policy**

Required policy rules:
- no online updates on `critic_dev` or `paper_eval`
- no `commit` before the minimum round gate
- `commit` only if:
  - commit score margin exceeds `tau_commit`
  - calibrated commit confidence exceeds `gamma_commit`
- critic only overrides heuristic edit if:
  - best critic edit score exceeds heuristic score by `tau_override`
- otherwise fall back to heuristic policy

This keeps the learned controller conservative and reviewer-friendly.

- [ ] **Step 3: Add failing tests for the replay buffer and safe policy**

Replay tests:
- online rows append correctly
- offline and online buffers can be mixed with a fixed ratio
- no cross-partition contamination

Policy tests:
- commit blocked before minimum round
- heuristic fallback triggers when margins are small
- critic override triggers only when margins exceed thresholds

- [ ] **Step 4: Implement batched online updates**

Recommended schedule:
- collect `K` episodes
- update the scorer head once per batch
- mix examples:
  - `offline_buffer_fraction = 0.7`
  - `online_buffer_fraction = 0.3`
- run for small `E` epochs only
- early stop using `critic_dev_groups`

This is online adaptation, but episodic and stable rather than fully continual in-run training.

- [ ] **Step 5: Implement the adaptation runner**

Run shape:
`python scripts/run_online_text_critic_adaptation.py --llm-config configs/openai_compatible.example.json --candidate-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g25_commit_enriched --partition-manifest outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g3p5_partitions/partition_manifest.jsonl --warmstart-model outputs/graph_critic_models/current_benchmarked_ours_eig_full_g4_text_warmstart/model.pkl --output-dir outputs/graph_critic_models/current_benchmarked_ours_eig_full_g45_text_online --episodes-per-batch 4 --num-batches 3`

Required artifacts:
- `episode_manifest.jsonl`
- `online_buffer.jsonl`
- `checkpoint_metrics.jsonl`
- `best_model.pkl`
- `best_metadata.json`

- [ ] **Step 6: Verify the online adaptation layer**

Run:
`python -m pytest tests/test_critic_replay.py tests/test_critic_policy.py tests/test_online_text_critic.py -q`

Expected:
- all replay and policy tests pass

---

### Task 4: Add Hindsight Commit Labels Without Manual Annotation

**Files:**
- Modify: `src/idea_graph/trajectory_dataset.py`
- Modify: `src/idea_graph/candidate_slate_dataset.py`
- Modify: `src/idea_graph/online_text_critic.py`
- Create: `tests/test_hindsight_commit_labels.py`

- [ ] **Step 1: Define the hindsight labeling rule**

For a sampled intermediate state:
- synthesize immediately from the current graph state
- evaluate the immediate proposal
- compare against the continued trajectory final proposal

Assign:
- `hindsight_commit = 1`
  if immediate synthesis is at least as good as continued evolution within a tolerance margin
- `hindsight_commit = 0`
  otherwise

Tolerance margin must be documented and fixed before large runs.

- [ ] **Step 2: Limit hindsight cost**

To keep this scalable:
- do not label every intermediate state
- sample at most:
  - one early state
  - one mid state
  - one near-terminal state
  per episode

This is the main answer to reviewer concerns about annotation burden.

- [ ] **Step 3: Add tests for hindsight namespace packaging**

Test requirements:
- hindsight labels are stored in a separate namespace
- missing hindsight labels do not break older rows
- commit-positive counts by namespace are reported correctly

- [ ] **Step 4: Integrate hindsight labels as an optional online loss**

Do not make hindsight mandatory for the first online pilot.
Instead:
- enable it by config
- log its sample count
- report whether it improves commit calibration

---

### Task 5: Freeze And Evaluate The Adapted Critic Cleanly

**Files:**
- Modify: `src/idea_graph/critic_policy.py`
- Create: `scripts/run_critic_controlled_packet.py`
- Create: `tests/test_critic_controlled_packet.py`
- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/paper_experiment_plan.md`
- Modify: `docs/experiment_execution_log.md`

- [ ] **Step 1: Define the freeze point**

The adapted critic becomes paper-evaluable only after:
- batched online adaptation finishes on `critic_train_groups`
- best checkpoint is selected on `critic_dev_groups`
- commit calibration is fixed on `critic_dev_groups`

After that, no further updates are allowed.

- [ ] **Step 2: Run the first frozen 4-case gate**

Use untouched benchmark cases for the first critic-controlled packet:
- `AI_Idea_Bench_2025`
  a 4-case smoke gate
- optional later:
  a small `LiveIdeaBench` frozen gate

This packet should compare:
- `ours-eig-heuristic`
- `ours-eig-critic-text-offline`
- `ours-eig-critic-text-online`

- [ ] **Step 3: Report the minimum reviewer-facing metrics**

For the frozen gate report:
- benchmark-native automatic metrics
- local development metrics as supplementary diagnostics
- commit rate
- premature-commit and late-commit rates when hindsight labels exist
- critic-vs-heuristic action agreement
- adaptation cost:
  - number of extra episodes
  - tokens
  - model-update time

- [ ] **Step 4: Verify the frozen evaluation packet**

Run:
`python -m pytest tests/test_critic_controlled_packet.py -q`

Expected:
- critic-controlled evaluation uses frozen checkpoints only

---

## Decision Gates

### Gate A: Is The Dataset Strong Enough?

Proceed only if:
- enriched `G2.5` has non-zero positive `commit` labels in both train and dev
- partition manifest is leakage-safe
- the recovered run coverage is explained

If not:
- generate additional `ours-eig` runs on `critic_train_groups` only before online adaptation

### Gate B: Is The Warm-Start Critic Better Than Random?

Proceed only if the warm-start model:
- beats a random or trivial ranking baseline
- has non-zero commit-positive support
- does not collapse on dev groups

If not:
- improve data coverage before adding online adaptation

### Gate C: Does Online Adaptation Help?

Proceed to larger controlled generation only if:
- `offline + online` beats `offline only` on held-out dev metrics
- commit calibration does not worsen sharply
- adaptation cost is moderate relative to generation cost

If not:
- keep `offline-only` as the stronger claim

## Reviewer-Facing Positioning

This plan should be described as:

- a `lightweight adaptive controller`
- warm-started from historical trajectories
- improved with `batched online updates` from outcome-derived weak supervision
- frozen before held-out evaluation

This plan should **not** be described as:

- full reinforcement learning
- end-to-end LLM fine-tuning
- continual test-time adaptation

That wording is important for both correctness and reviewer trust.

## Self-Review

### Spec coverage

- offline warm start:
  covered by Task 2
- online lightweight updates:
  covered by Task 3
- no train/dev/test leakage:
  covered by canonical partitions and Task 5 freeze rule
- scalable and low-burden supervision:
  covered by Tasks 1 and 4
- reviewer-facing stability story:
  covered by the safe policy and decision gates

### Placeholder scan

- no `TODO`
- no `TBD`
- no unresolved split ownership
- commands, artifacts, and file responsibilities are explicit

### Type consistency

- dataset stack:
  `G1 -> G2 -> G2.5 -> G3.5 -> G4 warmstart -> G4.5 online -> G5 frozen gate`
- partitions:
  `critic_train`, `critic_dev`, `paper_eval`
- supervision namespaces:
  `teacher_logged`, `terminal_commit`, `hindsight_commit`, `trajectory_return`

