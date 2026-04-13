# Critic-Train And Paper-Eval Split Design

## Goal

Make the learned-controller experiments reviewer-safe by separating:

1. `critic_train`
   used for offline warm start and online adaptation
2. `critic_dev`
   used for model selection and conservative threshold checks
3. `paper_eval`
   used only for frozen final benchmark results

The key point is simple:

> The current graph-critic dataset is a **development pool**, not the final
> paper benchmark pool.

## Current State

The current partition artifact is:

- `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions`

Current counts:

- total groups: `11`
- `critic_train`: `9`
- `critic_dev`: `2`
- `paper_eval`: `0`

Benchmark breakdown:

- `AI_Idea_Bench_2025`
  - `critic_train`: `6`
  - `critic_dev`: `1`
- `liveideabench`
  - `critic_train`: `3`
  - `critic_dev`: `1`

Current concrete assignment:

- `critic_train`
  - `AI_Idea_Bench_2025`
    - `13`
    - `15`
    - `18`
    - `21`
    - `3883`
    - `7909`
  - `LiveIdeaBench`
    - `earthquakes-70`
    - `meteorology-0`
    - `periodic table-23`
- `critic_dev`
  - `AI_Idea_Bench_2025`
    - `9849`
  - `LiveIdeaBench`
    - `weather forecasting-47`

This is acceptable for development, but it is **not** enough for a clean final
paper story, because `paper_eval` is still empty.

## Design Options

### Option A: Strict Three-Pool Protocol

Use the current `11` groups only for critic development, and create a new,
untouched `paper_eval` benchmark pool later.

Pros:

- strongest reviewer story
- no ambiguity about leakage
- clean `offline` / `online` / `final benchmark` separation

Cons:

- requires more generation cost
- final benchmark pool must be launched deliberately

### Option B: Reuse The Current Pool With Rotating Splits

Keep re-partitioning the current `11` groups and report the best split.

Pros:

- cheapest in API cost
- fastest to iterate

Cons:

- weak paper story
- easy for reviewers to question leakage and overfitting
- hard to present as a frozen final benchmark

### Option C: Train On AIIB Only, Evaluate On LiveIdeaBench

Use `AI_Idea_Bench_2025` as the critic-training pool and keep
`LiveIdeaBench` mostly untouched as transfer evaluation.

Pros:

- simple cross-benchmark story
- easier to explain

Cons:

- wastes useful `LiveIdeaBench` development data
- weakens the claim that the controller itself is benchmark-robust
- still does not solve the need for a proper final untouched AIIB pool

## Recommendation

Use **Option A**.

That means:

- treat the current `11` groups as `development only`
- keep the current `9 / 2` split as the active `critic_train / critic_dev`
  split
- do **not** report those groups as the final paper benchmark for the learned
  controller
- create a separate future `paper_eval` pool from new benchmark instances that
  are never used in critic training or adaptation

This is the cleanest and most persuasive setup.

## Recommended Protocol

### 1. Freeze The Current Development Pool

The current full graph-critic dataset stack should be treated as:

- `development_pool_v1`

It supports:

- offline warm start
- online adaptation debugging
- threshold tuning
- replay-policy iteration

It does **not** support:

- final frozen benchmark claims for the learned controller

### 2. Keep The Active Critic Split Narrow And Stable

For the next stage:

- keep current `critic_train = 9 groups`
- keep current `critic_dev = 2 groups`
- do not reshuffle them again unless there is a hard failure

Reason:

- the current split already has both benchmarks represented
- frequent repartitioning would make progress hard to interpret
- stability matters more than squeezing a slightly better dev score

### 3. Select Future `critic_train` Expansion Carefully

If we need more training episodes, add them only to a new
`critic_train_expansion` pool.

Selection rules:

- choose benchmark instances that are not intended for the final paper table
- prefer benchmark cases with:
  - stable saved artifacts
  - clean native evaluation availability
  - multi-round trajectories
  - nontrivial `commit` supervision
  - diverse difficulty and topic coverage
- prioritize `AI_Idea_Bench_2025` first, because it is the primary paper
  benchmark
- add `LiveIdeaBench` expansion only after AIIB expansion is stable

Practical rule:

- new training cases should be marked explicitly as `critic_train_only`
- once assigned there, they should never appear in the frozen final benchmark
  packet

### 4. Create A Truly Untouched `paper_eval` Pool

The learned-controller paper results should eventually come from a separate
pool:

- `paper_eval_v1`

Requirements:

- zero overlap with `critic_train`
- zero overlap with `critic_dev`
- no warm-start fitting
- no online adaptation
- no threshold tuning

This pool should be used only after the model is frozen.

### 5. Organize Artifacts By Purpose

Training and evaluation outputs should be separated physically, not only
conceptually.

Recommended organization:

- training datasets:
  - `outputs/graph_critic_datasets/...`
- training / adaptation models:
  - `outputs/graph_critic_models/...`
- frozen final benchmark packets:
  - `outputs/quality_batches/paper_eval_*`
  - or another clearly named benchmark-results root

Recommended naming convention:

- `development_pool_*`
- `critic_train_*`
- `critic_dev_*`
- `paper_eval_*`

Avoid ambiguous names like:

- `current_full`
- `final`
- `main`

unless the role is encoded explicitly.

### 6. Add A Human-Readable Split Registry

Before the next large run, we should keep one canonical registry file that
lists every benchmark instance and its role:

- benchmark
- instance name
- pool name
- partition role
- allowed usages

This registry should answer, at a glance:

- can this instance be used for training?
- can this instance be used for dev tuning?
- can this instance be reported in the final paper table?

## Immediate Next Step

The next implementation step should be:

1. freeze the current `11` groups as `development_pool_v1`
2. add a canonical split-registry artifact for those groups
3. define the first new untouched `paper_eval` candidate list
4. only then launch new train-group episode collection

## Why This Is The Right Story

This design keeps the paper claim honest:

- the current graph-critic work is still in the development stage
- the online controller can be improved on a stable development pool
- the final learned-controller evidence will later come from untouched benchmark
  instances

That separation is exactly what a careful NeurIPS reviewer will want to see.
