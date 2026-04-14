# Graph Critic Scale-Up And Main Evaluation Design

**Date:** 2026-04-14  
**Scope:** final development-only graph-critic expansion, controller diagnosis,
freeze criteria, and main benchmark launch plan

## Purpose

This spec turns the current graph-critic work into a concrete paper-facing
execution design.

The paper direction is now explicit:

- `ours-eig-graph-critic` is the intended main method
- `ours-eig` remains the nearest ablation and stable fallback
- online calibration is optional and must not block the main method track

The immediate problem is not whether the graph critic exists. It does.

The immediate problem is whether the current development-only training pool is
already strong enough to support a convincing learned-controller claim in the
paper. The answer is **not yet**.

The current development artifacts are already strong enough for:

- offline graph-critic training
- small controller diagnosis packets
- runtime debugging
- mechanism analysis

They are not yet strong enough for:

- a paper-ready learned-controller claim
- robust `commit` behavior
- a strong graph-versus-text superiority claim

Therefore, the next step is a **bounded final development-only data expansion**
before larger controller diagnosis and before any frozen paper-eval benchmark
packet.

## Current Grounded Status

### Stable Facts

Current active graph-critic training root:

- `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25`

Current dataset scale:

- groups: `23`
- runs: `72`
- states: `1267`
- candidate rows: `13004`
- explicit commit candidates: `1267`
- terminal commit-positive states: `72`
- validation groups: `6`

Current trusted offline learned-controller result:

- relation-aware graph critic beats the refreshed text scorer on the frozen
  development split

Current controller-in-the-loop result:

- 4-case sentinel packet is only **neutral overall**
- it is good enough to continue diagnosis
- it is not yet good enough for large-scale benchmark spending

### Key Interpretation

The current bottleneck is not raw row count. The main bottleneck is still
**too few leakage-safe benchmark-instance groups**, especially for learning a
stable controller policy and a credible `commit` decision.

That means the next expansion should prioritize:

- more distinct benchmark groups
- more varied hard cases
- more late-round states
- more explicit `commit`-relevant supervision

It should not prioritize:

- duplicating many more runs on the same few benchmark instances
- tuning on future paper-eval instances

## Design Principles

### 1. Keep Development And Paper Evaluation Strictly Separated

All new Stage A data remains development-only.

No instance used in:

- `development_pool_v1`
- `development_pool_v2_candidate_pool_v1`
- the new Stage A expansion pool

may later be used in the final frozen learned-controller paper-eval pool.

### 2. Graph Critic Is The Main Learned Method

The paper should eventually center:

- `ours-eig-graph-critic`

with:

- `ours-eig`

as the nearest mechanism ablation and stability reference.

### 3. Expansion Must Strengthen `commit`, Not Just Edit Ranking

The controller story is incomplete if the graph critic only learns edit
ranking while `commit` remains weak or absent.

Stage A must therefore enrich `commit` supervision without violating benchmark
fairness.

### 4. No True Online Calibration In The Critical Path

The main method should not depend on an adaptive benchmark-time calibration
loop.

Allowed later:

- frozen development-only commit-margin calibration

Not allowed in the main claim:

- per-benchmark-instance online retuning during final evaluation

### 5. Larger Final Tables Are Required

Small packets are useful for:

- diagnosis
- model selection
- failure analysis

They are not sufficient for the final main table.

## Approaches Considered

### Option A: Use Current Data And Scale Immediately

Pros:

- fastest
- no more development collection cost

Cons:

- weak paper story for the graph critic
- high risk that `commit` remains undertrained
- fragile if the 4-case neutral result was just variance

### Option B: Recommended Bounded Final Expansion Then Freeze

Pros:

- improves group diversity
- strengthens `commit`
- still keeps the paper moving quickly
- gives a much cleaner story before large benchmark spend

Cons:

- one more development collection round

### Option C: Large Expansion Before Any More Diagnosis

Pros:

- strongest dataset immediately

Cons:

- slower than necessary
- delays controller diagnosis on the already-implemented runtime path

## Chosen Approach

Use **Option B**.

The next work proceeds in five stages:

1. Stage A: final development-only critic expansion
2. Stage B: offline critic refresh and freeze gate
3. Stage C: medium controller diagnosis packet
4. Stage D: final controller freeze
5. Stage E: main benchmark experiments and ablations

## Stage A: Final Development-Only Critic Expansion

### Objective

Build one stronger final development-only training pool for the graph critic
before main benchmark experiments.

### Target Scale

Add approximately:

- `24` new `AI_Idea_Bench_2025` groups
- `12` new `LiveIdeaBench` groups

Target post-expansion totals:

- total groups: about `55` to `60`
- validation groups: at least `12`
- total runs: about `120` to `160`
- terminal commit-positive states: at least `140`

The exact final count may vary slightly due to run failures, but the expansion
must materially exceed the current `23`-group / `72`-run scale.

### Sampling Policy

The new expansion pool should prioritize:

- hard AIIB cases where controller selection matters
- cases with more ambiguous multi-claim idea development
- cases likely to produce longer or more conflict-heavy graphs
- LiveIdeaBench rows that stress weak-context behavior

The new expansion pool should avoid:

- near-duplicates of already overrepresented development groups
- trivial cases that mature immediately
- future paper-eval candidate instances

### Split Policy

Create a new explicit development-only partition layer with:

- `critic_train`
- `critic_dev`

Recommended ratio:

- about `75%` train groups
- about `25%` dev groups

The split unit remains the **benchmark-instance group**, never the state row.

### Run Policy

For each new group:

- collect at least `2` full runs when feasible
- use `3` runs on selected hard groups to increase action diversity
- keep runtime traces, graph snapshots, candidate slates, and evaluation
  outputs

This should give enough variation for controller training without exploding
collection cost.

### Commit Enrichment Policy

The expansion must include explicit `commit`-relevant signals beyond terminal
labels.

Required additions:

- export shadow-commit candidates for later rounds
- record whether `commit` would be preferred relative to the chosen edit
- preserve late-round states even when the final run continues

The purpose is to create richer supervision for:

- when the graph should continue evolving
- when the graph is already good enough to stop

### Required Artifacts

Stage A should produce a new active development dataset root under:

- `outputs/graph_critic_datasets/02_active_graph_critic/`

Expected artifact families:

- candidate pool manifest
- group partition manifest
- split registry or override file
- online episode collections
- combined `G1`
- combined `G2`
- combined `G2.5`
- dataset stats
- readiness report

### Stage A Success Criteria

Stage A is complete only if all of the following hold:

- expansion pool is fully documented and leak-safe
- new total groups reach about `55` or more
- validation groups reach at least `12`
- `commit`-positive supervision is materially larger than the current `72`
- dataset provenance is clear enough that future paper-eval separation remains
  auditable

## Stage B: Offline Critic Refresh And Freeze Gate

### Objective

Train the final development version of:

- refreshed text scorer
- graph critic

on the expanded Stage A dataset, then freeze the stronger graph critic before
any larger runtime packet.

### Required Comparisons

Evaluate on the frozen Stage A `critic_dev` groups:

- text scorer
- relation-aware graph critic
- optional hybrid text-plus-graph scorer if needed for diagnosis

### Promotion Rule

The graph critic may advance only if it is clearly at least as strong as the
text scorer on the frozen development validation groups, with the graph line
preferred as the paper method only if it is directionally better on the main
ranking metrics.

Recommended primary offline metrics:

- top-1 accuracy
- mean reciprocal rank
- edit-only top-1 accuracy
- edit-only mean reciprocal rank
- commit ranking diagnostics

### Stage B Success Criteria

- graph critic remains leakage-safe
- graph critic matches or beats the refreshed text scorer offline
- commit diagnostics are no longer empty or trivial

## Stage C: Medium Controller Diagnosis Packet

### Objective

Test whether the stronger graph critic actually improves end-to-end generation
quality, not only offline ranking.

### Packet Size

Recommended primary diagnosis packet:

- `24` AIIB cases

Optional secondary diagnosis packet:

- `8` to `12` LiveIdeaBench cases

These packets are still development-only and must come from development pools,
not future paper-eval pools.

### Compared Systems

- `ours-eig`
- `ours-eig-graph-critic`

### Required Runtime Analysis

For every run, record and later summarize:

- action selected each round
- action distribution by round and by benchmark
- controller override rate
- materialized override rate
- heuristic fallback rate
- shadow commit fire rate
- actual stop round
- stop reason
- total rounds without commit

### Stage C Success Criteria

- no obvious `3883`-style pathological regression
- mean native score is at least near-neutral and preferably positive
- controller materially affects execution rather than silently falling back
- action traces are interpretable enough for paper analysis

## Stage D: Final Controller Freeze

### Objective

Freeze the learned controller before untouched paper-eval benchmarking.

### Freeze Contents

The freeze must include:

- selected checkpoint
- runtime controller policy
- allowed fallback rules
- whether live `commit` is enabled
- any development-only frozen commit-margin threshold

After this freeze:

- no more tuning on development packets should affect the paper-eval method

## Stage E: Main Benchmark Experiments

### Main Automatic Evaluation

`AI_Idea_Bench_2025`:

- preferred size: `96` cases
- minimum acceptable size: `64` cases

`LiveIdeaBench`:

- preferred size: `64` cases
- minimum acceptable size: `48` cases

### Main Comparison Set

- `direct`
- `self-refine`
- `ai-researcher`
- `ours-eig`
- `ours-eig-graph-critic`

### Method Framing

In the main paper:

- `ours-eig-graph-critic` is the main method
- `ours-eig` is the nearest ablation / heuristic controller baseline

### Main Table Rule

The earlier `4`, `12`, and `24`-case packets are for:

- smoke verification
- diagnosis
- model selection

They must not be presented as the final main table for the learned-controller
claim.

## Ablations

Minimum planned ablations:

- `ours-eig`
- `ours-eig-graph-critic`

Secondary mechanism ablations, only if time permits:

- graph critic without live `commit`
- graph critic with heuristic stopping
- graph critic with reduced controller authority

## Human Evaluation

After the main automatic runs are stable, run blind human evaluation on a
balanced subset.

Recommended scale:

- `12` to `16` AIIB cases
- `12` to `16` LiveIdeaBench cases

## Risks And Mitigations

### Risk 1: More Data But Still Weak `commit`

Mitigation:

- explicitly enrich shadow-commit supervision during Stage A
- keep commit diagnostics visible during Stages B and C

### Risk 2: Offline Gains Do Not Transfer Online

Mitigation:

- require a medium controller diagnosis packet before the main benchmark launch
- analyze action materialization and fallback, not only final scores

### Risk 3: Development Set Becomes Messy

Mitigation:

- keep one canonical active graph-critic dataset root
- archive superseded intermediate roots
- maintain explicit manifests and split registries

### Risk 4: Calibration Becomes A Tuning Rabbit Hole

Mitigation:

- keep true online calibration out of scope
- allow only one frozen development-only commit-margin calibration if later
  needed

## Immediate Next Step

Begin **Stage A**:

- define the new development-only expansion pool
- freeze train/dev group assignments
- collect the new traced runs
- export the expanded `G1`/`G2`/`G2.5` artifacts
- write a new readiness report before any further controller promotion
