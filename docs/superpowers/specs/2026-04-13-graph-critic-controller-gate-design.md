# Graph Critic Controller Gate Design

**Date:** 2026-04-13  
**Scope:** first controller-in-the-loop integration of the leakage-safe relation-aware graph critic

## Purpose

This note defines the next learned-controller step after the offline
relation-aware graph critic cleared the frozen development gate.

The immediate goal is not to scale new benchmark batches. The immediate goal is
to test whether the stronger offline graph critic transfers into better
end-to-end `EIG` action selection on the same frozen 4-case AIIB controller
packet used for the text-critic pilot.

## Current Validated State

### Stable Main Method

- `ours-eig` remains the stable benchmarked main system.

### Mixed Learned Text-Controller Pilot

- `ours-eig-critic-text` is already integrated into runtime reranking.
- It improved trace quality and removed the worst early-stop pathology, but the
  end-to-end AIIB native result remained mixed.

### Strong Offline Graph-Critic Evidence

- refreshed text critic:
  - top-1 `0.7081`
  - MRR `0.8147`
- first graph-feature baseline:
  - top-1 `0.5024`
  - MRR `0.6824`
- leakage-safe relation-aware graph critic:
  - top-1 `0.8373`
  - MRR `0.8951`

So the graph critic is now strong enough to justify one narrow runtime gate.

## Core Decision

Use the current trained graph critic first, without collecting more data up
front.

If the first controller gate is unsatisfying, then move to targeted robustness
data expansion.

This is the fastest path that still keeps the learned-controller story honest.

## Scope

### In Scope

- add a runtime adapter for the relation-aware graph critic
- add a new baseline:
  - `ours-eig-critic-graph`
- use the graph critic for edit-action reranking only
- keep the current safety-biased maturity guard
- run the frozen 4-case AIIB gate:
  - `13`
  - `3883`
  - `7909`
  - `9849`
- compare against `ours-eig`
- log enough runtime trace information to diagnose success or failure

### Out Of Scope

- learned `commit`
- changing the offline training objective
- large benchmark packets before the small gate is revalidated
- collecting new robustness data before the first graph-controller transfer test

## Design Options

### Option A: Retrain First With More Data

Pros:

- stronger robustness before deployment

Cons:

- slower
- may waste time if the current graph critic already transfers well enough

### Option B: Direct Runtime Integration With The Current Artifact

Pros:

- fastest
- gives immediate end-to-end signal

Cons:

- requires careful runtime safety because the current artifact was trained
  offline, not as a live controller

### Option C: Recommended

Use the current artifact for one narrow runtime gate, but keep the deployment
conservative:

- edit reranking only
- learned `commit` disabled
- heuristic fallback on runtime mismatch
- frozen 4-case packet only

If the result is not satisfying, then expand the robustness dataset in a
targeted way.

## Recommended Runtime Architecture

### 1. Runtime Bundle Loader

The runtime graph critic should load from the existing model output directory:

- `model.pt`
- `metadata.json`
- `training_config.json`

The current artifact does not save explicit vocab maps. To avoid a full retrain
before the first controller gate, the runtime loader should rebuild the
necessary vocabularies deterministically from the frozen dataset roots recorded
in `training_config.json`:

- `candidate_dataset_dir`
- `g1_dataset_dir`
- `partition_manifest`

This keeps `development_pool_v2_relation_graph_sanitized_v1` usable for the
first gate.

### 2. Live Graph-To-Batch Conversion

At each runtime decision point:

- take the current active graph
- take the valid candidate edit specs from the engine
- reuse the same ontology as offline training:
  - node text
  - node type
  - node role
  - edge relation
  - candidate kind
  - target-node indices
- sanitize candidate text before embedding
- use the same sentence-transformer backend recorded in the training artifact

The runtime builder only needs one state-local batch per decision.

### 3. Safe Controller Policy

The graph critic should plug into the same safe reranking policy family already
used by the text critic:

- no learned `commit`
- same override-threshold logic
- same maturity-sensitive override guard
- same heuristic fallback semantics

This keeps the first comparison focused on ranking quality rather than on a
simultaneous stopping-policy redesign.

### 4. Runtime Safety Fallback

The graph critic must fall back to heuristic selection when:

- the model bundle cannot be loaded
- a required runtime token cannot be mapped safely into the trained ontology
- runtime scoring raises an exception
- the candidate batch is empty after commit filtering

Every fallback must be logged explicitly.

### 5. Trace Logging

The runtime trace must record:

- controller kind
- heuristic candidate id / kind / predicted gain
- critic-selected candidate id / kind / score
- top scored candidates
- override margin
- whether heuristic fallback happened
- fallback reason if any
- maturity snapshot used by the policy

This is required because the first graph-controller gate is a transfer test, so
diagnosis quality matters as much as the final mean score.

## Frozen Gate Protocol

### Benchmarks And Cases

Use only the frozen AIIB controller packet:

- `13`
- `3883`
- `7909`
- `9849`

### Compare

- `ours-eig`
- `ours-eig-critic-graph`

### Shared Runtime Settings

- same agent backend
- same LLM config
- same `max_rounds`
- same native evaluation setting
- same output-root layout

## Decision Rule

The first graph-controller gate is considered satisfying if all of the
following hold:

1. mean AIIB native score is at least neutral versus `ours-eig`
2. no new obvious maturity or early-stop pathology appears
3. no catastrophic single-case drop appears
4. traces show non-trivial critic usage rather than almost universal fallback

If those conditions hold, the graph critic is ready for a larger controller
packet.

If those conditions do not hold, do not scale generation yet.

## Failure Interpretation

If the first graph-controller gate underperforms, the first assumption should
not be that graph structure is useless.

The likely failure modes are:

- runtime transfer mismatch between offline and live candidate distributions
- insufficient robustness around late-round or near-maturity states
- poor calibration on disagreement states where heuristic and critic differ
- unmapped or weakly represented runtime action patterns

That is why the first fallback step should be targeted robustness data
expansion, not immediate abandonment of the graph critic.

## Conditional Robustness Expansion

If the gate is unsatisfying, the next data expansion should focus on:

- AIIB hard cases
- near-maturity late-round states
- heuristic-versus-critic disagreement states
- states with large native-score sensitivity

The expansion should remain development-only and should rebuild the same G1 /
G2 / G2.5 pipeline before retraining the graph critic.

## Files Likely To Change

- `src/idea_graph/relation_graph_critic_data.py`
- `src/idea_graph/relation_graph_runtime_critic.py`
- `src/idea_graph/runtime_critic.py`
- `src/idea_graph/baselines.py`
- `src/idea_graph/engine.py`
- `tests/test_relation_graph_runtime_critic.py`
- `tests/test_benchmark_mode_and_baselines.py`
- `tests/test_engine.py`
- `docs/experiment_execution_log.md`
- `docs/eig_graph_critic_plan.md`

## Expected Outcome

Best case:

- `ours-eig-critic-graph` beats or matches `ours-eig` on the frozen 4-case
  gate
- the graph critic becomes the new learned-controller candidate for larger
  runtime evaluation

Acceptable fallback:

- the gate is mixed
- traces expose exactly which states need more robustness data
- targeted dataset expansion becomes the next justified step
