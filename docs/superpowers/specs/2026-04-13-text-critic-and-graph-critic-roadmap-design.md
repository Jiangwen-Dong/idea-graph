# Text Critic And Graph Critic Roadmap Design

**Date:** 2026-04-13  
**Scope:** learned-controller roadmap for `EIG` after the current `G4.8` text-critic pilot

## Purpose

This note turns the current learned-controller situation into a concrete
research plan.

The central clarification is:

- the repo currently implements a **text critic**, not a true graph critic
- the text critic is already useful as a controller pilot
- the graph critic remains the next real learned-method stage

The goal of this roadmap is to keep paper progress fast while preventing the
critic line from drifting into an open-ended engineering project.

## Current State

### Stable Main Method

- `ours-eig` is still the stable main method for benchmarked paper progress
- the current learned critic must be treated as an extension / pilot rather
  than the main paper system

### Implemented Learned-Controller Stack

- frozen development pool:
  `development_pool_v1`
- explicit split registry:
  `critic_train`, `critic_dev`, future `paper_eval`
- real `critic_train` episode collection:
  `outputs/graph_critic_online_episodes/development_pool_v1_critic_train_qwen_v1`
- online replay buffer:
  `outputs/graph_critic_online_episodes/development_pool_v1_critic_train_qwen_v1/online_replay_buffer.jsonl`
- adapted text critic:
  `outputs/graph_critic_models/current_benchmarked_ours_eig_full_g46_text_online_real_train_v1`
- first runtime controller pilot:
  `ours-eig-critic-text`

### What The Current Critic Actually Is

The current learned scorer is a **text critic**:

- graph state is flattened into text
- candidate action is flattened into text
- the model scores `state_text [SEP] candidate_text`
- runtime use is conservative reranking only
- learned `commit` is not deployed
- graph structure is not encoded with message passing or graph attention

So the current critic is a valid learned-controller pilot, but not yet a true
graph critic.

## Dataset Sufficiency Judgment

### Current Dataset Facts

From the current aggregated and commit-enriched artifacts:

- runs: `60`
- benchmark-instance groups: `11`
- transitions: `910`
- candidate-slate states: `970`
- candidate rows: `10092`
- train candidate rows: `9067`
- validation candidate rows: `1025`
- terminal commit-positive states: `60`
- current held-out `critic_dev` groups: `2`

### What Is Sufficient Right Now

The current dataset is sufficient for:

- text-critic warm start
- online adaptation pilot
- conservative controller pilot
- offline ranking comparison between simple models
- debugging the full learned-controller pipeline

### What Is Not Yet Sufficient

The current dataset is **not yet sufficient** for:

- a strong paper-ready graph-critic claim
- a convincing graph-versus-text superiority claim
- robust commit calibration
- stable learned-controller generalization claims

### Main Reason

The bottleneck is **not** raw transition count alone. The main bottleneck is
too few leakage-safe benchmark-instance groups.

We currently have only `11` groups total and only `2` validation groups. That
is enough for a pilot, but too small for a convincing graph-critic result.

## Design Principles

### Principle 1: Keep The Paper Moving

`ours-eig` should keep advancing as the stable main method regardless of critic
progress.

### Principle 2: Separate Pilot From Main Claim

The text critic is a low-risk pilot that validates learning signal and runtime
control mechanics. The graph critic is the actual learned-method candidate.

### Principle 3: Offline First For The Graph Critic

A graph critic should first beat the text critic on frozen offline ranking
before it enters the runtime loop.

### Principle 4: Do Not Learn Commit Yet

The first learned graph critic should rank edit actions only. Learned stopping
should be postponed until action selection is stable.

## Approach Options

### Option A: Stop At The Text Critic

Use the current text critic only as a pilot / appendix result.

Pros:

- fastest
- lowest risk
- least engineering burden

Cons:

- weak learned-method novelty
- does not test whether graph structure matters

### Option B: Jump Directly To Runtime Graph Critic

Build a graph critic and put it directly into the controller loop.

Pros:

- strongest ambition

Cons:

- highest risk
- easiest way to slow paper progress
- hard to diagnose if both the graph encoder and runtime policy change at once

### Option C: Recommended Dual-Track Plan

Stabilize the text critic just enough to make it a trustworthy pilot, while
building the graph critic offline first.

Pros:

- keeps paper progress fast
- preserves a clean research story
- makes the graph critic easier to diagnose

Cons:

- requires disciplined milestone gates

## Recommended Plan

Use **Option C**.

This means:

- short, bounded work on the text critic
- stronger emphasis on offline graph-critic development
- no large controller batch until the same frozen 4-case gate is rerun

## Text Critic Plan

### T1: Maturity-Sensitive Safety Patch

**Goal:** prevent the learned reranker from making the unchanged heuristic
maturity logic stop too early.

**Why now:** the first 4-case gate failed mainly because case `3883` matured at
`Round2` instead of `Round4`.

**Scope:**

- keep learned `commit` disabled
- allow learned reranking only when the state is not near a fragile maturity
  boundary, or when the selected edit clearly increases evidence / grounding
- preserve controller traces for all future LLM-backed runs

**Success criterion:**

- no obvious early-stop pathology on the frozen 4-case AIIB packet

### T2: Frozen 4-Case Controller Rerun

**Cases:**

- `13`
- `3883`
- `7909`
- `9849`

**Compare:**

- `ours-eig`
- `ours-eig-critic-text`

**Goal:**

- decide whether the text critic is now neutral / positive or still too fragile

**Decision gate:**

- continue only if mean native score is at least neutral and no case shows a
  new obvious maturity regression
- otherwise freeze the text critic as a pilot result

### T3: Small Robustness Packet

**Run only if T2 is acceptable.**

**Goal:** check whether the text critic remains stable beyond the 4-case packet.

**Recommended size:**

- `8` to `12` development-only benchmark-instance groups

**Decision gate:**

- if results remain mixed, stop text-critic expansion

### T4: Freeze The Text-Critic Story

At the end of T2 or T3, explicitly freeze the framing as one of:

- positive controller pilot
- mixed but informative pilot
- negative pilot with diagnostic value

After this freeze, the text critic should stop expanding unless it becomes
necessary for the graph-critic comparison.

## Graph Critic Plan

### G1: Freeze The First Graph-Critic Task

The first true graph critic should solve:

- input:
  - graph state `G_t`
  - benchmark-visible input `x`
  - candidate action `a`
- output:
  - score `Q(G_t, a, x)`

**First task:** edit-action ranking only.

**Not in scope yet:**

- learned commit deployment
- direct proposal generation

### G2: Build Graph-Structured Features

Create a graph-facing dataset layer on top of the existing candidate-slate
artifacts.

Minimum graph features:

- node text embedding
- node type embedding
- role embedding
- evidence / support indicators
- edge relation embedding
- resolved / unresolved edge flag
- candidate action kind embedding
- target-node index features

**Output:** a leakage-safe offline dataset for graph-critic training over the
same `critic_train` / `critic_dev` story.

### G3: Implement Graph Critic V1

Use a lightweight graph model first.

**Recommended architecture:**

- relational GNN / R-GCN first
- small scorer head over graph state plus candidate-action representation

**Why not a heavy graph transformer first:**

- current data scale is still modest
- the first goal is a clean comparison against the text critic, not a maximal
  architecture search

### G4: Offline Graph-Versus-Text Comparison

Compare on frozen `critic_dev`:

- text critic
- graph critic v1

Primary metrics:

- top-1 next-action accuracy
- mean reciprocal rank

Secondary diagnostics:

- performance by benchmark
- performance by phase
- performance on maturity-sensitive states

**Decision gate:**

- only move the graph critic into the runtime loop if it beats the text critic
  offline

### G5: Runtime Graph-Critic Pilot

Only after G4 is positive:

- deploy graph critic as conservative edit reranker only
- keep heuristic stopping unchanged
- rerun the same frozen 4-case AIIB controller gate

This isolates whether graph structure helps end-to-end.

### G6: Commit Head Later

Do not implement learned `commit` next.

Learned stopping should come only after:

- graph critic beats text critic offline
- graph critic is at least neutral in the same frozen runtime gate

## Data Curation Plan

### C1: Do We Need More Data Right Now?

**For the text critic:** not urgently. The current dataset is enough for the
short text-critic stabilization plan.

**For the graph critic:** yes, more data is desirable if we want a serious
graph-critic result instead of only a pilot.

### C2: What Kind Of More Data?

Do **not** prioritize more reruns of already dominant groups such as:

- `ai-idea-bench-2025-13`
- `liveideabench-meteorology-0`

Instead prioritize:

- more **unique benchmark-instance groups**
- especially more leakage-safe AIIB groups
- then more LiveIdeaBench groups for benchmark diversity

### C3: Minimum Expansion Target

Before claiming a serious graph-critic result, aim for:

- at least `20+` unique benchmark-instance groups total
- at least `4` to `6` `critic_dev` groups
- more balanced benchmark coverage across AIIB and LiveIdeaBench

### C4: Suggested Expansion Order

1. expand unique AIIB development-only groups
2. expand unique LiveIdeaBench development-only groups
3. regenerate commit-enriched candidate-slate datasets
4. rerun offline text-versus-graph critic comparison

## Stop / Go Rules

### Text Critic

- **Go** if the frozen 4-case rerun is neutral or better without maturity
  regressions
- **Stop** if it remains mixed and fragile after the safety patch

### Graph Critic

- **Go to runtime** only if graph critic beats text critic offline on frozen
  `critic_dev`
- **Do not deploy** if the graph critic is only tied or worse offline

### Data Expansion

- **Expand now** if graph-critic offline work starts immediately
- **Do not block on expansion** for the short text-critic stabilization pass

## Near-Term Milestones

### M-text-1

- implement text-critic maturity-sensitive safety
- rerun frozen 4-case AIIB gate

### M-graph-1

- build graph-structured critic dataset layer
- implement offline graph critic v1

### M-graph-2

- compare graph critic versus text critic on frozen `critic_dev`

### M-graph-3

- only if `M-graph-2` is positive, run the frozen 4-case runtime graph-critic
  pilot

### M-paper

- continue the main `ours-eig` paper experiments regardless of critic status

## Final Recommendation

The current dataset is enough for a bounded text-critic pilot, but **not yet
enough for a convincing graph-critic claim**.

So the correct next step is:

1. stabilize the text critic quickly
2. start the graph critic offline-first
3. expand unique development groups while the graph critic is being built
4. keep `ours-eig` moving as the main paper method
