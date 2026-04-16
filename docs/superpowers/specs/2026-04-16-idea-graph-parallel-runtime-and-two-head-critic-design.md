# Idea-Graph Parallel Runtime And Two-Head Critic Design

**Date:** 2026-04-16  
**Scope:** revise the existing `idea-graph` repository in place so the
multi-agent runtime performs parallel same-round idea editing over a frozen
graph snapshot, while the graph critic is upgraded from a single scalar scorer
to a shared encoder with separate edit and commit heads.

## Purpose

The current `idea-graph` repository already contains:

- a stable sequential delayed-consensus runtime
- benchmark integrations and evaluation utilities
- online and offline graph-critic infrastructure
- role prompts and controller traces that are already useful

The next revision should preserve those assets rather than replace the repo.

The goal is to upgrade the internal runtime protocol so it is:

- faster in wall-clock time through parallel role calls
- cheaper through explicit `skip` handling and role activation
- cleaner architecturally through a shared-encoder two-head graph critic
- safer experimentally by preserving the current external input/output protocol

This is an in-repo protocol upgrade, not a standalone rewrite.

## Hard Constraints

The redesign must satisfy all of the following:

- remain inside the current `idea-graph` repository
- preserve the current top-level benchmark input/output contract as much as
  possible
- preserve the current five-role decomposition
- preserve existing prompts and role identities as much as possible
- allow paired comparison against the current sequential runtime
- improve wall-clock speed without silently changing benchmark-visible outputs

The redesign should **not** require a benchmark-side migration of packet
formats or final proposal schemas.

## Current Mismatch

The current engine still executes one role at a time inside each round and
materializes each role action immediately.

Consequences:

- later roles in the same round see a graph already modified by earlier roles
- same-round role proposals are not conditionally independent
- runtime wall-clock grows roughly with the number of active roles
- current graph-critic training targets are tied to sequential runtime traces

The current relation-aware graph critic also uses a single scoring pathway with
`is_commit` as a feature rather than a true split between:

- role-conditional edit selection
- graph-level global stop prediction

That design was sufficient for the first reranking line, but it is no longer
the cleanest fit for the intended method.

## Recommended Approach

### Option A: Minimal Parallel Call Patch

Keep the current engine and only parallelize backend calls.

Pros:

- fastest initial code change
- minimal immediate diff surface

Cons:

- same-round semantics would still be conceptually sequential
- graph critic architecture would remain awkward
- replay data would still mismatch the intended controller

Decision:

- reject

### Option B: Recommended In-Repo `parallel_graph_v2` Path

Add a new feature-flagged runtime path inside `idea-graph` while keeping the
current sequential runtime as the baseline.

Pros:

- preserves current external protocol and benchmark tooling
- keeps current roles and most prompts intact
- allows fast parallel execution and safe ablation against the current line
- enables a clean shared-encoder two-head critic without breaking old runs

Cons:

- temporarily duplicates some runtime logic
- requires new replay collection and new critic training

Decision:

- choose this option

### Option C: Replace The Current Default Runtime In Place

Rewrite the existing runtime path directly and retire the sequential protocol.

Pros:

- cleanest eventual codepath

Cons:

- high regression risk
- harder to compare against the old line
- too much protocol and model churn at once

Decision:

- do not use this as the first migration step

## Design Principles

### 1. Preserve External Contracts

Inputs and outputs should remain stable at the benchmark-facing layer.

Stable surfaces:

- benchmark packet loading
- run entry points
- final proposal structure
- round/run artifact conventions where practical
- evaluation wrappers

Changed surfaces:

- internal same-round execution semantics
- controller internals
- replay row semantics for graph-critic training

### 2. Preserve Roles And Mostly Preserve Prompts

The default role set remains:

- `MechanismProposer`
- `FeasibilityCritic`
- `NoveltyExaminer`
- `EvaluationDesigner`
- `ImpactReframer`

Prompt changes should be minimal and functional:

- state that every role sees the same frozen graph snapshot for the round
- ask for `K` candidates plus one explicit `skip`
- clarify that same-round proposals from other roles are not visible
- preserve the current role identity and action-family guidance

The first version should avoid prompt rewrites that could create large behavior
shift unrelated to the controller redesign.

### 3. Speed First, But Not At The Cost Of Uncontrolled Drift

The redesign should prioritize:

- parallel same-round role calls
- optional role activation before API calls
- `skip` as a first-class low-cost outcome
- deterministic action materialization order

The redesign should avoid:

- free-running asynchronous graph mutation
- roles repeatedly debating each other inside the same round
- hidden prompt/schema changes that make old and new runs incomparable

## Runtime Protocol

The new runtime mode should be named something explicit such as
`parallel_graph_v2`.

The runtime state for round `t` should be the pair:

- `(G_t, H_t)`
- `G_t`: the current idea graph
- `H_t = E(G_t)`: the cached shared-encoder representation of `G_t`

Initialization:

1. Build the initial graph `G_0`.
2. Encode once to obtain `H_0 = E(G_0)`.

At round `t`:

1. Start from cached state `(G_t, H_t)`.
2. Run a cheap role-activation gate on `G_t` and `H_t`.
3. Freeze the round snapshot at `G_t`.
4. Query all active roles in parallel on the same frozen snapshot.
5. Each active role returns a role-local candidate slate containing:
   - `K` concrete edit candidates
   - one explicit `skip` candidate
6. Validate and deduplicate each role-local slate.
7. Score each role-local slate with the edit head using cached graph
   embedding `H_t` plus role and candidate features.
8. Select exactly one `selected_role_decision` for each active role.
9. Execute the selected role decisions in parallel against the frozen
   snapshot to produce role-local `edit_patches`.
10. Map `skip` to an empty patch.
11. Deterministically merge all non-empty patches to form:
   - the updated graph `G_{t+1}`
   - the realized `materialized_graph_actions`
12. Encode the updated graph once to obtain `H_{t+1} = E(G_{t+1})`.
13. Run the global commit head only on `H_{t+1}`.
14. If commit fires, synthesize and stop.
15. Otherwise continue with cached state `(G_{t+1}, H_{t+1})`.

There should be no separate routine pre-check. The end-of-round commit check
for round `t` becomes the reused control state for round `t+1`, because if the
run continues we already have `H_{t+1}` cached for the next round's edit-head
scoring.

This keeps:

- proposal generation parallel
- graph mutation controlled and deterministic
- stopping global rather than role-local
- encoder recomputation at once per realized graph state

### Runtime Objects

The runtime should distinguish clearly between controller outputs and realized
graph mutations.

`selected_role_decisions`:

- one chosen decision per active role
- the direct supervision target for the edit head
- may be a concrete edit or `skip`

`edit_patches`:

- role-local tentative graph edits produced by executing the selected decision
  against the frozen snapshot
- do not mutate the shared graph immediately
- `skip` maps to an empty patch

`materialized_graph_actions`:

- the deterministic merged non-empty patches that are finally applied to the
  shared graph
- may be fewer than the number of active roles because some roles select
  `skip`

`post_round_commit_rows`:

- graph-level supervision rows derived from the realized post-merge graph
  `G_{t+1}`
- the direct supervision target for the commit head

### Paper-Style Pseudocode

```text
Initialize G_0
H_0 <- E(G_0)
for t = 0, 1, 2, ... do
    active_roles <- ActivateRoles(G_t, H_t)
    frozen_graph <- G_t
    slates <- ParallelCollectCandidates(frozen_graph, active_roles, include_skip=True)
    selected_role_decisions <- {
        r: EditHeadSelect(H_t, r, slates[r]) for r in active_roles
    }
    edit_patches <- ParallelExecute(frozen_graph, selected_role_decisions)
    materialized_graph_actions <- MergeNonEmptyPatches(edit_patches)
    G_{t+1} <- Apply(materialized_graph_actions, G_t)
    H_{t+1} <- E(G_{t+1})
    commit <- CommitHead(H_{t+1})
    LogRound(G_t, selected_role_decisions, materialized_graph_actions, commit)
    if commit then
        return SynthesizeFinalProposal(G_{t+1})
    end if
end for
```

## Role Activation And Skip

### Skip

Every active role must have a valid explicit `skip`.

Purpose:

- avoid low-value or repetitive edits
- reduce unnecessary graph growth
- reduce token and latency cost by normalizing no-op decisions
- give the edit head a clean abstention target

`skip` should remain a first-class candidate action in every role-local slate
and in the replay labels. It is filtered only at materialization time because
its realized `edit_patch` is empty, not because it is unimportant.

### Role Activation

Before any backend calls, the runtime may mark some roles inactive for the
round.

Version-one activation should be heuristic rather than learned.

Purpose:

- reduce latency and token cost
- avoid low-information calls when a role has little useful work to do
- preserve overall prompt behavior while making the runtime cheaper

The first implementation should log:

- active roles
- inactive roles
- activation reasons

so later learned activation can be considered with evidence.

## Graph Critic Redesign

### Shared Encoder

The existing relation-aware graph critic already contains a useful encoder core:

- node text features
- role embeddings
- node-type embeddings
- relation-aware message passing
- target-neighbor pooling

That structure should be retained where possible, but reorganized explicitly
into:

- `SharedGraphEncoder`
- `EditHead`
- `CommitHead`

### Encoder Reuse Across Rounds

The shared encoder should be run exactly once for each realized graph state.

Desired caching pattern:

- encode `G_t` once to obtain `H_t` for role activation and edit-head scoring
- after deterministic merge, encode `G_{t+1}` once to obtain `H_{t+1}` for the
  commit head
- if the commit head predicts continue, reuse `H_{t+1}` directly as the next
  round's edit-side graph representation

This is the clean reason to place commit prediction only after the round has
finished. It avoids an extra encode pass on the same graph while keeping both
heads attached to the same shared encoder.

### Edit Head

Input:

- cached graph embedding `H_t` derived from frozen graph snapshot `G_t`
- role id `r`
- candidate action `a`
- candidate targets and local neighborhood context

Output:

- score for materializing candidate `a` for role `r`

Responsibilities:

- rank role-local edit candidates
- treat `skip` as a valid outcome
- prefer specific, non-redundant, high-leverage edits
- remain role-aware

### Commit Head

Input:

- cached graph embedding `H_{t+1}` derived from realized post-round graph
  `G_{t+1}`
- graph-level progress or maturity summary features

Output:

- score or calibrated probability that the run should stop now

Responsibilities:

- judge global readiness to stop after a completed round
- remain independent of any one role or candidate text
- consume the realized post-merge graph rather than hypothetical pre-round
  choices
- provide only one routine stop decision per round

### Why Two Heads Are Necessary

Treating `commit` as just another candidate kind is no longer the right
abstraction.

Reasons:

- edit choice is role-conditional and target-aware on `G_t`
- commit is graph-global and is evaluated on realized `G_{t+1}`
- calibration is cleaner when stop prediction has its own head
- replay datasets naturally split into role-round decision rows and graph-level
  post-round commit rows
- the paper story is clearer

## Training Implications

New graph-critic training is required for the intended architecture.

### Edit Head Training

A new edit-head training set is required because:

- same-round semantics change from sequential to frozen parallel
- `skip` becomes first-class rather than incidental
- the teacher chooses at most one candidate per active role under the new
  protocol
- the supervision target is the `selected_role_decision`, not the later merged
  graph action

Warm-start option:

- encoder weights from the current relation graph critic may optionally be used
  as initialization if tensor shapes remain compatible

But:

- final edit-head training must use replay rows collected from the new
  `parallel_graph_v2` protocol

### Commit Head Training

A new commit-head training set is also required because:

- commit is now a separate prediction target
- commit rows should be graph-level post-round checks, not mixed with edit rows
- the current live evidence still suggests that enabling learned commit too
  early is risky

Commit-head supervision should be built from the realized post-round graph
state:

- input state: `G_{t+1}` and cached embedding `H_{t+1}`
- label: `commit` or `continue`
- row semantics: one graph-level row for each completed round
- no routine pre-check rows in the target design

Recommendation:

- train the commit head offline
- log it in shadow mode first
- calibrate on frozen development data
- do not enable live commit until the edit-head runtime is already stable

### Practical Answer

If the team wants the full intended design, then **yes, new graph-critic
training is needed**.

The fastest safe progression is:

1. land the parallel runtime
2. collect new replay
3. train the new edit head
4. shadow-train and calibrate the commit head
5. only then test live commit

## Replay And Dataset Redesign

The current graph-critic data pipeline should not simply relabel old sequential
traces.

New replay collection should write:

- `parallel_state_snapshots`
  - frozen `G_t` snapshots used to build edit-head rows
  - realized post-round `G_{t+1}` snapshots used to build commit-head rows
- `parallel_round_traces`
  - active and inactive roles
  - per-role candidate slates including `skip`
  - `selected_role_decisions`
  - role-local `edit_patches`
  - deterministic merge summary
  - realized `materialized_graph_actions`
  - post-round `commit` or `continue`
- `parallel_edit_rows`
  - one supervision row per active role-round
  - label is the chosen `selected_role_decision`
- `post_round_commit_rows`
  - one supervision row per completed round
  - label is the post-round `commit` or `continue` decision

Dataset outputs should be separated into:

- `edit_head_rows`
- `commit_head_rows`

During migration, compatibility adapters may still export
`candidate_dataset.jsonl` so existing loaders and evaluation code continue to
work. But the target two-head design should conceptually treat edit selection
and commit prediction as separate supervised tasks rather than as one mixed
candidate table.

Each row must record:

- label source
- schema version
- runtime protocol version

so future revisions do not become mixed accidentally.

### Supervision Label Quality Rules

High-quality supervision matters more than raw row count. Replay curation
should keep rows only when:

- candidate slates were parsed and validated successfully
- the selected role decision is unambiguous
- `skip` is preserved as a valid positive label when it was selected
- deterministic merge completed without hidden fallback behavior
- the post-round commit label is attached to the realized merged graph, not a
  hypothetical earlier graph
- train/dev/test splitting is grouped by benchmark instance or run family to
  avoid leakage across highly similar rounds

Recommended quality checks:

- inspect per-role label distributions including `skip`
- audit a sample of round traces by hand before large-scale training
- compare selected decisions against realized materialized actions to catch
  merge-path bugs
- reject rows with missing snapshot references or schema mismatches

## Compatibility Strategy

The current sequential runtime should remain available as the baseline.

Recommended runtime switch:

- `sequential_v1`
- `parallel_graph_v2`

This enables clean comparisons:

- current sequential baseline
- parallel runtime without learned critic
- parallel runtime with edit head only
- parallel runtime with edit head plus live commit

This also protects existing scripts and evaluation code from unnecessary churn.

## Concrete Module Plan

The revision should prefer extracting new modules rather than making
`engine.py` absorb all new protocol logic.

### Existing Modules Likely To Change

- `src/idea_graph/engine.py`
- `src/idea_graph/agent_backend.py`
- `src/idea_graph/action_candidates.py`
- `src/idea_graph/collaboration_protocol.py`
- `src/idea_graph/relation_graph_critic_model.py`
- `src/idea_graph/relation_graph_critic_data.py`
- `src/idea_graph/relation_graph_runtime_critic.py`

### New Modules Likely To Be Added

- `src/idea_graph/parallel_runtime.py`
- `src/idea_graph/parallel_role_executor.py`
- `src/idea_graph/role_activation.py`
- `src/idea_graph/parallel_replay.py`
- `src/idea_graph/graph_encoder.py`
- `src/idea_graph/graph_edit_head.py`
- `src/idea_graph/graph_commit_head.py`

### Responsibilities

`parallel_runtime.py`:

- frozen-round coordination
- cached runtime state `(G_t, H_t)`
- post-round commit checks only
- deterministic materialization

`parallel_role_executor.py`:

- parallel backend invocation
- candidate collection
- latency/token logging

`role_activation.py`:

- heuristic active-role selection
- activation trace logging

`parallel_replay.py`:

- write round-level replay artifacts
- write edit rows and post-round commit rows
- maintain compatibility exports for existing dataset tooling

`graph_encoder.py`:

- shared graph encoder and graph/context summary building

`graph_edit_head.py`:

- candidate scoring for role-local edits

`graph_commit_head.py`:

- graph-level post-round stop scoring and probability output

## Delivery Stages

### Stage 1: Parallel Runtime Skeleton

Goal:

- land a feature-flagged `parallel_graph_v2` runtime without changing external
  run inputs/outputs

Deliverables:

- runtime switch
- frozen graph snapshot execution
- parallel role querying
- explicit `skip`
- deterministic multi-edit materialization
- tests comparing snapshot semantics against sequential behavior

### Stage 2: Speed And Cost Controls

Goal:

- reduce latency and API cost before critic migration

Deliverables:

- heuristic role activation
- token and latency logging
- `skip` rate reporting
- cost comparison against sequential runtime

### Stage 3: Two-Head Critic Refactor

Goal:

- replace the single-score relation graph critic with a shared encoder plus two
  heads

Deliverables:

- shared encoder module
- edit head
- commit head
- runtime interfaces for separate edit and commit scoring

### Stage 4: Parallel Replay And Edit-Head Training

Goal:

- train the new edit head on protocol-matched replay

Deliverables:

- parallel replay collector
- edit-head dataset rows
- offline training loop
- runtime edit-head reranking integration

### Stage 5: Commit Shadow Mode And Calibration

Goal:

- safely assess learned stop prediction without destabilizing main runs

Deliverables:

- post-round commit-head dataset rows
- shadow commit logging
- frozen dev calibration reports
- go/no-go decision for live commit

## Risks And Mitigations

### Risk 1: Parallel Runtime Changes Too Much At Once

Mitigation:

- keep current roles and prompts mostly unchanged
- keep external packet and proposal schemas stable
- keep sequential runtime as a fallback baseline

### Risk 2: Cheap `skip` Collapses Useful Editing

Mitigation:

- log per-role skip rate
- inspect selected-versus-offered candidate mix
- require role-local slates to contain both concrete edits and `skip` where
  feasible

### Risk 3: Two-Head Critic Slows Progress

Mitigation:

- land parallel runtime first for immediate speed gains
- keep commit shadow-only at first
- focus early learned integration on the edit head

### Risk 4: Live Commit Repeats The Existing Transfer Failure

Mitigation:

- do not enable live commit immediately
- calibrate only on frozen development data
- require non-negative or clearly improved transfer before promotion

## Immediate Next Step

Write the implementation plan for:

- Stage 1: Parallel Runtime Skeleton
- Stage 2: Speed And Cost Controls
- Stage 3: Two-Head Critic Refactor

Reason:

- these stages deliver the fastest visible runtime benefit
- they preserve external compatibility
- they define the interfaces required before replay and training work

Only after that plan is approved should implementation begin.
