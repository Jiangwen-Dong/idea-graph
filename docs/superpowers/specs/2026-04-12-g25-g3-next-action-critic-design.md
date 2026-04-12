# G2.5 Candidate-Slate Export and G3 Next-Action Critic Design

## Purpose

This spec defines the next learned-control stage for `EIG` after the current
`G2` dataset construction pass.

The immediate goal is **not** to train the full final graph critic claimed by
the long-term method vision. The immediate goal is to build a **clean,
low-data, paper-honest pilot** that:

- predicts the next action from the current graph state,
- includes `commit` as a valid action,
- matches the final controller interface closely enough to be scientifically
  meaningful,
- avoids turning the current dataset stack into an unmaintainable mixture of
  partially overlapping formats.

The core design choice is:

> keep `G2` frozen as the stable trajectory-and-label dataset, add a separate
> `G2.5` candidate-slate layer, and implement `G3` as a scorer-based
> next-action critic over that candidate slate.

This preserves the long-term graph-encoder controller framing while making the
first learned pilot feasible under the current low-data regime.

## Why We Are Not Starting With A Decoder

The controller-level objective remains unchanged:

- input: current graph state
- output: the next action, including `commit`

However, there are two different ways to realize that objective:

1. `encoder + decoder`
   generate the next action directly
2. `encoder + scorer`
   enumerate candidate actions, score each candidate, and choose top-1

Under the current dataset, the scorer formulation is the better first step.

Reason:

- the current training data is small in effective benchmark-instance groups
- target node ids are graph-local and do not define a stable global label space
- one-step chosen-action logs are more naturally converted into candidate
  ranking data than into a free-form decoder target
- `commit` fits naturally as one more candidate action
- the scorer formulation preserves controller validity because the final output
  is still one full action

So the method objective stays the same, but the **first training formulation**
changes to a candidate-slate scorer.

## Design Principles

### 1. Freeze Existing Dataset Layers

The current dataset stack should stop growing by ad hoc field accretion.

We therefore freeze the meaning of:

- `G1`: raw trajectory export
- `G2`: split-ready critic dataset with weak/native labels

`G2` should remain the stable base layer for later critic experiments. Future
work should derive from it rather than mutating it in place whenever possible.

### 2. Add New Semantics In A New Layer

Candidate-slate supervision is meaningfully different from trajectory export.
It deserves its own layer:

- `G2.5`: candidate-action dataset derived from `G1` + `G2`

This keeps the dataset stack clean:

- `G1`: what happened
- `G2`: how we split and label states
- `G2.5`: what the controller could have done at each state

### 3. Keep The Long-Term Controller Interface Intact

Even though `G3` is text-only and scorer-based, it should already match the
future controller interface:

- input: state plus feasible candidates
- output: a score for each candidate
- decision: choose top-ranked candidate, including `commit`

Later `G4` can replace the text state encoder with a graph encoder while
keeping the same action-scorer interface.

### 4. Keep The Pilot Honest

This stage is a low-data pilot.

The paper framing should therefore say:

- `G3` is a modest text-only next-action critic
- it is a signal-validation pilot, not the final learned-controller claim
- the graph-structured controller remains the long-term target

## Layered Architecture

### G1: Trajectory Export (Frozen)

Current responsibility:

- saved run manifests
- transition examples
- pre-action state snapshots
- trace, token, and runtime indicators

No new semantic training fields should be added here unless they are universally
needed by all later derived layers.

### G2: Split-Ready Critic Dataset (Frozen Contract)

Current responsibility:

- benchmark-instance-level leakage-safe split assignment
- weak-local label package
- native label package
- coverage statistics

`G2` remains the authoritative source for:

- `group_id`
- `split`
- weak/native targets
- duplicate burden

### G2.5: Candidate-Slate Dataset (New)

New responsibility:

- derive a feasible candidate set for each state
- represent each candidate action in a clean, model-facing schema
- identify the logged chosen action when present
- attach weak/native downstream targets inherited from `G2`

This is the immediate new dataset layer needed for `G3`.

### G3: Text Critic (New)

Responsibility:

- score candidate actions from flattened state text plus flattened candidate
  text
- choose the top-ranked action
- optionally score `commit`

This is the first learned next-action critic.

### G4: Graph Critic (Future)

Responsibility:

- replace text-only state encoding with graph-structured encoding
- keep the same candidate scoring interface as `G3`

This preserves continuity between the pilot and the long-term method claim.

## G2.5 Dataset Design

## Input Sources

`G2.5` should be built from:

- `G1` trajectory examples
- `G1` state snapshots
- `G2` split manifest and label package

It should **not** rescan raw experiment directories directly.

## Unit Of Data

The primary training unit is:

- one graph state
- one candidate action

So `G2.5` should be a **candidate-level** dataset, not only a
state-level dataset.

## Candidate Set

For each state, build a finite candidate slate:

1. the logged selected action
2. deterministic policy candidates from the current editor/controller
3. a small bounded set of generic feasible alternatives
4. a special `commit` candidate

The candidate set should stay small and explicit. This is a low-data pilot, so
the goal is not exhaustive search. The goal is to create a valid and useful
decision slate.

## Candidate Representation

Each candidate row should contain:

- `state_id`
- `candidate_id`
- `group_id`
- `split`
- `run_dir`
- `step_index`
- `benchmark`
- `instance_name`
- `role`
- `round_name`
- `candidate_kind`
- `candidate_targets`
- `candidate_target_types`
- `candidate_target_texts`
- `candidate_source`
- `is_commit`
- `is_logged_selected`
- `state_text`
- `candidate_text`
- `weak_local`
- `native`
- `targets`

Where:

- `state_text` is a deterministic flattened summary of the current graph state
- `candidate_text` is a deterministic flattened summary of the candidate action

## Clean Action Semantics

Because node ids are graph-local, target identity should be represented through:

- node type
- node text
- local node id only as auxiliary metadata

The model-facing text should never depend on raw local ids alone.

This is one of the main reasons the scorer formulation is cleaner than a
decoder baseline at the current stage.

## Labels For G2.5

Each candidate row should expose:

- `is_logged_selected`
  - binary label for imitation-style learning
- `weak_value_01`
  - inherited from `G2`
- `native_value_01`
  - inherited from `G2` when available

Optional later fields:

- `teacher_predicted_gain`
- `heuristic_rank`
- `hindsight_commit_gain`

These are useful later, but they are not required for the first `G2.5` pass.

## Data Cleanliness Rules

To keep the dataset stack understandable, `G2.5` should obey:

1. no mutation of `G2` files
2. no hidden coupling to raw run directories
3. no benchmark-specific special-case fields unless documented explicitly
4. no mixed row types in one JSONL file
5. one dataset layer, one responsibility

Recommended output layout:

`outputs/graph_critic_datasets/<dataset_name>_g25/`

Files:

- `candidate_dataset.jsonl`
- `state_manifest.jsonl`
- `candidate_schema.json`
- `dataset_stats.json`
- `README.md`

## G3 Text-Critic Design

## Objective

Given a state and a candidate slate, score each candidate and choose:

\[
a_t^* = \arg\max_{a \in \mathcal{A}(G_t) \cup \{\mathrm{commit}\}} s_\phi(G_t, a)
\]

For the pilot, `s_\phi` is a text-based scorer rather than a graph encoder.

## Input Form

For each candidate row, create model input from:

- flattened graph-state text
- flattened candidate-action text
- lightweight metadata such as benchmark name or role when useful

This keeps the first model simple and compatible with the low-data setting.

## Model Family

The pilot should stay modest.

Recommended order:

1. very small lexical or bag-of-features baseline
2. sentence-embedding or lightweight text encoder baseline
3. shallow scorer head

The point is not to maximize benchmark score yet. The point is to answer:

> does the current dataset contain learnable signal for next-action selection?

## Training Targets

The first `G3` pilot should optimize the cleanest target first:

- primary:
  `is_logged_selected`

Supported auxiliary targets:

- weak-local return prediction
- native return prediction on labeled subset

But the first benchmark question is action selection, so the core task should
be:

- rank the logged action above alternative candidates from the same state

## Evaluation For G3

The first `G3` pilot should report:

- top-1 action accuracy against the logged chosen action
- top-k recall
- MRR or mean rank of the logged chosen action
- accuracy on `commit` versus non-commit when commit appears in the slate
- split-wise results on the held-out benchmark-instance groups

Secondary analysis:

- weak-local return correlation
- native-return correlation on labeled subset

This keeps the pilot aligned with the actual controller objective without
overclaiming end-task gains too early.

## Relationship To The Final Controller

The long-term controller remains:

- graph encoder
- candidate action encoder
- scorer over feasible action set including `commit`

So the intended progression is:

- `G3`:
  text state encoder + candidate scorer
- `G4`:
  graph state encoder + candidate scorer

The scorer-based interface is therefore not a detour. It is the stable
controller interface across the low-data and higher-capacity stages.

## Proposed Immediate Scope

### In Scope Now

- define `G2.5`
- build candidate-slate export from current `G2`
- train a modest text-only next-action critic
- report low-data pilot action-selection metrics

### Explicitly Out Of Scope Now

- graph decoder
- free-form action generation
- large graph encoder
- commit calibration claims
- full controller integration into generation

These should wait until the pilot proves the current data has usable signal or
until the dataset grows materially.

## Risks

### Risk 1: Candidate slates are too weak

If the alternative candidates are too trivial, the critic may appear good
without learning meaningful control.

Mitigation:

- include both logged chosen actions and plausible deterministic alternatives
- audit candidate diversity in `G2.5` statistics

### Risk 2: Data imbalance dominates training

A few benchmark instances currently dominate the dataset.

Mitigation:

- report group imbalance explicitly
- consider group-aware reweighting in the pilot
- keep claims conservative

### Risk 3: Logged actions are not always optimal

The chosen action is the action taken, not necessarily the best action.

Mitigation:

- frame `G3` as imitation-plus-signal validation
- reserve stronger optimality claims for later richer supervision

## Immediate Next Step After Spec Approval

1. write the implementation plan for `G2.5` candidate-slate export
2. implement the clean `G2.5` dataset builder
3. build the first full `G2.5` dataset from the current aggregated `G2`
4. implement the modest `G3` text-only scorer
5. report low-data pilot metrics with explicitly conservative framing
