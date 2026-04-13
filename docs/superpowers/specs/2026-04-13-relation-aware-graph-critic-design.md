# Relation-Aware Graph Critic Design

**Date:** 2026-04-13  
**Scope:** first learned graph encoder for offline next-action ranking on the
frozen `development_pool_v2_combined_g25` split

## Purpose

This spec defines the next graph-critic stage after the negative result of the
first lightweight graph-feature baseline.

The immediate goal is not to build the final full controller. The immediate
goal is to test a stronger and cleaner hypothesis:

> Preserving the evolving idea graph as a relation-aware graph, while encoding
> node text explicitly, improves offline next-action ranking over both
> flattened text and shallow hand-crafted graph features.

This stage remains **offline only**. It must beat the refreshed text scorer on
the same frozen validation groups before any controller-in-the-loop testing is
allowed.

## Why The Current Baseline Is Not Enough

The current graph-feature baseline uses:

- hand-crafted graph counts
- node-type and role histograms
- target-node type summaries
- a linear scorer

That preserves some structure, but it does not truly model:

- node semantics
- relation-specific information flow
- whether a candidate action is compatible with the target subgraph
- whether local graph context supports or undermines a proposed edit

As a result, the current baseline underperforms the refreshed text scorer on
the frozen `critic_dev` split.

## Design Requirements

The next graph critic must satisfy five constraints.

### 1. Preserve Graph Structure

The model must operate on nodes and edges directly. It cannot flatten the full
state back into one plain-text string.

### 2. Integrate Node Text

Each node representation must include the claim text itself, not only node
metadata.

### 3. Stay Small Enough For The Current Dataset

The available development-only dataset contains:

- `1267` states
- `13004` candidate rows
- `23` leakage-safe groups
- `6` frozen validation groups

This supports a small learned graph encoder. It does not justify a large graph
transformer trained aggressively from scratch.

### 4. Match The Existing Candidate-Slate Interface

The model must keep the current controller framing:

- input: graph state plus one candidate action
- output: a scalar score

This keeps the new model comparable to:

- the refreshed text scorer
- the first graph-feature scorer
- the long-term graph-controller story

### 5. Stay Offline First

This stage should not change runtime policy. It should only train and evaluate
on the frozen offline split.

## Approaches Considered

### Option A: Larger Hand-Crafted Feature Expansion

Add more structural counters, branch features, and target statistics.

Pros:

- low implementation burden
- cheap to train

Cons:

- still weak on node semantics
- unlikely to close the current gap to the text scorer
- weak reviewer story because it remains mostly feature engineering

### Option B: Relation-Aware Graph Encoder With Frozen Text Embeddings

Use frozen node-text embeddings, relation-aware message passing, and a
target-aware action scorer.

Pros:

- directly tests whether graph structure plus node semantics help
- fits the current data scale
- aligns with the paper motivation about preserving relational state

Cons:

- more implementation work than a feature baseline
- requires careful batching and masking

### Option C: Full Graph Transformer Or External GNN Stack

Use a heavier graph library or a larger graph transformer.

Pros:

- strongest raw model capacity

Cons:

- higher environment and dependency burden
- higher overfitting risk on the current dataset
- harder to defend as the first learned graph step

## Chosen Approach

Use **Option B**.

The next model is a **small relation-aware graph encoder with frozen text
embeddings and target-aware candidate scoring**.

This is the best balance between scientific value, implementation tractability,
and paper clarity.

## Model Overview

For each graph state \(G_t = (V_t, E_t)\) and candidate action \(a\), the
model predicts:

\[
Q_\theta(G_t, a).
\]

The model has four stages:

1. node initialization
2. relation-aware message passing
3. target-aware candidate representation
4. candidate scoring head

## Data Contract

### Frozen Inputs

The model uses the same offline roots as the current graph-feature baseline:

- candidate slates:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25`
- graph snapshots:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g1`
- partition manifest:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g2_partitions/partition_manifest.jsonl`

The split contract remains:

- `critic_train` for training
- `critic_dev` for validation
- ignore `paper_eval`

### Per-Node Fields

Each node contributes:

- `text`
- `type`
- `role`
- `confidence`
- `evidence_count`
- `status`
- `branch_id`

### Per-Edge Fields

Each edge contributes:

- source node id
- destination node id
- relation type
- resolved flag

### Per-Candidate Fields

Each candidate contributes:

- `candidate_kind`
- `candidate_text`
- `candidate_target_ids`
- `is_commit`

### Global Context Fields

Each state may additionally contribute:

- `benchmark`
- `round_name`
- `role`
- optional frozen embedding of `state_text`

These are auxiliary context features, not the primary representation.

## Node Initialization

Each node starts from a text-aware embedding:

\[
h_i^{(0)} = \mathrm{MLP}([e^\text{text}_i; e^\text{type}_i; e^\text{role}_i; s_i]),
\]

where:

- \(e^\text{text}_i\) is a frozen embedding of node text
- \(e^\text{type}_i\) is a learned node-type embedding
- \(e^\text{role}_i\) is a learned role embedding
- \(s_i\) contains scalar node features

### Text Encoder Choice

The first implementation should use a small frozen sentence encoder, cached
offline.

Preferred first choice:

- `sentence-transformers/all-MiniLM-L6-v2`

Reason:

- light enough for the current workflow
- stable and widely used
- sufficient for a first semantic graph encoder

If the local environment makes this unavailable, the fallback is a deterministic
TF-IDF projection or another small frozen encoder, but the preferred design is
MiniLM-style dense embeddings.

## Relation-Aware Message Passing

Use a shallow relation-aware message passing network with `2` layers.

For node \(i\):

\[
m_i^{(\ell)} = \sum_{j \in \mathcal{N}(i)}
W_{\mathrm{rel}(j, i)}^{(\ell)} h_j^{(\ell)},
\]

\[
h_i^{(\ell + 1)} =
\mathrm{LayerNorm}\left(
h_i^{(\ell)} +
\mathrm{MLP}^{(\ell)}([h_i^{(\ell)}; m_i^{(\ell)}; r_i^{(\ell)}])
\right),
\]

where \(r_i^{(\ell)}\) can include local scalar summaries such as:

- incoming support count
- incoming contradiction count
- resolved-neighbor fraction

This keeps the model small while preserving:

- node identity
- edge types
- local neighborhood structure

## Candidate Representation

Each candidate action gets a target-aware representation built from four parts.

### 1. Action Embedding

Embed:

- `candidate_kind`
- `is_commit`
- frozen embedding of `candidate_text`

### 2. Target Pool

Pool the encoded target-node embeddings for `candidate_target_ids`.

If there are no targets, use a learned null target vector.

### 3. Local Neighborhood Pool

Pool one-hop neighbors of the target nodes, grouped by relation type when
possible.

This gives the scorer local graph context around the proposed edit.

### 4. Global Graph Summary

Build a graph summary from all encoded nodes using masked mean pooling or a
small attention pool.

The first implementation should use masked mean pooling for stability and
simplicity.

### Final Candidate Vector

\[
z_a = [z^\text{graph}; z^\text{target}; z^\text{nbr}; z^\text{action}; z^\text{ctx}]
\]

where `ctx` contains optional benchmark and round context.

## Scoring Head

Predict:

\[
Q_\theta(G_t, a) = \mathrm{MLP}(z_a).
\]

The score is scalar. Higher means more likely to be the correct next action for
that state.

## Training Objective

The main loss should be **state-local ranking**, not only independent binary
classification.

### Primary Loss

For each state, apply a softmax over candidate scores and optimize the logged
selected candidate as the positive label.

This aligns the training objective with offline evaluation:

- exactly one logged positive candidate per state
- top-1 and MRR are state-local ranking metrics

### Secondary Stability Term

An optional auxiliary BCE loss may be added if optimization is unstable, but it
should remain secondary to the slate-ranking objective.

### Commit Handling

Keep `commit` in the offline slate for comparability, but do not deploy learned
commit at runtime in this stage.

The offline report should include:

- all-candidate ranking
- edit-only ranking

This helps separate edit quality from commit difficulty.

## Evaluation Protocol

Use the same frozen offline gate as the current baseline comparison.

### Mandatory Metrics

- validation top-1 accuracy
- validation mean reciprocal rank

### Mandatory Comparisons

- refreshed text scorer
- first graph-feature scorer
- new relation-aware graph scorer

### Required Fairness Conditions

All three models must use:

- the same candidate slates
- the same train/dev partitions
- the same commit weighting policy when applicable

### Promotion Gate

The new graph scorer is allowed to move toward runtime testing only if it
clearly beats the refreshed text scorer on the frozen validation groups.

If it does not, it remains an offline research artifact only.

## Implementation Boundaries

### In Scope

- a new graph-critic dataset loader derived from existing `G1` and `G2.5`
  artifacts
- cached frozen text embeddings for node and candidate text
- a small PyTorch relation-aware graph encoder
- an offline training script
- offline validation metrics and saved artifacts
- unit tests for graph batching, masking, target pooling, and offline scoring

### Out Of Scope

- runtime controller integration
- learned online adaptation
- learned `commit` deployment
- full graph transformer variants
- large benchmark generation reruns

## Risks And Mitigations

### Risk 1: Dataset Too Small For A Larger Encoder

Mitigation:

- freeze text embeddings
- keep message passing shallow
- keep hidden sizes modest
- start with masked mean graph pooling

### Risk 2: Training Becomes Hard To Diagnose

Mitigation:

- keep a clean separation between:
  - text baseline
  - feature baseline
  - relation-aware graph model
- log per-split ranking metrics consistently
- keep the same frozen offline gate

### Risk 3: Commit Candidates Distort Early Conclusions

Mitigation:

- report all-candidate and edit-only metrics
- keep learned runtime commit disabled

## Success Criteria

This stage is successful if it delivers:

1. a clean offline implementation of the relation-aware graph scorer
2. a reproducible frozen-split comparison against text and feature baselines
3. clear evidence about whether graph structure plus node semantics improve
   next-action ranking

This stage is **not** required to prove full controller superiority yet.

## Recommended Next Planning Step

Once this spec is approved, the implementation plan should follow three short
stages:

1. data and embedding pipeline
2. relation-aware graph model and offline trainer
3. frozen offline comparison and decision gate
