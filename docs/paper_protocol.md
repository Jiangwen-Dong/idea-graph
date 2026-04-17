# Paper Protocol

This document fixes the paper-facing protocol for the current repository under
the theme of `scientific ideation`. The goal is to keep the benchmark task,
method comparison, and evaluation design consistent with recent scientific
idea-generation literature while staying honest about what our repository can
and cannot currently measure.

## Scope

The paper studies `scientific ideation`, not the entire autonomous-research
stack.

- Input:
  a benchmark-defined ideation prompt plus the benchmark-permitted context
- Output:
  one structured research idea
- Main claim:
  explicit collaborative state tracking plus learned graph-state control
  improves scientific ideation quality
- Non-claim:
  this paper does not claim end-to-end autonomous research automation

## Benchmarks

### Main Benchmark

1. `AI_Idea_Bench_2025`

- Role in paper:
  primary benchmark
- Why it is the best fit:
  - directly targets literature-grounded AI research ideation
  - uses inspiration-to-target construction rather than unconstrained prompting
  - provides hidden target-paper fields for benchmark-native automatic scoring
  - large scale with `3,495` target papers
- Caveat:
  - AI-domain only
  - partly reconstructive because evaluation is anchored to held-out target
    papers

### Secondary Benchmark

2. `LiveIdeaBench`

- Role in paper:
  secondary robustness benchmark
- Why it is included:
  - broader scientific-domain coverage
  - tests keyword-conditioned ideation under much lighter context
  - complements the literature-grounded setting of `AI_Idea_Bench_2025`
- Caveat:
  - not literature-grounded in the same sense as `AI_Idea_Bench_2025`
  - should be interpreted as a robustness benchmark rather than the main task

### Non-Core Benchmarks

These are useful references, but they should not be the main paper benchmarks.

- `IdeaBench`
  useful as prior benchmark work, but less aligned with our target setting than
  `AI_Idea_Bench_2025`
- `ResearchBench`
  promising as a diagnostic benchmark for inspiration retrieval and composition,
  but not necessary for the main paper
- `ScienceAgentBench` and `MLR-Bench`
  broader research-agent or execution benchmarks rather than pure scientific
  ideation benchmarks

## Baselines

## Method Track

The forward paper method is `parallel EIG with a learned two-head graph
critic`.

- `Idea Graph Evolution` is the representation and collaboration process:
  agents edit a shared graph of partial scientific claims instead of rewriting a
  single draft after every turn.
- The active runtime protocol is `parallel_graph_v2`: the graph critic reads
  the current graph, selects role-local edit actions including `skip`, roles
  edit in parallel, materialized graph actions are applied, and the commit head
  checks the post-round graph.
- The `graph critic` is the controller with one shared graph encoder and two
  heads:
  - an edit/action head for role-local action selection
  - a commit head for post-round maturity prediction
- The earlier heuristic utility/maturity controller remains useful as:
  - a pre-critic prototype
  - an ablation baseline
  - a source of offline trajectories for critic training

This framing avoids treating maturity as a hand-tuned threshold. In the new
track, commitment is an adaptive action selected by the learned critic.

### Frozen Paper-Eval Split

The active frozen paper-eval split is tracked at:

- `data/splits/parallel_v2/paper_eval_v2_registry.jsonl`

It contains `256` groups: `128` from `AI_Idea_Bench_2025` and `128` from
`LiveIdeaBench`. The tracked disjointness audit reports zero overlap with the
critic train/dev pool used for teacher collection, critic training, checkpoint
selection, and calibration.

### Main Table Baselines

The recommended main comparison set is:

For `ai-researcher`, `scipip`, and `virsci`, the paper-facing claim should be
`benchmark-faithful reproduction under a unified evaluation interface`. Use
exact-upstream wording only when the exact upstream benchmark entrypoint is
actually used.

1. `direct`

- one-pass single-agent idea generation
- serves as the lower-bound baseline

2. `self-refine`

- single-agent draft, critique, and revision
- isolates iterative refinement without explicit multi-agent collaboration

3. `ai-researcher`

- preferred ideation-specific literature baseline
- should refer to the ICLR 2025 human-study ideation line rather than a broad
  end-to-end autonomous-research stack
- this is the strongest direct baseline for literature-grounded scientific
  ideation

4. `scipip`

- structured retrieval-and-decomposition literature baseline
- should be reported through the benchmark-faithful bridge or upstream path
  that preserves its decomposition-plus-synthesis structure
- useful because it tests whether structured problem breakdown alone explains
  the gains of the graph-based method

5. `virsci`

- preferred multi-agent literature baseline
- closest prior work to our collaboration-centered positioning
- should be reported as a benchmark-faithful fixed-topic bridge unless the
  exact upstream benchmark entrypoint becomes available; do not relabel a local
  proxy as `virsci` in the main table

6. `ours-eig`

- the main proposed method
- should be described as `Evolving Idea Graphs`
- the final paper target is the graph-critic variant, with the heuristic
  controller reported as an ablation or pilot baseline if needed

### Appendix Or Optional Baselines

- `Nova`
  useful as a search/planning ideation baseline if reproducible

### Development-Only Proxy Baselines

The local proxy wrappers remain useful for internal iteration, but they should
not be the headline evidence when exact or paper-faithful baselines are
available.

- `ai-researcher-proxy`
- `scipip-proxy`
- `virsci-proxy`

These must be labeled explicitly as `proxy` or `local approximation` whenever
reported.

## Shared Input Contract

All compared methods must consume the same benchmark-facing information budget.

### `AI_Idea_Bench_2025`

Allowed during generation:

- benchmark topic
- benchmark-provided inspiration or reference titles
- benchmark-safe reference snippets

Blocked during generation:

- target paper title
- gold motivation
- gold method summary
- held-out target-paper snippets

Reason:

- these fields are evaluation targets in the benchmark protocol
- exposing them during generation creates label leakage

### `LiveIdeaBench`

Allowed during generation:

- benchmark keyword prompt

Blocked during generation:

- scored benchmark idea text
- critique text
- held-out benchmark annotations

Reason:

- the benchmark should remain a keyword-conditioned ideation task

## Shared Output Contract

Every method should produce exactly one structured idea with the following
paper-facing sections:

- `Title`
- `Problem`
- `Existing Methods and Limitation`
- `Motivation`
- `Core Idea / Hypothesis`
- `Method Sketch`
- `Experiment Plan`
- `Expected Contribution`
- `Risk / Caveat`

The repository serialization maps these into:

- `title`
- `problem`
- `existing_methods`
- `motivation`
- `hypothesis`
- `method`
- `evaluation`
- `significance`
- `caveats`

## Evaluation

The paper should use a `three-layer evaluation design`.

### Layer 1: Benchmark-Native Automatic Metrics

These are the primary automatic metrics.

`AI_Idea_Bench_2025`

- `I2T`
- `I2I`
- `IMCQ`
- `IC`
- `NA`
- `FA`
- `FPS`

Notes:

- `IC` is a cross-system batch metric and should be computed on matched run
  pools rather than on single runs
- if some released metrics require unavailable assets or infrastructure, they
  must be marked unavailable rather than silently approximated

`LiveIdeaBench`

- `Originality`
- `Feasibility`
- `Fluency`
- `Flexibility`
- `Clarity`
- `Average`

### Layer 2: Shared Human Blind Review

Use the same human rubric on both benchmarks:

- `Novelty`
- `Significance`
- `Feasibility`
- `Clarity`
- `Context Adherence`
- `Overall`

Recommended evaluation subset:

- the frozen `256`-group paper-eval pool when budget allows
- a smaller balanced subset may be used only for pilot or ablation smoke runs
- `3` reviewers per idea when feasible

### Layer 3: Supplementary Mechanism And Cost Analysis

These are not headline outcome metrics, but they are necessary to validate the
collaboration mechanism and the learned controller.

- `Evidence coverage`
- `Contradiction resolution`
- `Claim-chain completeness`
- `Commit round`
- `Action diversity`
- critic-vs-heuristic action agreement
- premature-commit and late-commit rates when commit labels are available
- `API and token cost`
- `Wall-clock runtime`

## Role Of The Local Deterministic Evaluator

The repository's local deterministic evaluator is useful for development,
ablation debugging, and rapid iteration, but it should not be the main paper
evidence.

- keep it for controlled internal comparison
- do not use its merged heuristic score as the headline paper metric
- do not combine graph-process scores with benchmark-outcome scores in the main
  result table

## Fairness Controls

The main comparison should keep the following fixed whenever possible:

- same backbone model family
- same benchmark-mode input packet
- same output schema
- no hidden benchmark leakage during generation
- anonymized outputs for human review

The main table does `not` need to be matched-budget. Quality is the primary
claim. Cost should be reported explicitly as a separate axis rather than folded
into the main quality score.

## Recommended Main Experimental Table

Report:

- benchmark-native automatic metrics on `AI_Idea_Bench_2025`
- benchmark-native automatic metrics on `LiveIdeaBench`
- shared human blind-review scores on a balanced subset

Move graph diagnostics, fallback analysis, and cost analysis to dedicated
analysis tables or the appendix.

## Repo Status

The repository currently supports:

- benchmark-mode input packets for `AI_Idea_Bench_2025` and `LiveIdeaBench`
- parallel EIG v2 runtime replay with role-local selected decisions, edit
  patches, materialized graph actions, and post-round commit labels
- two-head critic dataset and training utilities for the parallel-v2 replay
- external baseline adapter entrypoints for `ai-researcher`, `scipip`, and
  `virsci`, provided the corresponding upstream repositories and assets are
  installed outside this repo
- local proxy wrappers for rapid iteration
- local deterministic evaluation artifacts
- optional benchmark-native evaluation artifacts

The paper-facing recommendation is to prioritize exact or benchmark-faithful
baselines and benchmark-native scoring, while treating local proxies and local
heuristic scores as development tools unless they are explicitly labeled
otherwise.
