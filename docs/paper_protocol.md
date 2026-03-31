# Paper Protocol

This document fixes the experimental protocol for the current repository so the
same benchmark-facing task, output schema, and comparison setup can be reused
across methods.

## Scope

The project studies scientific idea generation under a shared, benchmark-aware
interface.

- Input: a benchmark-defined topic plus the benchmark-provided context packet
- Output: one structured research idea in a fixed schema
- Primary setting: benchmark-faithful ideation
- Secondary setting: human blind review on a balanced subset

## Benchmarks

### Core Public Benchmarks

1. `AI_Idea_Bench_2025`

- Role in paper: primary benchmark
- Why it is included:
  - strong task match for literature-grounded ideation
  - explicit inspiration-to-target construction
  - hidden target-paper fields for automatic evaluation
  - large scale with `3,495` target papers
- Caveat:
  - AI-domain only
  - partly reconstructive rather than purely open-ended

2. `LiveIdeaBench v2`

- Role in paper: secondary robustness benchmark
- Why it is included:
  - broad scientific-domain coverage
  - minimal-context creativity stress test
  - useful complement to literature-grounded ideation
- Caveat:
  - not literature-grounded in the same way as `AI_Idea_Bench_2025`
  - should be interpreted as a robustness benchmark rather than the main task

### Human Evaluation Layer

Use expert blind review on a balanced subset from both public benchmarks.

- Suggested size:
  - `60` instances total
  - balanced across the two benchmarks
  - `3` reviewers per idea

## Evaluation

### Automatic Metrics

Use benchmark-native automatic metrics instead of forcing one artificial metric
across all datasets.

`AI_Idea_Bench_2025`

- `IMCQ`
- `I2I`
- `I2T`
- `NA`
- `FA`
- `FPS`
- `IC`
  Note: `IC` is a cross-system batch metric and should be computed over matched
  run sets, not single runs.

`LiveIdeaBench v2`

- `Originality`
- `Feasibility`
- `Fluency`
- `Flexibility`
- `Clarity`
- `Average`

### Shared Human Metrics

Use the same human rubric on both benchmarks:

- `Novelty`
- `Significance`
- `Feasibility`
- `Clarity`
- `Overall`

### Supplementary Process Metrics

Treat process metrics as supplementary validation and ablation evidence.

- `Evidence coverage`
- `Contradiction resolution`
- `Action diversity`
- `API and token cost`

These metrics validate the collaboration mechanism, but they are not the main
scientific outcome metrics.

## Baselines

### Runnable Local Baselines

These baselines are implemented directly in this repository and share the same
benchmark-mode input and output schema.

1. `ours-delayed-consensus`

- Multi-agent typed-graph collaboration with delayed consensus
- This is the main method

2. `direct`

- One-pass single-agent idea generation
- Lower bound baseline

3. `self-refine`

- Single-agent draft, critique, and revision
- Control baseline for iterative improvement without multi-agent structure

### Exact External Baseline Wrappers

These wrappers call the official upstream repositories and require an external
configuration file.

4. `ai-researcher`

- Exact external wrapper for the ICLR `AI-Researcher` baseline
- This is the preferred literature baseline when reporting faithful external
  comparisons

5. `scipip`

- Guarded external wrapper for `SciPIP`
- Use only when its external environment and data dependencies are configured
  correctly

6. `virsci`

- Guarded external wrapper for `Virtual-Scientists`
- Not currently benchmark-faithful for fixed-topic public benchmarks, so it
  should not be used in the main benchmark table until upstream control is
  improved

### Proxy Wrappers For Prior Literature

The repository also exposes prompt-style proxy wrappers for prior systems. These
are not exact reproductions of the original codebases and should be labeled as
local approximations unless replaced by direct external integrations.

7. `ai-researcher-proxy`

- Literature-grounded candidate-generation and ranking wrapper
- Intended to approximate the style of the ICLR `AI-Researcher` ideation
  pipeline

`ResearchAgent`

- excluded from the current baseline table
- reason:
  - it is a different system than `AI-Researcher`
  - it is temporarily banned from the current paper protocol

8. `scipip-proxy`

- Structured single-agent wrapper that emphasizes motivation and experiment-plan
  decomposition
- Intended to approximate the style of `SciPIP`

9. `virsci-proxy`

- Multi-agent wrapper using the delayed-consensus engine with a more discussion-
  oriented collaboration style
- Intended to approximate the style of `VirSci`

## Shared Input Contract

All methods must consume the same benchmark-facing information budget.

### `AI_Idea_Bench_2025`

Allowed input:

- benchmark topic
- inspiration-paper titles
- benchmark-provided reference snippets

Blocked during generation:

- target paper title
- gold motivation
- gold method summary
- target-paper snippets

Reason:

- these fields are evaluation targets in the benchmark protocol
- exposing them during generation would create label leakage

### `LiveIdeaBench v2`

Allowed input:

- benchmark keyword prompt

Blocked during generation:

- scored benchmark idea text
- benchmark critique text

Reason:

- the benchmark should remain a keyword-conditioned ideation task

## Shared Output Contract

Every method must produce exactly one final structured idea with the following
sections:

- `Title`
- `Problem`
- `Existing Methods and Limitation`
- `Core Idea / Hypothesis`
- `Method Sketch`
- `Experiment Plan`
- `Expected Contribution`
- `Risk / Caveat`

The repository serialization maps this into the internal fields:

- `title`
- `problem`
- `existing_methods`
- `hypothesis`
- `method`
- `evaluation`
- `significance`
- `caveats`

The `motivation` field may still be used internally, but the paper-facing output
should avoid abstract-style repetition.

## Fairness Controls

Use the same settings across methods in the main comparison whenever possible.

- same backbone model family
- same benchmark-mode input packet
- same output schema
- same generation budget or a clearly stated matched budget
- anonymized outputs for human review

## Recommended Main Table

Report:

- `AI_Idea_Bench_2025` automatic metrics
- `LiveIdeaBench v2` automatic metrics
- human blind-review subset scores

Move graph-process diagnostics to ablations or supplementary analysis.

## Repo Status

The current repository implementation now supports:

- `AI-Researcher` as the named proxy baseline instead of `ResearchAgent`
- exact external baseline entrypoints for `ai-researcher`, `scipip`, and
  `virsci`
- benchmark-native scoring artifacts via `benchmark_native_evaluation.json`
  and `benchmark_native_evaluation.md`

The current benchmark-native scorer is strongest on the public prompt-based
metrics that can be reproduced directly from released assets. Metrics that need
extra benchmark preprocessing or cross-system pools are exposed with explicit
availability flags instead of being silently approximated.
