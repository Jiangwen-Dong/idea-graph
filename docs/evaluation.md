# Evaluation

This repository supports two different evaluation layers:

- a `paper-facing` evaluation protocol for scientific ideation
- a `development-time` local deterministic evaluator for rapid iteration

These two layers should not be conflated.

## Paper-Facing Evaluation Protocol

For the paper, evaluation should follow a `three-layer design`.

### 1. Benchmark-Native Automatic Metrics

These are the primary automatic metrics.

`AI_Idea_Bench_2025`

- `I2T`
- `I2I`
- `IMCQ`
- `IC`
- `NA`
- `FA`
- `FPS`

`LiveIdeaBench`

- `Originality`
- `Feasibility`
- `Fluency`
- `Flexibility`
- `Clarity`
- `Average`

Guidelines:

- prefer the released benchmark protocols over ad hoc replacement metrics
- if a metric requires unavailable external assets or a batch-level comparison
  pool, mark it unavailable instead of approximating it silently
- treat `IC` as a matched-batch metric rather than a single-run score

### 2. Shared Human Blind Review

Use the same rubric on both benchmarks:

- `Novelty`
- `Significance`
- `Feasibility`
- `Clarity`
- `Context Adherence`
- `Overall`

Recommended setup:

- balanced subset from both benchmarks
- blind evaluation
- `3` reviewers per idea when feasible

### 3. Supplementary Process And Cost Analysis

These metrics validate the mechanism of the method, but they are not the main
scientific outcome metrics.

- `Evidence coverage`
- `Contradiction resolution`
- `Claim-chain completeness`
- `Commit round`
- `Action diversity`
- graph-critic action accuracy against hindsight labels, when available
- commit-vs-continue calibration
- `API and token cost`
- `Wall-clock runtime`

For the graph-critic track, process metrics should answer whether the learned
controller improves decision making over the heuristic prototype. They should
not be merged into the main benchmark score.

Recommended critic-specific ablations:

- `ours-eig-heuristic`
  the current utility/maturity controller
- `ours-eig-critic-text`
  a learned critic over flattened graph summaries
- `ours-eig-critic-graph`
  a learned critic over the structured idea graph
- `ours-eig-critic-no-commit`
  learned edit selection with fixed stopping
- `ours-eig-critic-calibrated`
  graph critic with calibrated commit decision

## Development-Time Local Deterministic Evaluator

The repository also includes a local deterministic evaluator that writes:

- `evaluation.json`
- `evaluation.md`

This evaluator is useful for:

- debugging prompt or controller changes
- ranking local ablations quickly
- checking whether graph maturity and grounding signals move in the expected
  direction

It is `not` the main paper-facing evaluator.

In particular:

- do not use the merged local heuristic `overall_score` as the headline result
- do not mix graph-process scores into the main comparison against non-graph
  baselines
- do not replace benchmark-native scores or human review with local heuristic
  surrogates in the final paper

## Benchmark-Native Scoring Artifacts

When `--native-eval` is enabled, the pipeline additionally writes:

- `benchmark_native_evaluation.json`
- `benchmark_native_evaluation.md`

These are the preferred automatic artifacts for paper reporting when the needed
judge model and benchmark assets are available.

## Recommended Reporting Structure

### Main Paper

- benchmark-native automatic metrics
- human blind-review subset

### Main Or Secondary Analysis

- cross-benchmark breakdowns
- quality-cost tradeoff discussion

### Appendix

- graph-process diagnostics
- fallback and controller analysis
- commit trajectories across rounds
- critic calibration curves
- extra ablations on prompt, retrieval, or stopping policies

## Practical Repo Policy

Use the local deterministic evaluator for fast iteration and internal decision
making. Use benchmark-native metrics and human evaluation for the final paper
claims.
