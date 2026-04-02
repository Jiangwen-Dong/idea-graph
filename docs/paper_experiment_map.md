# Paper Experiment Map

This note records the current experiment layout for the paper after the
cross-benchmark small-batch update on April 1, 2026.

## Current Main Paper

1. Main quality table.

- Benchmarks: `AI_Idea_Bench_2025`, `LiveIdeaBench`
- Methods: `direct`, `self-refine`, `scipip-proxy`, `ai-researcher-proxy`,
  `ours-delayed-consensus`
- Metrics: `Overall`, `Benchmark Alignment`, `Expert Quality`
- Principle: same benchmark-mode input packet, same structured output contract,
  same `qwen3-8b` backbone, separate cost reporting

2. One subsection each for:

- `Benchmarks`
- `Baselines`
- `Metrics`
- `Experimental Setup`
- `Main Results`
- `Process Analysis and Failure Analysis`
- `Cost and Current Scope`

3. Main experimental message.

- `ours-delayed-consensus` is best on both benchmarks and all primary quality
  columns in the current eight-target small batch.
- The paper should frame this as a coordination gain, not an efficiency gain.
- Graph-process and cost tables stay out of the main comparison table.

## Appendix

1. Graph-process table for `ours` only.

- Current file: `generated/cross_benchmark_graph_process_table.tex`
- Content: per-instance `Graph`, `Support`, `UCR`, `Utility`, `Rounds`,
  `Errors`, `Stop`

2. Cost table across all methods.

- Current file: `generated/cross_benchmark_cost_table.tex`
- Content: `Calls`, `Tokens`, and `x Direct`

3. Hard-case safeguard table.

- Current file: `generated/cross_benchmark_hard_case_table.tex`
- Case: `AI-18`
- Content: `Stop`, `Rounds`, `Actions`, `Errors`, `Overall`, `Graph`

4. Round-wise trajectory figure.

- Current file: `figures/hard_case_round_trajectory.pdf`
- Panels: `Support`, `UCR`, `Utility`

## Current Evidence Summary

1. Cross-benchmark main results are now based on:

- `outputs/quality_batches/20260401-merged-cross-benchmark-small-batch/batch_summary.json`

2. The current small-batch paper numbers are:

- Overall mean: `ours=6.508`, strongest non-graph baseline `=3.737`
- AI Idea Bench 2025: `ours=6.485`
- LiveIdeaBench: `ours=6.530`

3. Process takeaways for `ours`.

- Mean graph-process score: `8.915`
- Mean final support coverage: `0.953`
- Mean unresolved contradiction ratio: `0.000`
- Mean executed rounds: `3.25`
- All eight selected runs reached maturity

## Next Experimental Steps

1. Run a broader cross-benchmark slice with the current prompt policy,
   completeness safeguard, and duplicate-action salvage logic frozen.
2. Add benchmark-native scoring where the benchmark interface is stable enough
   to avoid noisy judge costs.
3. Add a small blind human-evaluation subset with the same output schema across
   all baselines.
4. Run ablations on:

- delayed consensus vs early synthesis
- maturity stop on/off
- completeness safeguard on/off
- literature grounding strength on/off
- fallback salvage on/off

5. Add a dedicated cost-quality tradeoff plot after the broader batch is stable.
