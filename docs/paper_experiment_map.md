# Paper Experiment Map

This note is the compact paper-facing map for the current experiment story.
The detailed execution roadmap lives in `docs/paper_experiment_plan.md`, and
the concrete run queue lives in `docs/paper_experiment_tracker.md`.

## Central Paper Claim

The paper should argue that scientific ideation improves when the evolving idea
state is externalized as a typed editable graph and refined through
utility-guided multi-agent edits, rather than being compressed into one draft
too early.

## Canonical Method Surface

- Main method:
  - `ours-eig`
- Backward-compatible alias only:
  - `ours-delayed-consensus`
- Current prototype components:
  - role-specialized graph edits
  - utility-guided candidate ranking
  - maturity-aware stopping
  - mature-subgraph-conditioned final synthesis

## Paper-Facing Benchmarks

- Primary benchmark:
  - `AI_Idea_Bench_2025`
- Secondary benchmark:
  - `LiveIdeaBench`

## Paper-Facing Baselines

- Main comparison set:
  - `direct`
  - `self-refine`
  - `ai-researcher`
  - `ours-eig`
- Conditional addition:
  - `virsci`, only if exact benchmark-faithful integration becomes runnable
- Supporting exact baseline:
  - `scipip`, if the official path remains stable
- Development-only baselines:
  - any `*-proxy` baseline

## Main Evidence Blocks

1. Benchmark-native automatic comparison on `AI_Idea_Bench_2025`.
2. Benchmark-native automatic comparison on `LiveIdeaBench`.
3. Shared human blind review on a balanced subset.
4. EIG ablations isolating utility ranking, maturity stop, and synthesis.
5. Appendix-level process, reliability, and cost analysis.

## Decision Gates

- Gate 1:
  - confirm exact `ai-researcher` benchmark-mode execution before any larger
    batch
- Gate 2:
  - decide whether `virsci` is truly benchmark-faithful; if not, remove it
    from the headline table rather than substituting a proxy
- Gate 3:
  - expand beyond the smoke slice only if benchmark-native scoring runs cleanly
    on both benchmarks
- Gate 4:
  - run human review only after the final proposal format is stable and
    non-generic

## Supplementary Metrics To Preserve

- `Evidence coverage`
- `Contradiction resolution`
- `Claim-chain completeness`
- `Rounds to maturity`
- `Action diversity`
- fallback rate
- controller override rate
- API calls, tokens, and wall-clock runtime

## Current Planning Priority

`M0` is complete and the current reference small-`M1` packet is:

- `outputs/quality_batches/20260411-000159-refreshed-m1-mini-synthesis-cleanup-v2-native`

The next paper-critical decision is whether to spend one more narrow refinement
pass on weak-context `LiveIdeaBench` meteorology stability before larger runs.

If we keep the current packet as the frozen small-`M1` reference, the next run
order should be:

1. optional weak-context stabilization for `ours-eig` on meteorology-like
   `LiveIdeaBench` rows
2. `M2` benchmark-native core automatic slice on `AI_Idea_Bench_2025`
3. `M2` benchmark-native core automatic slice on `LiveIdeaBench`
4. EIG ablations after the larger automatic results are stable
5. shared human blind review after proposal formatting is stable
