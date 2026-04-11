# EIG Robustness Cleanup Design

**Date:** 2026-04-11  
**Scope:** Narrow pre-`R009` revision of `ours-eig` before the larger `AI_Idea_Bench_2025` batch

## Goal

Improve `ours-eig` on benchmark-native `AI_Idea_Bench_2025` scoring without
changing the benchmark protocol, without introducing hidden-target leakage, and
without replacing the current transparent controller with a learned model.

The specific target is to close the remaining native gap on harder smoke cases
such as `AIIB-3883`, where the graph itself looks mature but the final proposal
is still less benchmark-faithful than the stronger single-agent baseline.

## Why This Revision Is Needed

The current smoke-stage evidence shows a mismatch between internal graph quality
and benchmark-native outcome quality.

- On the `R009A` smoke subset, `ours-eig` is clearly strongest under the local
  benchmark-facing metrics.
- But on the targeted native smoke check, `self-refine` is still slightly
  stronger on average.
- The failure is narrow rather than global:
  `ours-eig` already builds coherent graphs, but the selected mature subgraph
  and final synthesis do not always convert that graph state into a
  benchmark-native-friendly proposal.

Two concrete issues were identified during diagnosis:

1. The current safe grounding path can extract noisy dataset or metric items
   from visible reference snippets.
2. The current postprocess path can rebuild grounding from full internal
   metadata when no stored safe grounding is present, which risks benchmark
   unfairness in benchmark mode.

This means the next revision must improve both quality and protocol fidelity.

## Agreed Design Choice

Keep the controller transparent and heuristic.

We will **not** introduce a learned utility model or a learned maturity model
 at this stage. The benchmark-native supervision is too sparse, the data is too
small for a stable learned controller, and a learned value model would weaken
the paper's current story by adding complexity before the transparent graph
controller is strong enough.

Instead, we will strengthen four modules:

1. benchmark-safe grounding
2. benchmark-aware utility estimation
3. mature-subgraph selection and maturity gating
4. slot-wise final synthesis

## Design Principles

### 1. Benchmark-Safe Means No Hidden Target During Generation

In benchmark mode, generation-time and synthesis-time prompts must only use:

- benchmark topic
- benchmark reference packet
- prompt-safe paper snippets derived from visible inspiration papers

They must not rely on:

- `target_paper`
- hidden gold motivation
- hidden gold method summary
- target-paper-derived dataset or metric metadata

If the pipeline cannot obtain a clean dataset or metric anchor from visible
inputs, it should remain cautious rather than silently recovering hidden
benchmark metadata.

### 2. Utility Should Reward Benchmark-Facing Scientific Structure

The current utility estimator mostly measures graph-internal structure:

- promise
- support
- coherence
- evidence
- novelty
- contradiction reduction

That remains useful, but it is not enough.

The revised utility should additionally reward:

- benchmark/topic fidelity
- mechanism specificity
- experiment-to-method alignment
- role-balanced claim-chain coverage

and penalize:

- copied-reference collapse
- generic evaluation plans
- mature-looking but synthesis-unready chains

### 3. Mature Subgraphs Should Be Paper-Ready, Not Only Graph-Complete

The current claim chain can mark a graph as synthesis-ready when the chosen
slots are technically complete but still too generic.

The revised mature-subgraph logic should prefer:

- a concrete `Method` node over a broad `Hypothesis` when both are available
- an evaluation node that names visible evaluation assets, metrics, baselines,
  or ablations
- a gap node that reflects a literature-grounded limitation, not only a vague
  evidence need
- a caveat or repair node that is actually relevant to the selected mechanism

It should also be role-aware:

- `TaskFramer` should contribute the selected problem framing
- `LiteratureGrounder` should contribute the selected gap or overlap boundary
- `MethodArchitect` should contribute the selected mechanism
- `ExperimentDesigner` should contribute the selected evaluation plan
- `SkepticRepairer` should contribute the selected caveat or repair

### 4. Final Synthesis Should Be Slot-Wise Before It Is Prose-Wise

The current synthesis path asks the model to directly turn the selected
subgraph into final prose. This works on easy cases, but on harder cases the
evaluation section becomes stitched or mechanically appended.

The revised synthesis should be two-stage in spirit:

1. assemble a structured slot summary from the mature chain
2. turn that structured summary into compact proposal prose

This keeps the roles meaningful in the final output and reduces the chance that
good graph structure is lost during final writing.

## Alternative Paths Considered

### Option A: Prompt-Only Cleanup

Pros:

- lowest engineering cost
- quickest rerun

Cons:

- leaves the controller misalignment unresolved
- risks overfitting to one or two smoke cases

Decision:

- rejected as too weak

### Option B: Learned Utility Or Maturity Models

Pros:

- potentially stronger long-term controller

Cons:

- too little reliable supervision right now
- higher protocol and leakage risk
- complicates the paper story before the transparent controller is validated

Decision:

- rejected for the current revision stage

### Option C: Transparent Module-First Cleanup

Pros:

- directly addresses the observed native-gap failure mode
- keeps the method interpretable and paper-friendly
- improves fairness and robustness together

Cons:

- requires coordinated changes across several modules

Decision:

- chosen

## Module-Level Revision Plan

### A. Fair Grounding

- make benchmark-mode prompt and synthesis grounding always use safe metadata
- make postprocess grounding benchmark-safe by construction
- improve safe extraction from visible snippets so noisy OCR fragments do not
  become fake datasets or fake metrics

### B. Utility Revision

- add benchmark-facing utility terms
- add penalties for copied-method collapse and generic experiment plans
- keep the estimator transparent, scalar, and inspectable in saved summaries

### C. Mature-Subgraph Revision

- add specificity-aware and role-aware slot selection
- require better evaluation-node quality before maturity
- improve the connection between maturity and actual synthesis readiness

### D. Slot-Wise Synthesis Revision

- derive a structured slot payload from the selected claim chain
- synthesize the proposal around those slots instead of around a loose node bag
- make the evaluation paragraph one coherent benchmark-facing plan

## Robustness Validation Plan

We should not judge the revision on only one sample.

The immediate diagnosis subset after the cleanup should include:

- the hard failure case `3883`
- successful smoke cases `13` and `7909`
- mixed case `9849`
- plus a small number of additional AIIB cases if cost allows

The revision passes this stage only if:

- benchmark-safe grounding remains intact
- no new generic-collapse pattern appears
- `ours-eig` improves or at least stabilizes native AIIB performance on the
  hard cases
- the revised utility and maturity signals correlate better with benchmark
  outcome quality than before

## Out Of Scope For This Revision

- learned controller models
- main-table baseline redesign
- full `R009` 24-case launch
- new benchmark integrations
- paper text updates beyond experiment-log tracking

## Success Criteria

This cleanup is successful if all of the following hold:

1. benchmark-mode fairness is tightened, not loosened
2. `ours-eig` proposals on hard AIIB cases become less stitched and more
   benchmark-native-friendly
3. the transparent utility and maturity signals better reflect proposal quality
4. the next small AIIB diagnosis packet justifies launching the full `R009`
   slice
