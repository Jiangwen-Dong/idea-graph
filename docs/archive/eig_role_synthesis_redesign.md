# EIG Role And Synthesis Redesign

**Date**: 2026-04-10  
**Scope**: targeted diagnosis-driven revision for `ours-eig` only  
**Goal**: improve benchmark-faithful scientific ideation without redesigning the benchmark metrics or rewriting baseline methods around our role design

## Why This Revision

The refreshed `M1` result shows a stable pattern:

- `ours-eig` is strongest on local idea quality, benchmark alignment, and graph-process quality
- `self-refine` and `ai-researcher` remain slightly stronger on benchmark-native automatic scores
- the gap is not mainly about fluency alone; it is about proposal shape

The benchmark-native judges appear to reward four properties very consistently:

1. precise task fidelity
2. a concrete literature-grounded gap
3. one coherent mechanism rather than a stitched bundle
4. an evaluation plan that directly tests that mechanism

So the next revision should not change the benchmark metrics or tailor them to our internal roles. Instead, it should redesign `ours-eig` so that the evolving graph naturally constructs these four ingredients as part of real scientific collaboration.

## Design Principles

### Principle 1: Keep the benchmarks fixed

The benchmark-native metrics remain the evaluation target. We do not redesign them to fit our framework.

### Principle 2: Change `ours-eig`, not the baselines

The role redesign is part of our method contribution. Therefore:

- `ours-eig` will receive the functional-role redesign
- `direct`, `self-refine`, and `ai-researcher` should keep their own prompt families and baseline identities
- baseline updates should stay limited to generic hygiene, bug fixes, or faithful reproduction details, not adoption of our role structure

This keeps the comparison scientifically cleaner.

### Principle 3: Roles should map to real scientific functions

Each role should own one real research function rather than a vague editing style. That ownership should make the final proposal more benchmark-faithful for principled reasons.

### Principle 4: Synthesis should consume a validated claim chain

The final proposal should not be synthesized from a loose bag of connected nodes. It should be synthesized from one selected scientific chain that covers the minimum ingredients of a strong idea.

## Functional Role Redesign For `ours-eig`

The current roles are directionally useful, but they overlap too much. The new role contract is:

### `TaskFramer`

Owns:

- the exact benchmark task
- the concrete failure mode or bottleneck
- why the problem matters

Expected graph contribution:

- one problem node with clear task fidelity
- one motivation or significance node tied to the same task

This role replaces the broader current framing behavior with a narrower task-anchoring job.

### `LiteratureGrounder`

Owns:

- the visible reference-based gap
- overlap boundaries
- novelty positioning relative to the provided literature only

Expected graph contribution:

- one gap or novelty node grounded in the visible packet
- overlap markers or evidence needs when novelty is weak

This role should reduce generic novelty claims and improve benchmark-faithful literature grounding.

### `MethodArchitect`

Owns:

- the core mechanism
- the main design choices
- internal dependencies between components

Expected graph contribution:

- one central hypothesis or method node
- one small set of support or dependency nodes that explain how the mechanism works

This role must avoid multi-mechanism drift. One coherent mechanism is better than several loosely connected ones.

### `ExperimentDesigner`

Owns:

- the evaluation plan
- dataset or task anchors
- metrics
- baselines
- ablations or stress tests

Expected graph contribution:

- one evaluation node directly linked to the chosen mechanism
- explicit evaluation assets when the benchmark packet supports them

This role should make the experiment plan test the mechanism, not just the topic.

### `SkepticRepairer`

Owns:

- finding mismatches across nodes
- flagging unsupported leaps
- proposing repairs to restore coherence

Expected graph contribution:

- contradiction edges
- repair nodes
- risk or assumption nodes that sharpen the final idea

This role replaces pure criticism with actionable repair, which is more useful for graph evolution.

## Graph Coverage Contract

The graph should be considered synthesis-ready only when it contains one usable claim chain with all of the following:

1. one task/problem anchor
2. one literature-grounded gap or limitation
3. one core mechanism or hypothesis
4. one evaluation plan that tests that mechanism
5. one caveat, risk, or repair

This is not a new benchmark metric. It is an internal completeness contract for our method.

## Maturity Revision

The current maturity logic is helpful but still somewhat permissive. The revised maturity gate for `ours-eig` should require both:

- structural maturity:
  - sufficient support coverage
  - low unresolved contradiction ratio
  - stable utility
- functional coverage:
  - at least one complete claim chain satisfying the coverage contract above

This should reduce cases where the graph is coherent but still too generic.

## Synthesis Revision

### Current issue

Current synthesis can still overuse a broad connected subgraph. That allows generic coherence, but sometimes weakens benchmark faithfulness.

### Revised synthesis target

The synthesis stage should consume one selected claim chain:

- problem/task anchor
- literature gap anchor
- mechanism chain
- evaluation chain
- caveat or repair anchor

### Expected effect

This should improve:

- task fidelity
- method coherence
- motivation-to-experiment consistency
- clarity of the evaluation plan

### Constraint

Synthesis should still remain shorter than a paper and should not become a long related-work rewrite.

## Fairness Policy For Baselines

To keep the paper credible:

- `direct` remains a general single-pass baseline
- `self-refine` remains a general draft-critique-revise baseline
- `ai-researcher` remains an idea-generation baseline with its own seed, expansion, and ranking logic

We do **not** retrofit these baselines with our new functional-role decomposition unless an external baseline explicitly defines such a structure in its own method.

Allowed baseline updates:

- bug fixes
- JSON robustness
- cleanup of noisy copied snippet fragments
- benchmark-faithful prompt hygiene that does not import our role design

Disallowed baseline updates for this revision:

- task-gap-mechanism-evaluation role decomposition copied from `ours-eig`
- graph-style coverage contracts
- our maturity logic transplanted into baselines

## Implementation Order

1. revise the `ours-eig` role guidance and prompt contracts
2. add claim-chain coverage checks before maturity and synthesis
3. change final synthesis to consume a selected claim chain instead of a loose connected subgraph
4. keep baseline families stable except for generic hygiene
5. run one small regenerated packet before any larger new batch

## Expected Experimental Outcome

If this revision works, the next small packet should show:

- better benchmark-native fidelity for `ours-eig`
- stronger motivation-to-experiment consistency
- less mechanism stitching
- more specific evaluation plans
- no change in the identity of the baseline families

If it fails, the failure will be informative:

- either the graph is still not preserving the right proposal chain
- or the final synthesis is still washing out the graph structure
- or the benchmark-native advantage of simpler baselines is more fundamental than prompt shape

## Boundary Of This Revision

This revision is deliberately narrow.

It does **not** yet include:

- a learned action model
- external retrieval expansion in benchmark mode
- redesigned benchmark metrics
- a new baseline family

The purpose is to test whether a cleaner functional-role design and claim-chain synthesis make the graph method more benchmark-faithful while preserving its scientific identity.
