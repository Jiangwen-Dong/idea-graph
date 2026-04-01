# Baseline Integration Plan

This note records the next implementation steps for literature baselines so the
repo evolves toward a paper-valid comparison protocol.

## Current Status

- `ours-delayed-consensus` is runnable inside this repository.
- `direct` and `self-refine` are runnable inside this repository.
- `ai-researcher-proxy`, `scipip-proxy`, and `virsci-proxy` exist as local
  approximations.
- exact external wrapper entrypoints now exist for `ai-researcher`, `scipip`,
  and `virsci` through `configs/external_baselines.example.json`.
- `ResearchAgent` is banned from the current paper protocol.
- the proxy baselines are now being treated as behaviorally differentiated
  repo-native baselines rather than interchangeable prompt aliases:
  - `ai-researcher-proxy`: lightweight seed ideation, proposal expansion, and
    candidate ranking
  - `scipip-proxy`: low-cost structured single-pass decomposition baseline
  - `virsci-proxy`: discussion-oriented multi-agent proxy on top of the shared
    delayed-consensus engine

## Immediate Goal

Keep the local baselines reasonable, benchmark-faithful, and cost-aware now,
while upgrading to exact external integrations only when the upstream
repositories make this feasible and the comparison benefit is worth the setup
cost.

Practical note:

- for Aliyun DashScope or Qwen-based runs, the recommended near-term baseline is
  the local `ai-researcher-proxy`
- it now follows a lightweight three-stage pipeline:
  seed ideation, proposal expansion, and candidate ranking
- this is intentionally a coarse reproduction of the paper's core idea rather
  than a strict upstream reimplementation
- for low-cost comparison sweeps, use:
  - `direct`
  - `self-refine`
  - `scipip-proxy`
  - `ai-researcher-proxy`
- use `virsci-proxy` only for smaller validation subsets because it shares the
  multi-agent runtime cost profile of the main method

## Priority Order

1. `AI-Researcher`

- reason:
  - strongest ideation baseline match
  - official public repository is available
  - its topic plus paper-cache pipeline maps relatively cleanly to our
    benchmark-mode packet

2. `SciPIP`

- reason:
  - clearly relevant ideation baseline
  - public repository is available
  - input/output can be bridged through its background JSON interface
- caveat:
  - requires heavier environment preparation such as Neo4j and literature
    database assets

3. `VirSci`

- reason:
  - important multi-agent comparison baseline
  - public repository is available
- caveat:
  - requires large external assets and a more complex runtime environment
  - integration should be a guarded external wrapper rather than a silent local
    approximation

## Implementation Decisions

### Exact-vs-Proxy Naming

- exact external wrappers should use names without `-proxy`
  - `ai-researcher`
  - `scipip`
  - `virsci`
- local approximations should retain `-proxy`
  - `ai-researcher-proxy`
  - `scipip-proxy`
  - `virsci-proxy`

### Input Contract

All external baselines should consume the same benchmark-facing packet as our
main method.

- `AI_Idea_Bench_2025`
  - topic
  - reference titles and snippets only
  - no target-paper oracle fields during generation
- `LiveIdeaBench`
  - keyword only
  - no held-out benchmark idea text during generation

### Output Contract

All baselines should be mapped back into the same internal proposal schema:

- `title`
- `problem`
- `existing_methods`
- `motivation`
- `hypothesis`
- `method`
- `evaluation`
- `significance`
- `caveats`

## Planned Engineering Steps

1. Add an external-baseline configuration file format.
2. Add a guarded adapter layer for external repositories.
3. Implement `AI-Researcher` through its official scripts:
   - synthesize paper cache from benchmark packet
   - run grounded idea generation
   - run proposal expansion
   - run ranking
   - parse the best proposal back into our schema
4. Implement `SciPIP` wrapper:
   - synthesize background JSON
   - run `generator.py new-idea`
   - parse best generated idea
5. Implement `VirSci` wrapper:
   - launch external run
   - parse newest team info JSON
   - map idea or abstract back into our schema
6. Add benchmark-batch scripts after baseline wrappers stabilize.

## After Baseline Integration

The next research-facing steps should be:

1. run a small matched-budget sanity set over `direct`, `self-refine`,
   `scipip-proxy`, `ai-researcher-proxy`, and `ours-delayed-consensus`
2. freeze the baseline prompts and budgets before broad benchmark sweeps
3. add a batch benchmark runner for corpus-level comparison and `IC`
4. generate pilot paper tables over both public benchmarks
5. only then, do deeper prompt or collaboration-policy tuning
