# Scientific Ideation Landscape

This note summarizes the current scientific-ideation landscape and converts it
into a paper-facing recommendation for this repository.

## Positioning

Our work targets `scientific ideation`: generating one structured, benchmark-
faithful research idea from a constrained input packet. This is narrower than
end-to-end autonomous-research systems and should be evaluated as such.

## Reference Surveys

- [Awesome-Agent-Scientists](https://github.com/AgenticScience/Awesome-Agent-Scientists)
- [Awesome-LLM-Scientific-Discovery](https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery)
- [From AI for Science to Agentic Science: A Survey on Autonomous Scientific Discovery](https://arxiv.org/abs/2508.14111)

## Recommended Benchmarks

### Primary Benchmark

- [AI Idea Bench 2025](https://ai-idea-bench.github.io/)

Why:

- best public fit for literature-grounded scientific ideation
- explicit inspiration-to-target construction
- benchmark-native automatic evaluation
- large scale

### Secondary Benchmark

- [LiveIdeaBench](https://liveideabench.com/)

Why:

- broader scientific-domain coverage
- minimal-context robustness test
- complements the literature-grounded setting

### Useful But Non-Core References

- [IdeaBench](https://arxiv.org/abs/2411.02429)
- [ResearchBench](https://arxiv.org/abs/2503.21248)
- [ScienceAgentBench](https://openreview.net/pdf?id=6z4YKr0GK6)
- [MLR-Bench](https://arxiv.org/abs/2505.19955)

These are useful for context, but they are not the strongest main-benchmark
choices for a paper focused specifically on scientific ideation.

## Recommended Baselines

### Main Table

- `Direct`
  one-pass single-agent baseline
- `Self-Refine`
  iterative single-agent critique and revision
- [AI-Researcher](https://github.com/NoviScl/AI-Researcher)
  strongest ideation-specific literature baseline for our setting
- [VirSci / Virtual-Scientists](https://arxiv.org/abs/2410.09403)
  closest multi-agent ideation baseline
- `Ours (EIG)`
  main method

### Optional Appendix Baselines

- [SciPIP](https://arxiv.org/abs/2410.23166)
- [Nova](https://arxiv.org/abs/2410.14255)

### Development-Only Proxies

- `ai-researcher-proxy`
- `scipip-proxy`
- `virsci-proxy`

These are useful for internal sweeps, but they must be labeled as proxies or
local approximations when reported.

## Recommended Evaluation Stack

### Layer 1: Benchmark-Native Automatic Metrics

Use benchmark-native metrics rather than forcing one artificial automatic score
across all datasets.

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

### Layer 2: Shared Human Blind Review

Use the same rubric on both benchmarks:

- `Novelty`
- `Significance`
- `Feasibility`
- `Clarity`
- `Context Adherence`
- `Overall`

### Layer 3: Supplementary Process And Cost Analysis

- `Evidence coverage`
- `Contradiction resolution`
- `Claim-chain completeness`
- `Rounds to maturity`
- `Action diversity`
- `API and token cost`
- `Wall-clock runtime`

## Evaluation Principle

The local deterministic evaluator in this repository is valuable for iteration,
but it should remain a development tool.

Paper-facing claims should rely primarily on:

- benchmark-native automatic metrics
- blind human review
- supplementary process and cost analyses

## Relevant Prior Systems

### Scientific Ideation Systems

- [Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers](https://proceedings.iclr.cc/paper_files/paper/2025/file/ea94957d81b1c1caf87ef5319fa6b467-Paper-Conference.pdf)
- [AI-Researcher](https://github.com/NoviScl/AI-Researcher)
- [SciPIP](https://arxiv.org/abs/2410.23166)
- [Scideator](https://arxiv.org/abs/2409.14634)
- [Nova](https://arxiv.org/abs/2410.14255)

### Multi-Agent Scientific Collaboration

- [VirSci / Virtual-Scientists](https://arxiv.org/abs/2410.09403)
- [AutoGen](https://arxiv.org/abs/2308.08155)
- [Graph of Thoughts](https://arxiv.org/abs/2308.09687)

### Broader Autonomous Research Systems

- [AI-Researcher: Autonomous Scientific Innovation](https://openreview.net/pdf/a1c63cdd0495de94664b1513f7d95a3aedcb483a.pdf)
- [The AI Scientist](https://arxiv.org/abs/2408.06292)

These broader systems are important context, but they should not replace
ideation-specific baselines in the main table.

## Practical Recommendation For This Repo

If the paper must stay focused and convincing, the cleanest setup is:

- benchmarks:
  `AI_Idea_Bench_2025` + `LiveIdeaBench`
- main baselines:
  `Direct`, `Self-Refine`, `AI-Researcher`, `VirSci`, `Ours (EIG)`
- optional appendix baselines:
  `SciPIP`, `Nova`
- main metrics:
  benchmark-native automatic metrics + human blind review
- supplementary analysis:
  graph process and cost

This framing gives the clearest story for a paper on scientific ideation rather
than on general autonomous science.
