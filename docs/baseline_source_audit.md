# Baseline Source Audit

Audit date: 2026-04-17

This document records the public source-code status of the baseline papers and
how each method should be treated in our fixed benchmark protocol. The main
distinction is source availability versus benchmark compatibility: most source
code exists, but not every repository exposes a clean fixed-topic entrypoint
that consumes our benchmark packet and emits our proposal schema without a thin
adapter.

## Summary

| Baseline | Paper / Method | Public source | Local clone status | Benchmark status | Recommended use |
| --- | --- | --- | --- | --- | --- |
| `direct` | Controlled one-pass generation baseline | Not an external paper baseline | Implemented locally | Native benchmark protocol | Headline lower-bound control |
| `self-refine` | Self-Refine: Iterative Refinement with Self-Feedback | `https://github.com/madaan/self-refine` | Not cloned; remote HEAD verified as `9a206d4` | Local paper-faithful reproduction of generate-feedback-refine loop | Headline controlled iterative baseline |
| `ai-researcher` | Can LLMs Generate Novel Research Ideas? / AI-Researcher | `https://github.com/NoviScl/AI-Researcher` | `.tmp-baselines/AI-Researcher`, HEAD `e5dd05a` | Paper-faithful adapter or exact upstream path depending on provider/config | Headline if adapter status is disclosed |
| `scipip` | SciPIP: An LLM-based Scientific Paper Idea Proposer | `https://github.com/cheerss/SciPIP` | `.tmp-baselines/SciPIP`, HEAD `a0a927a` | Upstream layout present; OpenAI-compatible bridge now implemented for benchmark packets | B0 passed; B1 pending |
| `virsci` | Many Heads Are Better Than One / VirSci | `https://github.com/open-sciencelab/Virtual-Scientists` | `.tmp-baselines/Virtual-Scientists`, HEAD `07097fd` | Exact upstream remains benchmark-incompatible, but a fixed-topic bridge now exists | B0 passed; B1 pending |
| `ai-scientist` | The AI Scientist | `https://github.com/SakanaAI/AI-Scientist` | Not cloned; remote HEAD verified as `1de1dbc` | Full autonomous experiment/paper loop, not same idea-only task | Future supplementary only |

## Baseline-Specific Notes

### Direct

`direct` is not claimed as a reproduced paper baseline. It is a transparent
one-pass controlled baseline under the same benchmark input and output schema as
our method. It should remain in the headline table as a lower-bound control.

### Self-Refine

The official Self-Refine repository is available. However, the repository is
organized around task-specific examples such as dialogue, code, sentiment, and
math rather than scientific idea generation packets. For our benchmark, the
faithful adaptation is to preserve the core method contract:

- generate an initial proposal;
- ask the same model family for structured feedback;
- revise using the feedback;
- stop after the configured refinement budget.

This is appropriate to report as a controlled local reproduction, not as an
exact upstream script run.

### AI-Researcher

The public AI-Researcher repository matches our intended baseline most closely:
it consumes a research topic, searches or consumes related papers, generates
grounded seed ideas, expands them into project proposals, ranks them, and can
filter for novelty/feasibility.

Our adapter preserves the central seed-expansion-ranking structure while adding:

- benchmark packet ingestion;
- reference-packet-to-paper-cache conversion;
- provider configuration;
- output normalization into our proposal schema.

The current Qwen-compatible bridge is paper-faithful rather than bit-for-bit
exact-upstream. If we later run the upstream scripts directly with their
expected provider and cache format, we can upgrade the status to
`exact-upstream`.

### SciPIP

The SciPIP repository is public and has a terminal entrypoint for idea
generation. Its faithful operation depends on a Neo4j literature database,
preloaded paper assets, model configuration, and optional embedding resources.
Because of this heavy retrieval stack, we implemented a speed-first
OpenAI-compatible bridge that preserves the problem-decomposition plus
idea-synthesis structure while consuming our fixed benchmark packet. The exact
upstream generator path remains available, but the bridge is the practical path
for current benchmark sweeps.

Current gate:

- the real config and database paths are enabled locally;
- a B0 smoke passes without proxy fallback;
- a B1 smoke passes on the same paper-eval slice as the other baselines.

Current status: bridge B0 passed on `AI_Idea_Bench_2025:13` and
`liveideabench:0`; B1 is still pending.

### VirSci

The VirSci source and data are public, but the upstream run is designed as a
science-of-science collaboration simulation using author/team data, paper
databases, embeddings, and an epoch-based platform. The repository does not
expose a simple fixed-topic entrypoint that cleanly maps one benchmark packet to
one generated proposal. To avoid blocking on that stack, we implemented a
benchmark fixed-topic bridge that preserves the paper spirit more honestly than
the old local proxy: multiple scientist turns discuss the same benchmark topic,
then a team-synthesis stage consolidates the final proposal.

For paper safety, do not describe this as exact-upstream VirSci. It is a
paper-faithful fixed-topic adapter. The exact upstream stack is still not a fair
headline baseline for our benchmark unless its native assets and entrypoints are
used directly.

Current status: bridge B0 passed on `AI_Idea_Bench_2025:13` and
`liveideabench:0`; B1 is still pending.

### AI Scientist

The AI Scientist code is public and paper-faithful reproduction would require a
different evaluation contract: it generates ideas, writes code, runs
experiments, creates figures, writes full papers, and reviews them. This is
substantially broader than our current idea-generation benchmark and should not
be mixed into the headline table unless we define a separate supplementary
experiment.

## If A Future Baseline Has No Source

For a source-unavailable baseline, use paper-faithful reproduction only if all
of the following are true:

- the paper describes a clear algorithmic protocol;
- prompts, model family, iteration counts, retrieval sources, and ranking rules
can be reconstructed with low ambiguity;
- the reproduction is labeled as `paper-faithful-reimplementation`;
- the paper text and our implementation choices are cited in an appendix table;
- the baseline is not used to overclaim exact reproduction.

Current conclusion: no immediate paper-only reconstruction is necessary for the
core baselines. The main engineering work is validating adapters and deciding
which source-backed baselines are fair for the headline table.
