# Baseline Reproduction Matrix

This matrix records which baselines are valid for headline paper evaluation
under the shared benchmark I/O contract.

## Eligibility Labels

- `controlled-local`: simple local baseline with transparent protocol
- `exact-upstream`: upstream repository scripts run the method logic
- `paper-faithful-adapter`: original stages are preserved with thin benchmark I/O adapters
- `appendix-only`: useful diagnostic but not headline evidence
- `exclude`: not benchmark-faithful enough for the main table

## Current Matrix

| Baseline | Target Label | Preserved Method Structure | Adapter Scope | Main-Table Gate |
| --- | --- | --- | --- | --- |
| `direct` | `controlled-local` | one-pass single-agent idea generation | shared benchmark packet and output schema | eligible |
| `self-refine` | `controlled-local` | draft, critique, revision | shared benchmark packet and output schema | eligible |
| `ai-researcher` | `paper-faithful-adapter` | seed generation, proposal expansion, candidate ranking | paper-cache construction, provider config, output normalization | B1 passed; include with adapter status disclosed |
| `scipip` | `paper-faithful-adapter` | upstream `generator.py new-idea` with retrieval/decomposition | benchmark background JSON, config path, output normalization | appendix-only until enabled and B1-passed |
| `virsci` | `exclude` until fixed-topic adapter passes | multi-agent team discussion and synthesis | fixed-topic packet injection if feasible | exclude; fixed-topic adapter unavailable |
| `ai-researcher-proxy` | `appendix-only` | local approximation of seed/expand/rank | implemented inside this repo | not headline |
| `scipip-proxy` | `appendix-only` | local structured decomposition approximation | implemented inside this repo | not headline |
| `virsci-proxy` | `appendix-only` | local discussion-style approximation | implemented inside this repo | not headline |

## Smoke Gate Results

- B0 core smoke passed on `AI_Idea_Bench_2025:13` and `liveideabench:0` for `direct`, `self-refine`, `ai-researcher`, and `ours-eig`.
- B1 core smoke passed on 8 total cases: `AI_Idea_Bench_2025:13,15,18,21` and `liveideabench:0,23,47,70`.
- B1 aggregate local scores: `ours-eig` overall `6.01`, `self-refine` `5.96`, `direct` `5.68`, `ai-researcher` `4.48`.
- B1 aggregate benchmark alignment: `ours-eig` `4.50`, `self-refine` `4.45`, `direct` `4.02`, `ai-researcher` `1.97`.
- `ai-researcher` is technically runnable under the paper-faithful OpenAI-compatible bridge, but its small-smoke quality is weaker than the local controls and should be interpreted carefully.
- `scipip` has a locally present upstream layout, but the current Qwen config disables it, so it is not headline-ready until explicitly enabled and smoke-tested.
- `virsci` remains excluded from headline evaluation because the upstream repository lacks a fixed-topic benchmark entrypoint.

## Paper Wording

Use:

> We implemented benchmark-faithful reproductions of prior baselines under a
> unified evaluation interface, preserving each method's core stages while
> adding thin adapters for benchmark packet ingestion and output normalization.

Avoid claiming bit-for-bit upstream reproduction unless the exact-upstream path
is actually used.
