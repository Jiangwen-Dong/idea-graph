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
| `scipip` | `paper-faithful-adapter` | structured problem decomposition and idea synthesis grounded in the benchmark packet | upstream generator path or OpenAI-compatible bridge, output normalization | B1 bridge passed; headline candidate with adapter disclosure |
| `virsci` | `paper-faithful-adapter` | multi-agent discussion and team synthesis around a fixed benchmark topic | benchmark fixed-topic bridge, discussion transcript capture, output normalization | B1 bridge passed; appendix-only for now |
| `ai-researcher-proxy` | `appendix-only` | local approximation of seed/expand/rank | implemented inside this repo | not headline |
| `scipip-proxy` | `appendix-only` | local structured decomposition approximation | implemented inside this repo | not headline |
| `virsci-proxy` | `appendix-only` | local discussion-style approximation | implemented inside this repo | not headline |

## Smoke Gate Results

- B0 core smoke passed on `AI_Idea_Bench_2025:13` and `liveideabench:0` for `direct`, `self-refine`, `ai-researcher`, and `ours-eig`.
- B0 bridge smoke passed on `AI_Idea_Bench_2025:13` and `liveideabench:0` for `scipip` and `virsci`.
- B1 core smoke passed on 8 total cases: `AI_Idea_Bench_2025:13,15,18,21` and `liveideabench:0,23,47,70`.
- B1 aggregate local scores: `ours-eig` overall `6.01`, `self-refine` `5.96`, `direct` `5.68`, `ai-researcher` `4.48`.
- B1 aggregate benchmark alignment: `ours-eig` `4.50`, `self-refine` `4.45`, `direct` `4.02`, `ai-researcher` `1.97`.
- `ai-researcher` is technically runnable under the paper-faithful OpenAI-compatible bridge, but its small-smoke quality is weaker than the local controls and should be interpreted carefully.
- `scipip` passed the 8-case B1 bridge smoke with mean overall `4.63`, mean alignment `2.25`, about `2` calls, and about `3.3k` tokens per case. It now qualifies as a headline candidate if we disclose adapter status.
- `virsci` passed the 8-case B1 bridge smoke with mean overall `4.63`, mean alignment `2.19`, about `4` calls, and about `6.2k` tokens per case. It is currently better treated as appendix-only because the fixed-topic bridge is the heaviest adaptation and the quality gain is modest.

## Paper Wording

Use:

> We implemented benchmark-faithful reproductions of prior baselines under a
> unified evaluation interface, preserving each method's core stages while
> adding thin adapters for benchmark packet ingestion and output normalization.

Avoid claiming bit-for-bit upstream reproduction unless the exact-upstream path
is actually used.
