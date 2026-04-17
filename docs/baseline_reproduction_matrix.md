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
| `ai-researcher` | `paper-faithful-adapter` | seed generation, proposal expansion, candidate ranking | paper-cache construction, provider config, output normalization | B0 then B1 |
| `scipip` | `paper-faithful-adapter` | upstream `generator.py new-idea` with retrieval/decomposition | benchmark background JSON, config path, output normalization | B0 then B1 |
| `virsci` | `exclude` until fixed-topic adapter passes | multi-agent team discussion and synthesis | fixed-topic packet injection if feasible | feasibility audit first |
| `ai-researcher-proxy` | `appendix-only` | local approximation of seed/expand/rank | implemented inside this repo | not headline |
| `scipip-proxy` | `appendix-only` | local structured decomposition approximation | implemented inside this repo | not headline |
| `virsci-proxy` | `appendix-only` | local discussion-style approximation | implemented inside this repo | not headline |

## Paper Wording

Use:

> We implemented benchmark-faithful reproductions of prior baselines under a
> unified evaluation interface, preserving each method's core stages while
> adding thin adapters for benchmark packet ingestion and output normalization.

Avoid claiming bit-for-bit upstream reproduction unless the exact-upstream path
is actually used.
