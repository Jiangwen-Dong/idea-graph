# Paper-Faithful Baseline Hardening Design

## Goal

Harden the paper-evaluation baselines quickly enough to support the main table
without spending excessive time on exact upstream reproduction. The paper claim
should be defensible: baselines are benchmark-faithful reproductions or are
clearly excluded from the headline comparison.

## Positioning

We should not claim that every baseline is bit-for-bit identical to upstream
systems. The paper-safe claim is:

> We implemented benchmark-faithful reproductions of prior systems under a
> unified evaluation interface, preserving each method's core stages and
> prompting logic while adding thin adapters for benchmark packet ingestion and
> output normalization.

This wording supports fast progress while avoiding an overclaim that reviewers
could challenge.

## Baseline Categories

### Exact Upstream

The upstream repository runs its own scripts and method logic. Our code only
constructs benchmark-safe inputs, launches the upstream commands, and normalizes
the output proposal.

### Paper-Faithful Adapter

The method preserves the original system's core stages and prompting logic, but
requires a thin local adapter for benchmark-mode input/output. This is
acceptable for the main table if the adapter is documented and does not become a
new method.

### Proxy Or Diagnostic

The method is implemented primarily inside this repository as an approximation.
These baselines can be used for debugging and appendix diagnostics, but not as
headline paper baselines unless explicitly labeled as proxies.

## Main-Table Eligibility Gate

A baseline can enter the main paper table only if it passes all checks below.

- It consumes the same benchmark-facing input packet as EIG.
- It does not see hidden target-paper fields, gold motivation, gold method
  summary, scored LiveIdeaBench idea text, or other oracle fields.
- It produces the shared structured output schema.
- Its run artifacts include raw command/config metadata and normalized output.
- It completes a fixed smoke packet on AI Idea Bench 2025 and LiveIdeaBench.
- It does not silently fall back to a `*-proxy` implementation.
- Its preserved method stages are documented in a reproduction matrix.

## Baseline-Specific Design

### Direct

`direct` is a controlled lower-bound local baseline. It is not an external
reproduction, but it is valid for the main table because it has a simple,
transparent, benchmark-shared one-pass generation protocol.

### Self-Refine

`self-refine` is a controlled local iterative baseline. It is valid for the main
table because it isolates single-agent critique and revision under the same
benchmark I/O contract.

### AI-Researcher

Preferred path:

- use the upstream-script execution path in `src/idea_graph/external_baselines.py`
- preserve seed idea generation, proposal expansion, and tournament ranking
- write benchmark-safe paper cache inputs from the shared packet
- normalize the selected upstream proposal into the shared output schema

Fallback path:

- use the existing OpenAI-compatible bridge only if upstream execution is
  blocked by provider or setup issues
- label this as a paper-faithful bridge only if the run preserves the
  seed-expansion-ranking structure and does not call the local `ai-researcher-proxy`
  baseline as a hidden fallback

Decision:

- main table if upstream-script or paper-faithful bridge passes the smoke gate
- appendix-only if it remains a mostly local proxy

### SciPIP

Preferred path:

- use upstream `src/generator.py new-idea`
- feed benchmark-safe background text through the wrapper
- verify the configured retriever/assets are actually available
- parse generated ideas from upstream output without substituting local proxy
  generation

Decision:

- main table if real upstream generation completes on the smoke packet
- appendix-only if required retrieval/config assets are missing or degenerate

### VirSci

Preferred path:

- add the thinnest possible fixed-topic benchmark adapter if the upstream
  system can accept a seed topic/background without changing its collaboration
  algorithm
- preserve multi-agent discussion and team synthesis
- normalize the final team proposal into the shared schema

Decision:

- main table only if a fixed-topic benchmark run is possible without rewriting
  VirSci into a different method
- otherwise exclude from the headline table and document the no-go result

## Smoke Gates

### B0 Feasibility Smoke

Use one AI Idea Bench 2025 case and one LiveIdeaBench case. A baseline passes B0
if both runs finish, produce valid output schema, and record no oracle leakage.

### B1 Main-Table Smoke

Use four AI Idea Bench 2025 cases and four LiveIdeaBench cases from the frozen
paper-eval pool or a clearly marked pre-paper smoke slice. A baseline passes B1
if all runs finish or failures are rare and explainable, and normalized outputs
can be scored by the existing evaluation tools.

## Main-Table Plan

The target main table is:

- `direct`
- `self-refine`
- `ai-researcher`
- `scipip`
- `virsci`, only if B0 and B1 pass without method rewrite
- `ours-eig` with `parallel_graph_v2`

If time is tight, freeze the main table with:

- `direct`
- `self-refine`
- `ai-researcher`
- `scipip`, if B0 passes
- `ours-eig`

This keeps the paper defensible while avoiding a long VirSci integration delay.

## Non-Goals

- Do not spend days mirroring every upstream environment detail.
- Do not force a broken upstream repository into the main table.
- Do not silently replace a failed external baseline with a proxy.
- Do not tune EIG or the learned critic on paper-eval outputs while hardening
  baselines.

## Success Criteria

- A reproduction matrix records exact, paper-faithful, appendix-only, or exclude
  status for each baseline.
- B0 smoke commands are available and runnable for every attempted external
  baseline.
- B1 smoke commands are available for baselines that pass B0.
- The final main-table baseline set contains only controlled local baselines,
  exact upstream baselines, or paper-faithful adapters.
