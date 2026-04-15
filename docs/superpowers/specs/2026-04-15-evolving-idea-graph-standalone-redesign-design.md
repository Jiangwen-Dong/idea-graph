# Evolving-Idea-Graph Standalone Redesign

**Date:** 2026-04-15  
**Scope:** define a new fully standalone repository, `evolving-idea-graph`,
for the next-generation parallel EIG system with a central graph critic,
global commit control, full benchmark runner, and proposal synthesis pipeline.

## Purpose

The current `idea-graph` repository has become the stable base for:

- the sequential delayed-consensus runtime
- the current heuristic and critic-assisted controller lines
- existing benchmark integrations
- archived and active experiment artifacts

The next redesign is large enough that it should not be built by incrementally
mutating the current codebase.

The new repository should:

- be named `evolving-idea-graph`
- have no code import or runtime dependency on `idea-graph`
- own a cleaner end-to-end architecture for the parallel controller line
- support full benchmark execution and final proposal generation
- remain simple enough to explain in the paper as one coherent system

This redesign is not a small patch. It is a new system.

## Hard Boundary

`evolving-idea-graph` must be fully standalone.

That means:

- no Python imports from `idea-graph`
- no git submodule to `idea-graph`
- no runtime path references into `idea-graph`
- no hidden read-through of `idea-graph/outputs/...`
- no direct reuse of old internal module names as a dependency layer

Allowed reuse:

- exported artifacts in neutral formats such as `json`, `jsonl`, `csv`, `pt`,
  or `md`
- manual rewriting of ideas and interfaces after reading the old code
- manual migration of selected benchmark packets or dataset manifests into the
  new repo

The new repository may be informed by the current one, but it must not depend
on it.

## Approaches Considered

### Option A: Refactor The Current Repo In Place

Pros:

- fastest initial code change path
- immediate reuse of all current utilities

Cons:

- high risk of architecture drift and legacy coupling
- hard to explain the method boundary in the paper
- difficult to keep the old sequential line stable while redesigning the new
  parallel controller

Decision:

- reject

### Option B: Thin New Repo With Runtime Dependency On The Old Repo

Pros:

- lower initial rewrite cost
- easy access to existing datasets and helpers

Cons:

- the new system would still inherit hidden assumptions from the old runtime
- versioning and reproducibility would become confusing
- the user explicitly wants no connection or code reference to the old repo

Decision:

- reject

### Option C: Recommended Standalone Repo With Artifact Export Boundary

Pros:

- clean architecture and paper story
- no hidden coupling to old runtime internals
- safe place for large protocol redesign
- easier to keep the old repo stable while the new one evolves rapidly

Cons:

- requires an upfront migration boundary
- some stable logic must be rewritten or reowned

Decision:

- choose this option

## Repository Goal

`evolving-idea-graph` should implement one coherent method:

> Role-specialized parallel scientific graph evolution with centralized
> relation-aware graph-critic control and learned global commit decisions.

This repo should contain:

- graph schema and runtime
- multi-role proposal generation
- role activation and `skip` handling
- graph critic edit head
- graph critic global commit head
- deterministic merge/application logic
- final claim-chain selection and proposal synthesis
- end-to-end benchmark runner
- training, calibration, and offline evaluation utilities
- reproducible logs and experiment artifacts

## Non-Goals

The first version of `evolving-idea-graph` should not include:

- all legacy baselines from `idea-graph`
- generic wrapper support for unrelated historical pipelines
- online continual adaptation as a required main-method component
- large asynchronous free-running multi-agent infrastructure

Those can be added later only if the core parallel graph-controller line is
already strong and stable.

## High-Level Architecture

The system should be organized around five clean layers.

### 1. Benchmark And Artifact Layer

Responsibilities:

- load benchmark instances
- load frozen benchmark packets
- load migrated legacy artifacts
- write standardized run outputs

Key principle:

- benchmark inputs are plain data, not implicit repo-specific code

### 2. Graph State Layer

Responsibilities:

- typed node and edge schema
- branch / role ownership metadata
- graph snapshots
- action application and validation
- maturity and utility summaries

Key principle:

- the graph is the central state object for all later modules

### 3. Role Proposal Layer

Responsibilities:

- maintain role definitions
- build role-scoped graph views
- activate only needed roles for a round
- query active role agents in parallel
- produce multiple candidate actions plus `skip`

Key principle:

- role agents propose; they do not directly mutate the graph

### 4. Controller Layer

Responsibilities:

- score role-specific action candidates with a graph critic edit head
- score global stop decisions with a commit head
- apply safe threshold and calibration logic
- select one action per active role at most
- coordinate the commit pre-check and post-check

Key principle:

- the controller judges and arbitrates; it does not generate free-form actions

### 5. Proposal Synthesis Layer

Responsibilities:

- select mature or near-mature claim chains
- produce the final structured proposal
- generate benchmark-evaluable sections
- log provenance for analysis

Key principle:

- final synthesis is downstream of the graph, not a replacement for it

## Core Runtime Protocol

The new runtime should use a synchronized parallel round protocol.

At round `t`:

1. Freeze graph snapshot `G_t`.
2. Run the global commit pre-check on `G_t`.
3. If commit fires, stop and synthesize the final proposal.
4. Run a cheap role-activation gate on `G_t`.
5. Query all active role agents in parallel on the same snapshot.
6. Each active role returns:
   - `K` role-appropriate edit candidates
   - one `skip` candidate
7. Validate and deduplicate candidates.
8. Score all role candidates with the graph critic edit head.
9. Select at most one action per active role.
10. Apply selected actions in a deterministic role order.
11. Recompute graph utility and maturity.
12. Run the global commit post-check.
13. If commit fires, stop; otherwise continue.

This protocol keeps:

- proposal generation parallel
- graph mutation controlled
- commit global rather than role-local

## Graph Critic Function

The graph critic should be split into two learned heads over a shared graph
representation.

### Edit Head

Input:

- graph snapshot `G_t`
- role id `r`
- candidate action `a`

Output:

- score for materializing action `a` for role `r`

Function:

- decide which role should do which action now
- arbitrate among multiple role proposals
- prefer meaningful edits over low-value repetition
- support `skip` as a valid decision outcome

### Commit Head

Input:

- graph snapshot `G_t`

Output:

- score or calibrated probability that the graph should stop now

Function:

- replace heuristic-only maturity gating as the main stopping decision
- prevent wasted late rounds
- stop early when the graph is already good enough

### Why Split The Heads

This separation is cleaner than treating `commit` as just another ordinary role
action.

Benefits:

- edit selection remains role-specific
- commit remains a global graph-level decision
- calibration is easier to reason about
- the paper story is cleaner

## Role Proposal Design

The first version should keep five roles.

- `MechanismProposer`
- `FeasibilityCritic`
- `NoveltyExaminer`
- `EvaluationDesigner`
- `ImpactReframer`

Reason:

- the current graph schema and existing evidence already support these roles
- the five-role decomposition maps well to scientific proposal dimensions
- parallel execution reduces the wall-clock penalty relative to the current
  sequential runtime

The first version should not increase beyond five roles.

### Candidate Policy Per Role

Each role should have a role-specific action family.

`MechanismProposer`:

- refine or add method/hypothesis structure
- add support or dependency edges for mechanism coherence
- repair mechanism ambiguity
- `skip`

`FeasibilityCritic`:

- add contradiction or risk structure
- request evidence for fragile assumptions
- propose repair for feasibility failures
- `skip`

`NoveltyExaminer`:

- mark overlap
- request novelty evidence
- refine novelty-claim boundaries
- `skip`

`EvaluationDesigner`:

- refine evaluation plans
- attach evidence about datasets, metrics, or baselines
- connect method to evaluation requirements
- `skip`

`ImpactReframer`:

- refine problem framing and significance
- connect problem, motivation, and hypothesis
- repair overbroad framing
- `skip`

The action vocabulary must remain role-aware.

The new runtime should not expose a uniform generic slate where every role sees
the same action family without filtering.

## Skip And Role Activation

The redesign should support two different mechanisms:

### Skip Candidate

Each active role returns one explicit `skip` candidate.

Purpose:

- avoid forcing a low-value edit
- reduce graph noise
- let the controller prefer no change when appropriate

### Role Activation Gate

Before API calls, the runtime should decide which roles are worth querying for
the current snapshot.

Purpose:

- reduce token cost
- avoid unnecessary parallel API calls
- keep later synthesis cleaner by reducing low-information actions

The activation gate can be heuristic in version one.

The first implementation should not require a learned role-activation module.

## Number Of Agents

Default recommendation:

- keep `5` roles for the main method

Secondary ablation later:

- compact `3`-role version

Suggested compact ablation:

- `Architect`
- `SkepticGrounder`
- `Experimenter`

This ablation can be useful later to study whether five roles are necessary,
but it should not replace the main five-role configuration in version one.

## Model Recommendation

Default first-version role proposal model:

- `qwen3-8b`

Reasons:

- it already matches much of the current collected runtime experience
- it is strong enough for structured JSON role proposals
- it is affordable enough for parallel role calls
- it avoids unnecessary distribution shift while the new controller is still
  being stabilized

Initial recommendation:

- use one common model family for all five roles
- keep role-specific prompt instructions rather than role-specific model
  changes

Optional later ablations:

- stronger proposal model such as `qwen-plus`
- stronger synthesis-only model

But the first clean version should keep:

- role proposal model fixed
- synthesis model fixed

to minimize confounds during controller validation.

## Standalone Data Strategy

The new repo must not read old artifacts by path from the current repo.

Instead, the old repo should produce a neutral export package.

### Export Boundary

The export package should include only neutral artifacts such as:

- benchmark packets
- instance manifests
- candidate-slate datasets
- partition manifests
- graph snapshots
- run-level summaries
- model checkpoints if still useful

The export package should be copied into the new repo or into an external
storage location owned by the new project.

The new repo should then import that package into its own internal data layout.

### Internal Ownership

After import, the new repo treats all data as its own working assets.

That means:

- no backward path references
- no dynamic queries into the old repo
- no hidden dependency on old file naming conventions after import

## End-To-End Benchmark Runner

The first version must include a full benchmark runner.

Required capabilities:

- load benchmark instances from the new repo's owned data format
- execute end-to-end role proposal, controller selection, graph evolution, and
  synthesis
- save graph snapshots and round logs
- save selected actions, skipped roles, and commit decisions
- evaluate final proposals with benchmark-native or benchmark-specific scoring
  wrappers
- write machine-readable manifests and human-readable summaries

The benchmark runner should be designed around frozen packet manifests, not
ad-hoc CLI arguments only.

## Proposal Synthesis Pipeline

The new repo must include the full final proposal generation stage.

Required behavior:

- select the final claim chain or mature subgraph from the evolved graph
- convert graph state into a structured proposal
- produce the benchmark-facing sections:
  - problem
  - existing methods
  - motivation
  - hypothesis
  - method
  - evaluation
  - significance
  - caveats

The synthesis stage should remain separate from the controller.

The graph critic should not directly write the final proposal text.

## Suggested Repository Layout

The new repository should start with a small but explicit package structure.

Suggested top-level layout:

- `README.md`
- `pyproject.toml`
- `configs/`
- `data/`
- `exports/`
- `outputs/`
- `scripts/`
- `src/evolving_idea_graph/`
- `tests/`
- `docs/`

Suggested Python package layout:

- `src/evolving_idea_graph/benchmarks/`
- `src/evolving_idea_graph/graph/`
- `src/evolving_idea_graph/roles/`
- `src/evolving_idea_graph/controller/`
- `src/evolving_idea_graph/runtime/`
- `src/evolving_idea_graph/synthesis/`
- `src/evolving_idea_graph/training/`
- `src/evolving_idea_graph/evaluation/`
- `src/evolving_idea_graph/io/`

This structure keeps the new repo understandable and separate from the old
single-runtime layout.

## Concrete Build Stages

The redesign should be executed in seven stages.

### Stage 0: Standalone Repo Bootstrapping

Goal:

- create the `evolving-idea-graph` repo skeleton
- define the standalone package structure
- define the data import / export contract

Deliverables:

- new repository scaffold
- README with architecture summary
- data contract document
- empty benchmark and output directories

### Stage 1: Core Graph And Benchmark Foundations

Goal:

- implement the new graph schema, action schema, and benchmark packet schema

Deliverables:

- typed graph dataclasses
- graph snapshot format
- action validation and application utilities
- benchmark instance loader
- run manifest writer

### Stage 2: Parallel Role Proposal Runtime

Goal:

- implement synchronized parallel rounds with role activation and `skip`
  support

Deliverables:

- role prompt builders
- role-scoped candidate generators
- async or thread-pooled parallel API execution
- one-action-per-role application protocol
- deterministic merge order

### Stage 3: Graph Critic V2

Goal:

- implement the shared graph encoder with edit head and global commit head

Deliverables:

- graph encoder
- per-role edit scorer
- global commit scorer
- offline training loop
- offline validation metrics

### Stage 4: Commit Calibration And Controller Policy

Goal:

- calibrate the commit head and define the safe controller policy

Deliverables:

- commit calibration reports
- threshold selection logic
- commit pre-check and post-check
- heuristic fallback only where necessary

### Stage 5: Proposal Synthesis Pipeline

Goal:

- generate final proposals from the evolved graph

Deliverables:

- claim-chain or mature-subgraph selector
- structured proposal writer
- synthesis logging and provenance

### Stage 6: Full Benchmark Runner

Goal:

- run end-to-end benchmark experiments with the new protocol

Deliverables:

- packet runner
- run manifests
- summary tables
- evaluation wrappers
- artifact bundle for paper figures and analysis

### Stage 7: Main Experiments And Ablations

Goal:

- validate the new repo as the main paper implementation

Required comparisons:

- sequential current-style EIG reference
- new parallel EIG without graph critic
- new parallel EIG with graph critic edit head only
- new parallel EIG with graph critic edit head plus commit head
- optional compact `3`-role ablation

## Risks

### Risk 1: The New Repo Repeats Old Complexity

Mitigation:

- keep module boundaries strict
- prohibit imports from the old repo
- start from minimal components only

### Risk 2: Parallelization Creates Noisy Or Conflicting Edits

Mitigation:

- one action per role maximum
- deterministic merge order
- strong candidate validation before application

### Risk 3: Commit Learning Is Unstable

Mitigation:

- keep commit as a separate global head
- calibrate on frozen development data
- retain conservative fallback rules

### Risk 4: The New Repo Becomes Too Large Too Fast

Mitigation:

- implement stages in order
- finish each stage with tests and artifact examples
- do not import all historical baselines

## Recommended Immediate Next Step

Write the implementation plan for **Stage 0 + Stage 1 + Stage 2** first.

Reason:

- these stages define the standalone repo boundary and the new runtime core
- later critic and synthesis work depend on these interfaces
- they are the minimum foundation for the rest of the redesign

Only after those plans are written and approved should implementation begin.
