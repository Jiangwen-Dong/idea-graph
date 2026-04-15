# Evolving-Idea-Graph End-To-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully standalone `evolving-idea-graph` repository that supports parallel role proposal generation, centralized graph-critic control, global commit prediction, full benchmark execution, and final proposal synthesis without importing or depending on the current `idea-graph` codebase.

**Architecture:** The new repository is a clean-room implementation with a hard artifact boundary. Benchmark packets and replay artifacts may be imported only through neutral exported files. Runtime control is organized around a typed scientific graph, parallel role-specific candidate generation, a central controller with separate edit and commit heads, and a synthesis layer that turns the final graph into benchmark-evaluable proposals.

**Tech Stack:** Python 3.10+, `pydantic` or stdlib dataclasses, PyTorch, `transformers` or sentence-transformers, JSON/JSONL/CSV artifact IO, pytest, PowerShell, Markdown

---

## Scope And Ground Rules

All file paths below are relative to the new repository root after Task 1 creates
`..\evolving-idea-graph`.

Required constraints:

- no Python imports from `idea-graph`
- no runtime path reads from `idea-graph`
- no git submodule or package dependency on `idea-graph`
- only neutral exported artifacts may cross the boundary
- every controller decision must be logged for offline diagnosis
- benchmark execution and final proposal synthesis are part of the MVP

Target runtime shape:

- frozen graph snapshot per step
- global commit pre-check
- role activation gate
- parallel candidate generation for active roles
- per-role `skip` candidate
- centralized edit scoring and selection
- deterministic action materialization
- global commit post-check
- final proposal synthesis from graph state

## Repo Layout

Planned top-level structure:

- `README.md`
- `pyproject.toml`
- `src/evolving_idea_graph/`
- `scripts/`
- `tests/`
- `configs/`
- `data/benchmarks/`
- `data/artifact_imports/`
- `outputs/`
- `docs/`

Core package layout:

- `src/evolving_idea_graph/benchmarking/`
- `src/evolving_idea_graph/graph/`
- `src/evolving_idea_graph/roles/`
- `src/evolving_idea_graph/controller/`
- `src/evolving_idea_graph/critic/`
- `src/evolving_idea_graph/synthesis/`
- `src/evolving_idea_graph/runtime/`
- `src/evolving_idea_graph/io/`

## Delivery Strategy

Build in this order:

1. repository bootstrap and artifact boundary
2. graph schema, actions, and benchmark packets
3. parallel role proposal runtime and heuristic controller
4. graph critic encoder, edit scorer, and commit scorer
5. synthesis and end-to-end benchmark runner
6. offline training, calibration, and verification artifacts

Each task below should end in a small commit so the new repo stays easy to
review and reset.

### Task 1: Bootstrap The Standalone Repository

**Files:**
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `src/evolving_idea_graph/__init__.py`
- Create: `tests/test_repo_smoke.py`
- Create: `docs/architecture.md`

- [ ] Initialize `..\evolving-idea-graph` as a new git repository with its own
  `.gitignore`, package metadata, and `src/` layout.
- [ ] Write `README.md` with the standalone guarantee, repo scope, and the
  minimum runtime loop.
- [ ] Add a smoke test that imports `evolving_idea_graph` and checks the package
  version string or package sentinel.
- [ ] Run `python -m pytest tests/test_repo_smoke.py -q` and confirm the repo
  boots cleanly.
- [ ] Commit with `chore: bootstrap evolving-idea-graph repository`.

### Task 2: Define The Artifact Import Boundary

**Files:**
- Create: `src/evolving_idea_graph/io/artifact_manifest.py`
- Create: `src/evolving_idea_graph/io/artifact_loader.py`
- Create: `data/artifact_imports/README.md`
- Create: `tests/test_artifact_loader.py`
- Modify: `README.md`

- [ ] Define a neutral manifest schema for imported benchmark packets, replay
  rows, and training artifacts.
- [ ] Implement import utilities that read only declared files under
  `data/artifact_imports/`.
- [ ] Reject absolute paths, parent-directory traversal, and repo-external file
  reads in the loader.
- [ ] Add tests that prove allowed neutral imports succeed and cross-repo path
  reads fail.
- [ ] Commit with `feat: add standalone artifact import boundary`.

### Task 3: Implement Typed Graph State And Action Dataclasses

**Files:**
- Create: `src/evolving_idea_graph/graph/schema.py`
- Create: `src/evolving_idea_graph/graph/actions.py`
- Create: `src/evolving_idea_graph/graph/state.py`
- Create: `src/evolving_idea_graph/graph/validation.py`
- Create: `tests/test_graph_state.py`

- [ ] Define typed node, edge, graph snapshot, and action dataclasses for
  scientific ideation state.
- [ ] Support role ownership, provenance ids, timestamps, and candidate/action
  ids so later traces are auditable.
- [ ] Implement graph validation for missing endpoints, duplicate ids, invalid
  edge types, and illegal action payloads.
- [ ] Add state-application tests for add-node, add-edge, revise-node,
  revise-edge, and `skip`.
- [ ] Commit with `feat: add typed graph state and action schema`.

### Task 4: Add Benchmark Packets And Run Manifest Primitives

**Files:**
- Create: `src/evolving_idea_graph/benchmarking/packets.py`
- Create: `src/evolving_idea_graph/benchmarking/instances.py`
- Create: `src/evolving_idea_graph/runtime/run_manifest.py`
- Create: `configs/benchmarks/aiib.yaml`
- Create: `configs/benchmarks/liveideabench.yaml`
- Create: `tests/test_benchmark_packets.py`

- [ ] Define benchmark-instance dataclasses, packet manifests, and run metadata
  that are independent of the current repo naming.
- [ ] Support frozen packet ids, benchmark names, split labels, and per-instance
  provenance fields.
- [ ] Implement serialization for per-run manifests and step-level artifact
  pointers.
- [ ] Add tests that load a tiny packet fixture and round-trip both packet and
  run manifest JSON.
- [ ] Commit with `feat: add benchmark packets and run manifest primitives`.

### Task 5: Build The Role Registry, Views, And Candidate Schema

**Files:**
- Create: `src/evolving_idea_graph/roles/registry.py`
- Create: `src/evolving_idea_graph/roles/base.py`
- Create: `src/evolving_idea_graph/roles/views.py`
- Create: `src/evolving_idea_graph/roles/prompts.py`
- Create: `tests/test_role_registry.py`

- [ ] Implement the default five-role registry:
  `MechanismProposer`, `FeasibilityCritic`, `NoveltyExaminer`,
  `EvaluationDesigner`, and `ImpactReframer`.
- [ ] Define role-specific graph views so each role receives only the fields it
  needs plus a compact state summary.
- [ ] Define a role proposal format with role id, candidate rank, action type,
  natural-language rationale, structured payload, and explicit `skip`.
- [ ] Add tests that enforce role registration, view filtering, and `skip`
  candidate availability for every role.
- [ ] Commit with `feat: add role registry and candidate schema`.

### Task 6: Implement Parallel Role Proposal Execution And Activation Gating

**Files:**
- Create: `src/evolving_idea_graph/runtime/role_activation.py`
- Create: `src/evolving_idea_graph/runtime/proposal_executor.py`
- Create: `src/evolving_idea_graph/runtime/llm_client.py`
- Create: `configs/models/qwen3_8b.yaml`
- Create: `tests/test_proposal_executor.py`
- Modify: `configs/benchmarks/aiib.yaml`

- [ ] Implement a cheap role-activation gate that decides which roles should be
  queried on the current graph snapshot.
- [ ] Implement parallel role proposal execution over the active role set using
  a shared client interface and per-role prompts.
- [ ] Set the first default proposal model configuration to `qwen3-8b` and keep
  the MVP role set fixed at the approved five roles.
- [ ] Enforce one snapshot in, many proposals out, with no direct graph mutation
  inside the role workers.
- [ ] Log token counts, latency, model name, request id, and raw proposal text
  for later overhead analysis.
- [ ] Add tests with a fake client that prove inactive roles are skipped and
  active roles return role-local candidates plus `skip`.
- [ ] Commit with `feat: add parallel role proposal execution`.

### Task 7: Build The Round Runner And Deterministic Materialization Loop

**Files:**
- Create: `src/evolving_idea_graph/runtime/round_runner.py`
- Create: `src/evolving_idea_graph/runtime/materialize.py`
- Create: `src/evolving_idea_graph/runtime/tracing.py`
- Create: `tests/test_round_runner.py`

- [ ] Implement the synchronized runtime step:
  commit pre-check, activation, parallel proposals, validation, scoring,
  per-role selection, deterministic materialization, commit post-check.
- [ ] Enforce the MVP policy of at most one materialized action per active role
  in each step.
- [ ] Write step traces that store selected action ids, rejected candidates,
  selected-role distribution, and graph deltas.
- [ ] Add tests that prove deterministic materialization order is stable even
  when proposal return order changes.
- [ ] Commit with `feat: add round runner and materialization loop`.

### Task 8: Add Controller Interfaces And A Heuristic Baseline

**Files:**
- Create: `src/evolving_idea_graph/controller/interfaces.py`
- Create: `src/evolving_idea_graph/controller/heuristic_controller.py`
- Create: `src/evolving_idea_graph/controller/selection.py`
- Create: `tests/test_heuristic_controller.py`
- Modify: `README.md`

- [ ] Define controller interfaces for edit scoring, commit scoring, and final
  action selection so learned and heuristic controllers share one runtime path.
- [ ] Implement a heuristic baseline that scores candidates from graph deficits,
  novelty-risk balance, evaluation coverage, and action repetition penalties.
- [ ] Keep commit as a separate global decision interface rather than folding it
  into ordinary edit ranking.
- [ ] Add tests that prove the heuristic controller can select per-role edits
  while leaving commit to the commit interface.
- [ ] Commit with `feat: add controller interfaces and heuristic baseline`.

### Task 9: Implement The Shared Graph Critic Encoder And Edit Head

**Files:**
- Create: `src/evolving_idea_graph/critic/dataset.py`
- Create: `src/evolving_idea_graph/critic/encoder.py`
- Create: `src/evolving_idea_graph/critic/edit_head.py`
- Create: `src/evolving_idea_graph/critic/train_edit_head.py`
- Create: `scripts/train_edit_critic.py`
- Create: `tests/test_edit_critic.py`

- [ ] Define the critic dataset format around graph snapshots, role ids,
  candidate actions, selected labels, and replay metadata.
- [ ] Implement a relation-aware graph encoder with node text integration,
  relation embeddings, and target-aware pooling.
- [ ] Implement an edit head that scores one candidate action for one role on
  one frozen graph snapshot.
- [ ] Train and evaluate the scorer on imported replay artifacts with frozen
  train and validation manifests.
- [ ] Add tests for dataset collation, encoder forward shape, and candidate
  ranking output.
- [ ] Commit with `feat: add graph critic edit scorer`.

### Task 10: Implement The Global Commit Head And Calibration Layer

**Files:**
- Create: `src/evolving_idea_graph/critic/commit_head.py`
- Create: `src/evolving_idea_graph/critic/calibration.py`
- Create: `scripts/train_commit_head.py`
- Create: `scripts/calibrate_commit_head.py`
- Create: `tests/test_commit_head.py`

- [ ] Implement a graph-level commit head over the shared encoder output to
  predict whether the current graph should stop now.
- [ ] Keep shadow-commit training and live-commit thresholding as separate
  artifacts so calibration can be updated without retraining the encoder.
- [ ] Support probability calibration on a frozen development split with saved
  metrics and threshold tables.
- [ ] Add tests that verify commit probabilities, threshold lookup, and shadow
  versus live commit decisions.
- [ ] Commit with `feat: add commit scorer and calibration layer`.

### Task 11: Build Proposal Synthesis From Final Graph State

**Files:**
- Create: `src/evolving_idea_graph/synthesis/claim_chain.py`
- Create: `src/evolving_idea_graph/synthesis/proposal_writer.py`
- Create: `src/evolving_idea_graph/synthesis/sections.py`
- Create: `tests/test_proposal_synthesis.py`

- [ ] Implement claim-chain extraction from the final graph, including support,
  contradiction, evaluation, and impact links.
- [ ] Build a synthesis layer that turns the final graph into benchmark-facing
  proposal sections with provenance references to graph nodes and edges.
- [ ] Support proposal generation both after live commit and after max-step
  fallback termination.
- [ ] Add tests that prove the synthesized proposal includes method, novelty,
  feasibility, and evaluation content sourced from graph state.
- [ ] Commit with `feat: add proposal synthesis pipeline`.

### Task 12: Add The End-To-End Benchmark Runner

**Files:**
- Create: `src/evolving_idea_graph/runtime/benchmark_runner.py`
- Create: `scripts/run_benchmark_packet.py`
- Create: `scripts/evaluate_benchmark_packet.py`
- Create: `configs/runtime/default.yaml`
- Create: `tests/test_benchmark_runner.py`

- [ ] Implement an end-to-end runner that loads a frozen packet, executes the
  runtime controller per instance, writes full traces, and saves final proposal
  outputs in a standard output tree.
- [ ] Save run-level profiling fields including number of active roles, total
  proposals, tokens, latency, API cost, selected action distribution, commit
  decision path, and termination reason.
- [ ] Add a paired-evaluation mode so heuristic, text-only, and graph-critic
  controllers can be compared on the same packet.
- [ ] Add a smoke test over a tiny packet fixture that exercises the full path
  from instance load to final proposal artifact.
- [ ] Commit with `feat: add end-to-end benchmark runner`.

### Task 13: Verification, Docs, And Experiment Hand-Off

**Files:**
- Create: `docs/runtime_protocol.md`
- Create: `docs/datasets.md`
- Create: `docs/experiments.md`
- Create: `docs/migration_notes.md`
- Modify: `README.md`

- [ ] Document the runtime protocol, artifact boundary, dataset schema, model
  training flow, and benchmark output layout.
- [ ] Write a migration note that explains how neutral exports from the current
  repo can be copied in without creating a dependency.
- [ ] Run the verification suite:
  `python -m pytest tests -q`, training-script smoke tests, and one tiny
  benchmark packet dry run.
- [ ] Record the initial execution recipe for offline critic training, commit
  calibration, heuristic baseline runs, and paired benchmark evaluation.
- [ ] Commit with `docs: add evolving-idea-graph implementation handoff`.

## Final Acceptance Checklist

- [ ] New repo exists at `..\evolving-idea-graph`
- [ ] No code import or runtime path dependency on `idea-graph`
- [ ] Parallel role proposal runtime is functional
- [ ] Centralized edit scoring and separate commit scoring are wired in
- [ ] End-to-end packet execution produces run manifests and final proposals
- [ ] Training and calibration scripts operate on neutral artifact inputs
- [ ] Runtime traces expose action distribution, commit behavior, token usage,
  latency, and cost for later paper analysis
- [ ] Docs explain both the method story and the engineering boundary clearly
