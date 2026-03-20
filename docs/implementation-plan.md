# Implementation Plan

This plan is based on the experimental protocol extracted from `exp.docx`.

## Goal

Implement a research prototype that tests whether graph-mediated delayed
consensus improves scientific idea generation relative to:

- single-agent self-refinement
- multi-agent discussion plus voting
- graph collaboration without delayed consensus
- delayed consensus without a typed graph

## Experimental Unit

Each ideation instance contains:

- a research topic
- a retrieved literature set
- a fixed inference budget
- one final abstract-length proposal

## Fixed Roles

- `MechanismProposer`
- `FeasibilityCritic`
- `NoveltyExaminer`
- `EvaluationDesigner`
- `ImpactReframer`

## Typed Graph Schema

Node types:

- `Problem`
- `Hypothesis`
- `Method`
- `Assumption`
- `Risk`
- `EvidenceNeed`
- `EvalPlan`
- `NoveltyClaim`
- `Repair`

Edge types:

- `supports`
- `contradicts`
- `refines`
- `depends_on`
- `requires_evidence`
- `overlaps_prior`
- `repairs`

## Implementation Phases

### Phase 1

Build the protocol as a deterministic state machine.

Deliverables:

- graph data model
- role-specific seed templates
- graph merge with provenance
- constrained graph actions for three rounds
- maturity checks
- final subgraph selection
- structured proposal synthesis

### Phase 2

Plug in real agent calls.

Deliverables:

- prompt builders per role
- focused graph-view builders
- structured action parsing
- evidence attachment hooks

### Phase 3

Support retrieval and datasets.

Deliverables:

- benchmark instance loader
- retrieved literature format
- per-run traces and snapshots
- logging for process metrics

### Phase 4

Implement baselines under matched budgets.

Deliverables:

- single-agent self-refinement baseline
- discussion-plus-voting multi-agent baseline
- graph without delayed consensus baseline
- delayed consensus without graph baseline

## Immediate Coding Plan

1. Create a clean Python package under `src/idea_graph`.
2. Implement domain models and schema constants.
3. Implement merge, action application, and maturity logic.
4. Add a deterministic prototype runner.
5. Leave clear extension points for real LLM-backed agents.

## Notes

- The DOCX extraction was successful and gives a precise implementation target.
- The PDF could not be fully text-extracted in the current local environment, so
  the code plan is anchored primarily on the DOCX protocol.
- A working Python interpreter is not currently exposed in this shell, so I may
  not be able to execute the Python code locally after writing it.
