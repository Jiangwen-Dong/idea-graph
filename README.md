# idea-graph

Python scaffold for a delayed-consensus scientific ideation system built around
a shared typed idea graph.

The implementation direction comes from the extracted `exp.docx` protocol:

- five fixed epistemic roles
- private seed-graph generation
- shared graph merge with provenance
- three constrained collaboration rounds
- no early whole-idea voting
- maturity checks before final synthesis
- final proposal generation from a selected high-utility subgraph

## Current status

This repository now contains the first Python implementation step:

- typed graph schema and data models
- deterministic placeholder seed generation
- graph merge and constrained graph actions
- maturity checks and final subgraph selection
- a runnable prototype script

The current prototype is intentionally model-free. It exercises the protocol as
code before we plug in retrieval, LLM agents, and baselines.

## Layout

- `docs/implementation-plan.md`
- `pyproject.toml`
- `data/sample_instance.json`
- `src/idea_graph/`
- `scripts/run_pipeline.py`
- `scripts/run_prototype.py`

## Run

```bash
python scripts/run_pipeline.py
```

or keep the original wrapper:

```bash
python scripts/run_prototype.py
```

The default pipeline run will:

- load `data/sample_instance.json`
- execute the deterministic graph collaboration pipeline
- print a short terminal summary
- write `graph.json`, `summary.json`, and `final_proposal.md` into `outputs/<timestamp>-<instance>/`

## Next steps

1. Replace deterministic seed generation with role-specific LLM prompts.
2. Replace heuristic edit selection with structured graph actions from agents.
3. Add benchmark loading and retrieved literature inputs.
4. Implement the comparison baselines from the protocol.
5. Add full trace logging and evaluation metrics.
