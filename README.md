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
- `docs/benchmarks.md`
- `pyproject.toml`
- `data/sample_instance.json`
- `src/idea_graph/`
- `scripts/run_pipeline.py`
- `scripts/fetch_ai_idea_bench_2025.py`
- `scripts/fetch_liveideabench.py`

## Run

```bash
python scripts/run_pipeline.py
```

The default pipeline run will:

- load `data/sample_instance.json`
- execute the deterministic graph collaboration pipeline
- print a short terminal summary
- write `graph.json`, `summary.json`, and `final_proposal.md` into `outputs/<timestamp>-<instance>/`

## Package Structure

- `src/idea_graph/instances.py`
  Typed experiment-instance input objects shared by local JSON and benchmark loaders.
- `src/idea_graph/models.py`
  Graph, node, edge, branch, action, and proposal data models.
- `src/idea_graph/schema.py`
  Graph schema and benchmark-aware seed-template construction.
- `src/idea_graph/engine.py`
  Seed graph construction, merge, collaboration rounds, maturity checks, and synthesis.
- `src/idea_graph/io.py`
  Local instance loading and run artifact writing.
- `src/idea_graph/benchmarks/`
  Benchmark-specific downloaders, parsers, and converters.

## AI Idea Bench 2025

The pipeline can also use the official `AI_Idea_Bench_2025` benchmark metadata
instead of the local sample instance.

Download the benchmark metadata:

```bash
python scripts/fetch_ai_idea_bench_2025.py
```

Download and extract the full paper archive:

```bash
python scripts/fetch_ai_idea_bench_2025.py --include-paper-assets --extract-paper-assets --allow-large-download
```

Run one benchmark row:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 0
```

Or let the runner download metadata automatically if it is missing:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 0 --download-if-missing
```

Notes:

- The default benchmark integration uses the official `target_paper_data.json`
  metadata file.
- `--benchmark-index` accepts either the official benchmark `index` value or a
  zero-based row position in the downloaded metadata file.
- The optional `--include-paper-assets` flag downloads the official
  `Idea_bench_data.zip` archive, which is about 31 GB.
- Large paper-asset downloads are blocked by default and require the extra
  `--allow-large-download` confirmation flag.
- The optional `--extract-paper-assets` flag unpacks that archive into the
  official `papers_data/` layout expected by the benchmark metadata.
- For the current prototype, literature context is built from the benchmark's
  provided top-reference titles and their local paper paths when available.

## liveideabench

The pipeline can also use the official `liveideabench` dataset hosted on
Hugging Face.

Download the benchmark CSV:

```bash
python scripts/fetch_liveideabench.py
```

Run one row by zero-based row index:

```bash
python scripts/run_pipeline.py --benchmark liveideabench --benchmark-index 0
```

Run the first row for a given keyword:

```bash
python scripts/run_pipeline.py --benchmark liveideabench --benchmark-keyword meteorology --benchmark-index 0
```

Or let the runner download it automatically if missing:

```bash
python scripts/run_pipeline.py --benchmark liveideabench --benchmark-keyword meteorology --download-if-missing
```

Notes:

- `liveideabench` is distributed as a single CSV benchmark file.
- The benchmark provides a keyword prompt and scored idea text, not retrieved
  literature in the same style as `AI_Idea_Bench_2025`.
- For this prototype, the keyword is used as the ideation topic and the scored
  benchmark idea is kept in run metadata rather than fed back as literature.

## Next steps

1. Replace deterministic seed generation with role-specific LLM prompts.
2. Replace heuristic edit selection with structured graph actions from agents.
3. Add benchmark loading and retrieved literature inputs.
4. Implement the comparison baselines from the protocol.
5. Add full trace logging and evaluation metrics.
