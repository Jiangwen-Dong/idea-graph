# idea-graph

`idea-graph` is a Python research prototype for scientific ideation with
Evolving Idea Graphs (EIG). It represents a developing research idea as a typed
graph, lets role-specialized agents edit that shared state, and synthesizes a
final proposal from the committed graph.

This repository is prepared as a code-only public version. It intentionally
does not include experiment outputs, benchmark datasets, trained model
checkpoints, private split registries, private calibration artifacts, or
internal research logs.

## What Is Included

- Core EIG graph data models and runtime logic in `src/idea_graph/`
- Deterministic local backend for smoke tests and reproducible graph mechanics
- OpenAI-compatible backend support for local provider configuration
- Benchmark loader and evaluation code paths
- Baseline adapters and controller variants used by the research prototype
- Public configuration examples in `configs/`
- Unit tests in `tests/`

## Public Surface

The public entrypoints are:

- `python scripts/run_pipeline.py ...`
- `python scripts/check_openai_compatible.py ...`
- `python scripts/fetch_ai_idea_bench_2025.py`
- `python scripts/fetch_liveideabench.py`

Research utilities are grouped under:

- `scripts/analysis/`
- `scripts/data_prep/`
- `scripts/eval/`
- `scripts/train/`

See `scripts/README.md` for the grouped layout.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e .
```

The package depends on `numpy`, `scikit-learn`, `torch`, and
`sentence-transformers`. Learned-controller paths can require local model
artifacts that are intentionally not part of this public release.

## Quick Start

Run the pipeline on your own JSON instance:

```bash
python scripts/run_pipeline.py --input /path/to/instance.json
```

The input JSON should contain the fields `name`, `topic`, and `literature`.
A minimal example looks like:

```json
{
  "name": "my-idea",
  "topic": "Collaborative scientific ideation over typed idea graphs",
  "literature": [
    "Prior systems often decide too early over whole drafts.",
    "Structured state can preserve partial claims and dependencies."
  ],
  "metadata": {
    "source": "local"
  }
}
```

Runs write artifacts under `outputs/`, which is ignored by Git.

Run a specific local baseline on your JSON input:

```bash
python scripts/run_pipeline.py --input /path/to/instance.json --baseline direct
python scripts/run_pipeline.py --input /path/to/instance.json --baseline self-refine
python scripts/run_pipeline.py --input /path/to/instance.json --baseline ours-eig --runtime-protocol parallel_graph_v2
```

## Baseline Families

The repository includes several baseline families:

- Direct proposal generation: `direct`
- Single-agent revision: `self-refine`
- EIG variants: `ours-eig` and controller-based variants such as text critic,
  relation-graph critic, two-head graph critic, signal heuristic control, fixed
  control, and random control
- Single-model graph reasoning: `graph-of-thought`
- External upstream wrappers: `ai-researcher`, `scipip`, `virsci`
- Local workflow variants inspired by upstream systems:
  `ai-researcher-guided`, `scipip-structured`, and `virsci-discussion`

The three local workflow variants are benchmark-facing local implementations
that follow the high-level workflow shape of the named systems. They are not
claims of exact upstream reproduction.

## OpenAI-Compatible Backend

Start from the public example config:

```bash
configs/openai_compatible.example.json
```

Set credentials through environment variables rather than writing keys into JSON
files:

```powershell
$env:DASHSCOPE_API_KEY="your_real_key"
python scripts/check_openai_compatible.py --llm-config configs/openai_compatible.example.json
python scripts/run_pipeline.py --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json
```

The config field `api_key_env` should contain an environment-variable name such
as `DASHSCOPE_API_KEY` or `OPENAI_API_KEY`, not a literal key.

## Benchmarks

Benchmark data is not committed. By default, the fetch scripts download the
official files into ignored local paths under `data/benchmarks/`:

```bash
python scripts/fetch_ai_idea_bench_2025.py
python scripts/fetch_liveideabench.py
```

Then run an instance by benchmark index:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 0
python scripts/run_pipeline.py --benchmark liveideabench --benchmark-index 0
```

If you want the runner to fetch missing benchmark metadata automatically, use:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 0 --download-if-missing
python scripts/run_pipeline.py --benchmark liveideabench --benchmark-index 0 --download-if-missing
```

If you prefer a different local cache location, override it with
`--benchmark-root /path/to/benchmarks`.

Large paper archives and benchmark assets remain local. Do not commit them.

## Learned Controllers And Private Artifacts

The code contains controller integration paths for text critics, relation-graph
critics, two-head graph critics, signal-heuristic control, fixed control, and
random control. Trained critic models and paper-evaluation calibration artifacts
are not included in this repository.

If you have private controller artifacts, keep them in ignored local paths such
as `outputs/critic_models/` or provide explicit paths through local configs or
metadata. Temporary external-baseline workspaces should likewise stay under
ignored paths such as `outputs/tmp/`.

The public repository includes the runtime code for these controllers, but not
the learned checkpoints needed to reproduce the trained-controller results.

## External Baselines

The external baselines `ai-researcher`, `scipip`, and `virsci` require local
copies of their upstream repositories plus local configuration. Their temporary
workspaces should remain under ignored paths such as
`outputs/tmp/external-baseline-runs/`.

The fixed-control ablation uses the public example schedule:

```bash
configs/fixed_control_policy.example.json
```

## Tests

Run the test suite with:

```bash
pytest
```

For a faster smoke check of the public-release cleanup behavior:

```bash
pytest tests/test_benchmark_mode_and_baselines.py tests/test_experiment_plans.py -q
```

## Repository Hygiene

Before pushing, check:

```bash
git status --short
git ls-files outputs data docs models checkpoints
```

The second command should not list files for a code-only public push. Keep API
keys, generated outputs, benchmark datasets, trained models, and internal plans
outside Git.
