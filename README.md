# idea-graph

`idea-graph` is a Python research prototype for scientific ideation with
Evolving Idea Graphs (EIG). The system represents a developing research idea as
a typed graph, lets role-specialized agents propose graph edits, and synthesizes
a final proposal from the committed graph state.

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
- A synthetic toy input in `data/sample_instance.json`

## What Is Not Included

- `outputs/`: generated runs, tables, logs, and paper-evaluation results
- `data/benchmarks/`: downloaded benchmark files and paper assets
- `data/splits/`: private train/dev/evaluation split registries
- `docs/`: internal plans, execution logs, and project records
- `models/`, `checkpoints/`, and serialized critic artifacts
- Any API keys or provider-specific private configs

The `.gitignore` is configured to keep those artifacts local.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e .
```

The package depends on `numpy`, `scikit-learn`, `torch`, and
`sentence-transformers`. Some paths, especially learned-controller paths, may
also require private model artifacts that are not part of this public code
release.

## Quick Start

Run the deterministic toy example:

```bash
python scripts/run_pipeline.py
```

This uses `data/sample_instance.json` and writes local artifacts under
`outputs/`, which is ignored by Git.

Run a specific local baseline on the toy input:

```bash
python scripts/run_pipeline.py --baseline direct
python scripts/run_pipeline.py --baseline self-refine
python scripts/run_pipeline.py --baseline ours-eig --runtime-protocol parallel_graph_v2
```

## OpenAI-Compatible Backend

Start from the public example config:

```bash
configs/openai_compatible.example.json
```

Set credentials through environment variables rather than writing keys into
JSON files:

```powershell
$env:DASHSCOPE_API_KEY="your_real_key"
python scripts/check_openai_compatible.py --llm-config configs/openai_compatible.example.json
python scripts/run_pipeline.py --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json
```

The config field `api_key_env` should contain an environment-variable name such
as `DASHSCOPE_API_KEY` or `OPENAI_API_KEY`, not a literal key.

## Benchmarks

Benchmark data is not committed. To run benchmark loaders locally, download the
official files into ignored local directories:

```bash
python scripts/fetch_ai_idea_bench_2025.py
python scripts/fetch_liveideabench.py
```

Then run an instance by benchmark index:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 0
python scripts/run_pipeline.py --benchmark liveideabench --benchmark-index 0
```

Large paper archives and benchmark assets remain local under `data/benchmarks/`.
Do not commit them.

## Learned Controllers And Private Artifacts

The code contains controller integration paths for text critics, graph critics,
two-head graph critics, fixed control, and random control. Trained critic models
and paper-evaluation calibration artifacts are not included in this repository.

If you have private controller artifacts, keep them in ignored local paths such
as `outputs/critic_models/` or provide explicit paths through local configs or
metadata. If no private calibration artifact is present, calibrated controller
variants fall back to the built-in uncalibrated defaults and mark the
calibration as missing in runtime metadata.

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
git ls-files outputs data/benchmarks data/splits docs
```

The second command should not list files for a code-only public push. Keep API
keys, generated outputs, benchmark datasets, trained models, and internal plans
outside Git.
