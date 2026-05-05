<div align="center">

# 🧠 idea-graph

**Evolving Idea Graphs (EIG) for Multi-Agent Scientific Ideation**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.4-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

`idea-graph` is a Python research prototype for scientific ideation with Evolving Idea Graphs (EIG). It represents a developing research idea as a **typed graph**, lets role-specialized agents edit that shared state, and synthesizes a final proposal from the committed graph.

---

## 📋 Table of Contents

- [✨ What Is Included](#-what-is-included)
- [🚀 Quick Start](#-quick-start)
- [🧪 Baseline Families](#-baseline-families)
- [🔌 OpenAI-Compatible Backend](#-openai-compatible-backend)
- [📊 Benchmarks](#-benchmarks)
- [🎓 Learned Controllers](#-learned-controllers)
- [🔗 External Baselines](#-external-baselines)
- [🧪 Tests](#-tests)
- [🧹 Repository Hygiene](#-repository-hygiene)

---

## ✨ What Is Included

| Component | Description |
|-----------|-------------|
| `src/idea_graph/` | Core EIG graph data models and runtime logic |
| `tests/` | Unit tests and smoke checks |
| `configs/` | Configuration examples for backends and control policies |
| `scripts/` | Pipeline runners, analysis, data prep, evaluation, and training |

- **Deterministic local backend** for smoke tests and reproducible graph mechanics
- **OpenAI-compatible backend** support for local provider configuration
- **Benchmark loader** and evaluation code paths
- **Baseline adapters** and controller variants
- **Training pipeline** for graph critics with offline supervision collection

### 📂 Public Entrypoints

```bash
# Main pipeline
python scripts/run_pipeline.py ...

# Backend connectivity check
python scripts/check_openai_compatible.py ...

# Benchmark fetchers
python scripts/fetch_ai_idea_bench_2025.py
python scripts/fetch_liveideabench.py
```

Research utilities are grouped under:

- `scripts/analysis/`
- `scripts/data_prep/`
- `scripts/eval/`
- `scripts/train/`

See [`scripts/README.md`](scripts/README.md) for the grouped layout.

---

## 🚀 Quick Start

### 📦 Installation

Requires **Python 3.10 or newer**:

```bash
python -m pip install -e .
```

Core dependencies: `numpy`, `scikit-learn`, `torch`, `sentence-transformers`.

### ▶️ Run the Pipeline

On your own JSON instance:

```bash
python scripts/run_pipeline.py --input /path/to/instance.json
```

The input JSON should contain the fields `name`, `topic`, and `literature`. A minimal example:

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

Runs write artifacts under `outputs/`.

### 🎛️ Run a Specific Baseline

```bash
# One-shot generation
python scripts/run_pipeline.py --input /path/to/instance.json --baseline direct

# Single-agent revision
python scripts/run_pipeline.py --input /path/to/instance.json --baseline self-refine

# Full EIG with parallel graph runtime
python scripts/run_pipeline.py --input /path/to/instance.json \
  --baseline ours-eig --runtime-protocol parallel_graph_v2
```

---

## 🧪 Baseline Families

| Family | Baseline Key | Description |
|--------|--------------|-------------|
| Direct | `direct` | One-shot proposal generation |
| Self-Refine | `self-refine` | Single-agent iterative revision |
| **EIG (Ours)** | `ours-eig` | Full EIG with graph-based collaboration and learned control |
| Graph of Thoughts | `graph-of-thought` | Single-model graph reasoning |
| AI-Researcher | `ai-researcher` | Multi-stage literature-grounded pipeline |
| SciPIP | `scipip` | Structured planning-and-drafting pipeline |
| VirSci | `virsci` | Discussion-oriented multi-agent proposal system |

**Controller variants** (for `ours-eig`):

- Text critic
- Relation-graph critic
- Two-head graph critic
- Signal-heuristic control
- Random control

---

## 🔌 OpenAI-Compatible Backend

Start from the public example config:

```bash
configs/openai_compatible.example.json
```

Set credentials through **environment variables** rather than writing keys into JSON files:

```powershell
# PowerShell
$env:DASHSCOPE_API_KEY="your_real_key"
python scripts/check_openai_compatible.py \
  --llm-config configs/openai_compatible.example.json

python scripts/run_pipeline.py \
  --agent-backend openai-compatible \
  --llm-config configs/openai_compatible.example.json
```

> **Important:** The config field `api_key_env` should contain an environment-variable name such as `DASHSCOPE_API_KEY` or `OPENAI_API_KEY`, **not a literal key**.

---

## 📊 Benchmarks

Fetch the official benchmark files:

```bash
python scripts/fetch_ai_idea_bench_2025.py
python scripts/fetch_liveideabench.py
```

Datasets are hosted on HuggingFace:

- [AI Idea Bench 2025](https://huggingface.co/datasets/yanshengqiu/AI_Idea_Bench_2025)
- [LiveIdeaBench](https://huggingface.co/datasets/6cf/liveideabench)

Then run by benchmark index:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 0
python scripts/run_pipeline.py --benchmark liveideabench --benchmark-index 0
```

Auto-fetch missing metadata:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 \
  --benchmark-index 0 --download-if-missing
```

Override the local cache location:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 \
  --benchmark-index 0 --benchmark-root /path/to/benchmarks
```

> **Note:** Benchmark assets are downloaded to local paths under `data/benchmarks/` by default.

---

## 🎓 Learned Controllers

The repository includes controller integration paths and training code for:

- Text critics
- Relation-graph critics
- Two-head graph critics
- Signal-heuristic control
- Random control

The training pipeline is in `scripts/train/`. It consumes offline profiling traces collected under the EIG runtime and produces trained critic checkpoints. See `scripts/train/` for training configuration and `scripts/data_prep/` for supervision corpus construction.

---

## 🔗 External Baselines

The external baselines `ai-researcher`, `scipip`, and `virsci` require local copies of their upstream repositories plus local configuration. Their temporary workspaces are written to `outputs/tmp/external-baseline-runs/` by default.

The fixed-control ablation uses the example schedule:

```bash
configs/fixed_control_policy.example.json
```

---

## 🧪 Tests

Run the full test suite:

```bash
pytest
```

Fast smoke check:

```bash
pytest tests/test_benchmark_mode_and_baselines.py \
       tests/test_experiment_plans.py -q
```

---

## 🧹 Repository Hygiene

Before pushing, verify no artifacts are staged:

```bash
git status --short
git ls-files outputs data docs models checkpoints
```

The second command should return **nothing** for a clean push. Keep API keys, generated outputs, benchmark datasets, trained models, and internal plans outside Git.

---

<div align="center">

**[⬆ Back to Top](#-idea-graph)**

</div>
