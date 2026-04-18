# idea-graph

Python implementation of an `Evolving Idea Graph (EIG)` system for
benchmark-faithful scientific ideation.

The current prototype treats scientific ideation as iterative editing over a
shared typed graph rather than a single draft. It supports both deterministic
debugging runs and OpenAI-compatible LLM-backed runs.

Core design elements:

- five fixed epistemic roles
- private seed-graph generation
- shared graph merge with provenance
- configurable collaboration rounds with maturity-based stopping
- no early whole-idea voting
- utility-guided graph editing before final synthesis
- final proposal generation from a selected high-utility subgraph

## Current status

This repository now contains a runnable research prototype with:

- typed graph schema and data models
- deterministic and OpenAI-compatible multi-agent backends
- graph merge and constrained graph actions
- maturity checks, claim-chain selection, and final subgraph synthesis
- benchmark loaders and runnable pipeline scripts
- benchmark-native evaluation hooks and local development-time evaluation
- external baseline adapters plus local baseline wrappers

The pipeline can now run in two modes:

- `deterministic`
  Useful for local validation, debugging, and reproducible graph mechanics.
- `openai-compatible`
  Uses an OpenAI-style `chat/completions` API protocol so you can target
  ChatGPT-style, Qwen-style, DeepSeek-style, or other compatible providers by
  changing the base URL, API key, and model names.

## Docs

Start with:

- `docs/README.md`

Most useful active docs:

- `docs/paper_protocol.md`
- `docs/reproducibility.md`
- `docs/eig_graph_critic_plan.md`
- `docs/evaluation.md`
- `docs/critic_pools.md`
- `docs/paper_experiment_plan.md`
- `docs/paper_experiment_tracker.md`
- `docs/experiment_execution_log.md`

Current experiment status:

- The active runtime is `parallel_graph_v2`.
- The current bootstrap teacher is the parallel-v2 heuristic controller.
- The tracked critic train/dev split is:
  `data/splits/parallel_v2/critic_train_dev_registry.jsonl`
  - `300` critic-train groups
  - `100` critic-dev groups
- The tracked frozen paper-eval split is:
  `data/splits/parallel_v2/paper_eval_v2_registry.jsonl`
  - `256` total groups
  - `128` AI Idea Bench 2025 groups
  - `128` LiveIdeaBench groups
  - zero overlap with critic train/dev in the tracked audit
- The tracked frozen-dev controller calibration artifact is:
  `data/splits/parallel_v2/frozen_dev_joint_controller_calibration.json`
  - `ours-eig-critic-graph-twohead` explicitly disables calibration
  - `ours-eig-critic-calibrated` and `ours-eig-critic-no-edit` resolve this artifact without CLI overrides
- The older sequential and text-critic artifacts remain useful historical
  ablations, but new method development should use parallel-v2 unless an older
  result is being reproduced intentionally.

## Layout

- `docs/`
- `configs/openai_compatible.example.json`
- `configs/external_baselines.example.json`
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
- write `graph.json`, `summary.json`, `final_proposal.md`, `evaluation.json`, and `evaluation.md` into `outputs/<timestamp>-<instance>/`
- optionally write `benchmark_native_evaluation.json` and `benchmark_native_evaluation.md` when `--native-eval` is enabled

Re-evaluate an existing run:

```bash
python scripts/evaluate_run.py --run-dir outputs/<timestamp>-<instance>
```

Re-evaluate an existing run with benchmark-native scoring:

```bash
python scripts/evaluate_run.py --run-dir outputs/<timestamp>-<instance> --native-eval --llm-config configs/openai_compatible.example.json
```

Run a local baseline wrapper under the same benchmark-facing I/O contract:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline direct
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline self-refine
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline scipip-proxy
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline ai-researcher-proxy
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline virsci-proxy
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline ours-eig --runtime-protocol parallel_graph_v2
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline ours-eig-critic-graph-twohead --runtime-protocol parallel_graph_v2
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline ours-eig-critic-calibrated --runtime-protocol parallel_graph_v2
```

The `*-proxy` baselines are diagnostic local approximations. They are useful
for fast development but should not be used as headline paper baselines unless
they are explicitly labeled as proxies.

For DashScope or Qwen models, `ai-researcher` supports an OpenAI-compatible
bridge inside this repo. Treat that bridge as a convenience adapter unless the
configured run exactly follows the upstream AI-Researcher implementation.

Budget guidance:

- `direct` is the cheapest lower-bound baseline.
- `self-refine` is a modest-cost single-agent control.
- `scipip-proxy` is a low-cost structured local baseline.
- `ai-researcher-proxy` is a lightweight local fallback for Qwen-based runs.
- `virsci-proxy` and `ours-eig` are higher-cost multi-agent
  baselines and are better reserved for smaller validation subsets before large
  sweeps.

Run an external baseline through its configured adapter:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline ai-researcher --external-baseline-config configs/external_baselines.example.json
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline scipip --external-baseline-config configs/external_baselines.example.json
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline virsci --external-baseline-config configs/external_baselines.example.json
```

Notes:

- `ai-researcher`, `scipip`, and `virsci` need `--external-baseline-config`.
- `.tmp-baselines/*` upstream clones are local-only and are not tracked by Git.
- `ai-researcher` can run in two modes:
  - exact upstream mode for providers the upstream repo natively supports
  - `openai-compatible-bridge` mode for DashScope/Qwen-style providers
- `scipip` currently runs either through its upstream layout or through the
  configured benchmark-faithful OpenAI-compatible bridge, depending on the
  local config and provider setup.
- `virsci` currently uses the benchmark fixed-topic bridge for benchmark-mode
  evaluation; exact upstream benchmark entrypoints are still not used in the
  paper-facing runs.
- `ai-researcher-proxy`, `scipip-proxy`, and `virsci-proxy` remain local approximations implemented entirely inside this repo.

## Package Structure

- `src/idea_graph/instances.py`
  Typed experiment-instance input objects shared by local JSON and benchmark loaders.
- `src/idea_graph/models.py`
  Graph, node, edge, branch, action, and proposal data models.
- `src/idea_graph/settings.py`
  Runtime settings for OpenAI-compatible multi-agent execution.
- `src/idea_graph/llm.py`
  Lightweight OpenAI-compatible chat client built on the Python standard library.
- `src/idea_graph/agent_backend.py`
  Multi-agent seed generation, action generation, and final synthesis protocols.
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
- Oracle-style fields from the benchmark such as the target paper title,
  gold motivation, and gold method summary are retained in artifacts for
  inspection, but the LLM prompt context now hides them to avoid directly
  leaking the ground-truth paper into generation.
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

## OpenAI-Compatible Multi-Agent Mode

The action model can now be driven by an OpenAI-compatible API instead of the
deterministic placeholder policy.

### Configure

Start from:

```bash
configs/openai_compatible.example.json
```

Fill in:

- your provider's OpenAI-compatible `base_url`
- your `model`
- your `provider` such as `dashscope` or `openai`
- your `reasoning_mode` when the provider exposes a thinking toggle
- your API key via an environment variable such as `OPENAI_API_KEY`

### Run

Use the local sample:

```bash
python scripts/run_pipeline.py --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json
```

Use a benchmark row:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json
```

Add benchmark-native scoring with the same judge configuration:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --native-eval
```

Use the active parallel-v2 EIG protocol:

```bash
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline ours-eig --runtime-protocol parallel_graph_v2 --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json
```

You can also override config values directly:

```bash
python scripts/run_pipeline.py --agent-backend openai-compatible --llm-base-url https://your-provider.example/v1 --llm-model your-model-name
```

Notes:

- The backend uses the OpenAI-style `/chat/completions` protocol for maximum compatibility.
- If an LLM response is malformed or invalid for the graph schema, the engine falls back to the deterministic policy for that step.
- Run artifacts keep backend settings and model traces without storing the API key.
- `api_key_env` must be an environment-variable name such as `DASHSCOPE_API_KEY`, not the API key value itself.
- If you do not want role-specific models yet, keep `role_models` empty. Placeholder names such as `your-strong-reasoning-model` will cause provider errors.
- `reasoning_mode` supports `auto`, `off`, and `on`.
- For DashScope Qwen reasoning-capable models, this client is currently non-streaming. `reasoning_mode=auto` or `off` will automatically send `enable_thinking=false` when needed. `reasoning_mode=on` is intentionally blocked until streaming support is added.
- Some DashScope models are effectively always-thinking, such as `QwQ`, `DeepSeek-R1`, and Qwen variants with `thinking` in the model name. Those are not supported by the current non-streaming client.

### Quick Provider Check

Before running the full pipeline, you can test whether your provider and model
respond correctly:

```bash
python scripts/check_openai_compatible.py --llm-config configs/openai_compatible.example.json
```

For Qwen on DashScope, a typical setup is:

```powershell
$env:DASHSCOPE_API_KEY="your_real_key"
python scripts/check_openai_compatible.py --llm-config configs/openai_compatible.example.json
python scripts/run_pipeline.py --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json
```

You can also override the provider and reasoning mode on the command line:

```bash
python scripts/check_openai_compatible.py --llm-config configs/openai_compatible.example.json --llm-provider dashscope --llm-reasoning-mode off
python scripts/run_pipeline.py --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --llm-provider dashscope --llm-reasoning-mode off
```

If you are using Anaconda Prompt or `cmd.exe`, use `set` instead of `$env:`:

```bat
set DASHSCOPE_API_KEY=your_real_key
"D:\Anaconda\anaconda\python.exe" scripts\check_openai_compatible.py --llm-config configs\openai_compatible.example.json
"D:\Anaconda\anaconda\python.exe" scripts\run_pipeline.py --agent-backend openai-compatible --llm-config configs\openai_compatible.example.json
```

If you want to keep everything on one PowerShell line, use `;` between commands:

```powershell
$env:DASHSCOPE_API_KEY="your_real_key"; & 'D:\Anaconda\anaconda\python.exe' scripts\run_pipeline.py --agent-backend openai-compatible --llm-config configs\openai_compatible.example.json
```

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

1. Harden and validate the exact `AI-Researcher` integration on full benchmark runs.
2. Replace the guarded `SciPIP` and `VirSci` wrappers with better benchmark-native upstream integrations where feasible.
3. Extend AI Idea Bench native scoring to the metrics that need auxiliary assets or batch-level cross-system pools.
4. Add a retrieval stage so keyword-only benchmarks such as `liveideabench` get real literature context.
5. Add human-eval hooks on top of the current local rubric.
