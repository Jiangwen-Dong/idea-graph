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
- deterministic and OpenAI-compatible multi-agent backends
- graph merge and constrained graph actions
- maturity checks and final subgraph selection
- benchmark loaders and runnable pipeline scripts

The pipeline can now run in two modes:

- `deterministic`
  Useful for local validation, debugging, and reproducible graph mechanics.
- `openai-compatible`
  Uses an OpenAI-style `chat/completions` API protocol so you can target
  ChatGPT-style, Qwen-style, DeepSeek-style, or other compatible providers by
  changing the base URL, API key, and model names.

## Layout

- `docs/implementation-plan.md`
- `docs/benchmarks.md`
- `configs/openai_compatible.example.json`
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

1. Improve prompt quality and validation for benchmark-specific action selection.
2. Add a retrieval stage so keyword-only benchmarks such as `liveideabench` get real literature context.
3. Expand supported graph actions beyond the current safe subset.
4. Implement the comparison baselines from the protocol.
5. Add stronger evaluation logging and replay tooling for LLM traces.
