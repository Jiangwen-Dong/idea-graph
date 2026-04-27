# Scripts

Public entrypoints stay flat under `scripts/`:

- `run_pipeline.py`
- `check_openai_compatible.py`
- `fetch_ai_idea_bench_2025.py`
- `fetch_liveideabench.py`

Research utilities are grouped by purpose:

- `scripts/analysis/`: result summarization and plotting helpers
- `scripts/data_prep/`: dataset curation, packet building, and export helpers
- `scripts/eval/`: evaluation runners and paper-artifact builders
- `scripts/train/`: critic training, adaptation, and calibration tools

All grouped scripts still resolve the repository root correctly when run from the
repository top level with `python scripts/<group>/<name>.py ...`.
