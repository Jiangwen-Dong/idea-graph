# R009 AIIB Launch Plan

This note turns `R009` into a launch-ready plan for the larger
`AI_Idea_Bench_2025` slice.

## Goal

Run the first larger paper-facing automatic batch on the primary benchmark:

- benchmark: `AI_Idea_Bench_2025`
- milestone: `R009`
- methods:
  - `direct`
  - `self-refine`
  - `ai-researcher`
  - `ours-eig`

The purpose is to produce the first serious automatic table on the primary
benchmark before spending further budget on the weaker-context
`LiveIdeaBench` slice.

## Why AIIB First

- it is the primary benchmark in the paper protocol
- the current small-`M1` packet already shows the strongest regime for
  `ours-eig` on `AI_Idea_Bench_2025`
- it is the most benchmark-faithful setting for the paper's scientific-ideation
  claim
- it lets us test the main claim on the cleaner benchmark before paying for the
  harder weak-context expansion

## Fixed 24-Case Slice

The `R009` slice uses a fixed 24-case list:

`13, 1308, 3001, 3883, 4756, 6415, 7909, 8589, 8747, 8905, 9062, 9215, 9376, 9534, 9690, 9849, 10011, 10175, 10336, 10498, 10659, 10821, 11496, 13250`

Selection rule:

- evenly spaced over the sorted benchmark record order
- filtered to cases with usable topic text
- broad enough for a first paper-facing automatic slice without hand-picking
  only favorable examples

## Smoke Subset

Before the full 24-case batch, use this 4-case smoke subset:

`13, 3883, 7909, 9849`

Reason:

- spans early, middle, and late parts of the benchmark order
- includes both visually grounded and language-heavy topics
- keeps the wrapper and judge check low-cost before the full batch

## Method Settings

Use the current main comparison set and method-plan-equivalent settings:

| Method | Max Rounds | Stop When Mature | Extra Config |
|---|---:|---|---|
| `direct` | `1` | yes | none |
| `self-refine` | `1` | yes | none |
| `ai-researcher` | `1` | yes | `configs/external_baselines.qwen.json` |
| `ours-eig` | `5` | yes | none |

Generation backend:

- `--agent-backend openai-compatible`
- `--llm-config configs/openai_compatible.example.json`

## Output Layout

Use dedicated output roots so `R009` stays easy to inspect:

- smoke generation:
  - `outputs/m2_aiib_r009_smoke`
- main generation:
  - `outputs/m2_aiib_r009_main`

Each run will still write its own timestamped artifact directory under the
chosen root.

## Cost-Control Policy

Approximate generation-only token budget from the current small-`M1`
AI-Idea-Bench averages:

- `direct`: about `1.9k` tokens per case
- `self-refine`: about `6.6k`
- `ai-researcher`: about `12.6k`
- `ours-eig`: about `93.7k`

Rough total per benchmark case across all four methods:

- about `115k` tokens

Rough total for `24` AIIB cases:

- about `2.7M` generation tokens before native-judge passes

Therefore `R009` should be staged:

1. smoke generation
2. smoke native-eval check
3. full generation
4. native reevaluation over successful saved runs
5. batch aggregation and paper-table prep

PowerShell execution note:

- do not combine the launch loop with a native-stderr-to-exception policy
- on this benchmark, PDF extraction may emit the harmless warning
  `Ignoring wrong pointing object 40 0 (offset 0)` to stderr
- treat `$LASTEXITCODE` as the true run-status signal and keep a UTF-8 log
  rather than stopping the whole batch on that warning

## Phase 0: Preflight

Run once before smoke generation:

```powershell
python scripts/check_openai_compatible.py --llm-config configs/openai_compatible.example.json
```

Success condition:

- provider returns a valid response
- no API/auth/config failure

## Phase 1: Smoke Generation

Use this PowerShell pattern:

```powershell
$indices = @(13, 3883, 7909, 9849)
$plans = @(
  @{ baseline = 'direct'; max_rounds = 1; external = $false },
  @{ baseline = 'self-refine'; max_rounds = 1; external = $false },
  @{ baseline = 'ai-researcher'; max_rounds = 1; external = $true },
  @{ baseline = 'ours-eig'; max_rounds = 5; external = $false }
)

foreach ($idx in $indices) {
  foreach ($plan in $plans) {
    $cmd = @(
      'scripts/run_pipeline.py',
      '--benchmark', 'ai_idea_bench_2025',
      '--benchmark-index', "$idx",
      '--baseline', $plan.baseline,
      '--agent-backend', 'openai-compatible',
      '--llm-config', 'configs/openai_compatible.example.json',
      '--output-dir', 'outputs/m2_aiib_r009_smoke',
      '--max-rounds', "$($plan.max_rounds)"
    )
    if ($plan.external) {
      $cmd += @('--external-baseline-config', 'configs/external_baselines.qwen.json')
    }
    & python @cmd
  }
}
```

Smoke decision gate:

- at least `15 / 16` runs finish successfully
- no repeated `ai-researcher` wrapper crash pattern
- no repeated malformed LLM-output failure pattern in `ours-eig`
- spot-check `final_proposal.md` on at least:
  - one `ours-eig` run
  - one `self-refine` run
  - one `ai-researcher` run

## Phase 2: Smoke Native-Eval Check

Do not judge every smoke run immediately. First check the strongest single-agent
baseline and our method:

```powershell
$judgeTargets = Get-ChildItem 'outputs/m2_aiib_r009_smoke' -Directory | Where-Object {
  $_.Name -match 'ai-idea-bench-2025' -and (
    (Get-Content ($_.FullName + '\\summary.json') -Raw) -match '\"baseline_name\": \"self-refine\"' -or
    (Get-Content ($_.FullName + '\\summary.json') -Raw) -match '\"baseline_name\": \"ours-eig\"'
  )
}

foreach ($run in $judgeTargets) {
  python scripts/evaluate_run.py --run-dir $run.FullName --native-eval --llm-config configs/openai_compatible.example.json
}
```

Smoke judge gate:

- native scoring completes cleanly on all selected smoke runs
- no systematic topic-alignment failure
- no proposal-format collapse on the judged runs

## Current Status: 2026-04-11

- preflight: passed
  - `python scripts/check_openai_compatible.py --llm-config configs/openai_compatible.example.json`
  - provider responded successfully on the DashScope Qwen endpoint
- smoke generation: passed
  - output root:
    `outputs/m2_aiib_r009_smoke`
  - completed runs:
    `16 / 16`
  - wrapper issue resolved at the launch-script level by judging success with
    `$LASTEXITCODE` rather than stderr presence
- smoke local result summary:
  - `ours-eig`: mean overall `6.00`, mean alignment `4.37`
  - `self-refine`: mean overall `4.65`, mean alignment `2.36`
  - `direct`: mean overall `4.61`, mean alignment `2.36`
  - `ai-researcher`: mean overall `4.46`, mean alignment `2.01`
- smoke native-eval check: passed operationally, mixed scientifically
  - judged methods:
    `self-refine`, `ours-eig`
  - judged runs:
    `8 / 8`
  - `self-refine`: mean native available-average `8.36 / 10`
  - `ours-eig`: mean native available-average `8.07 / 10`
- decision after smoke:
  - do not spend the full `24`-case `R009` generation budget yet
  - first do one narrow `ours-eig` grounding/synthesis cleanup aimed at the
    remaining native AIIB gap, especially experiment-plan fluency and harder
    cases like `3883`
  - the agreed cleanup has now been written up as:
    - `docs/superpowers/specs/2026-04-11-eig-robustness-cleanup-design.md`
    - `docs/superpowers/plans/2026-04-11-eig-robustness-cleanup.md`
  - if that narrow cleanup looks stable, proceed directly to Phase 3 with the
    same fixed `24`-case slice
- post-cleanup narrow diagnosis rerun: completed, mixed
  - output root:
    `outputs/m2_aiib_r009_diagnosis`
  - method:
    `ours-eig`
  - cases:
    `13, 3883, 7909, 9849`
  - native mean:
    `8.29 / 10`
  - local means:
    overall `4.61 / 10`, alignment `2.31 / 10`, graph `9.18 / 10`
  - key win:
    hard case `3883` recovered to native `7.43 / 10`
  - remaining issue:
    several final experiment plans still contain OCR-like or stitched benchmark-evaluation fragments, so the current surface form is not yet stable enough for the full `24`-case spend
  - updated decision at that point:
    do one more narrow grounding/synthesis cleanup focused on noisy evaluation extraction and benchmark-plan rewriting before Phase 3
- follow-up benchmark-safety cleanup: implemented and locally verified
  - benchmark reference packets now sanitize OCR-like GUI dump snippets before
    prompt construction
  - `run_experiment(...)` now stores `literature_grounding` from
    `generation_safe_metadata`, so hidden target-paper summaries do not persist
    into fresh benchmark-mode EIG artifacts
  - metric signal matching is now boundary-aware, removing false `IoU` matches
    inside words like `various`
  - verification:
    `python -m pytest tests/test_benchmark_mode_and_baselines.py tests/test_agent_backend.py tests/test_engine.py tests/test_literature_grounding.py -v`
    with `82 passed`
- current gate before Phase 3:
  - post-fix hard-case `ours-eig` spot check on `3883`: completed
    - output:
      `outputs/m2_aiib_r009_postfix_spotcheck/20260411-175908-ai-idea-bench-2025-3883`
    - the OCR-like experiment-plan corruption is gone
    - saved `literature_grounding` is now prompt-safe
    - the remaining issue is no longer safety or corruption, but genericity:
      the proposal is readable yet still under-anchored on concrete datasets
      and metrics for this hard case
  - next decision:
    - either accept the current cleanup as sufficient and rerun the 4-case
      `R009B` packet on `13, 3883, 7909, 9849`
    - or do one more lightweight grounding-specific patch for benchmark-title
      anchors before that rerun
  - title-anchor rerun: completed
    - output root:
      `outputs/m2_aiib_r009_diagnosis_title_anchor`
    - mean local overall / alignment / graph:
      `5.91 / 4.43 / 9.24`
    - mean native:
      `7.36 / 10`
    - interpretation:
      title-derived anchors substantially improved local benchmark grounding and
      cleaned the hard-case `3883` proposal surface form, but native quality is
      now less stable than in the earlier `R009B` packet
  - current gate decision:
    - do not launch the full `24`-case Phase 3 slice yet
    - first refine benchmark-faithful synthesis so title-derived anchors help
      specific evaluation planning without over-templating the final proposal
  - safe-grounding cleanup after the title-anchor rerun: implemented and locally verified
    - title-derived fallback is now dataset-only
    - title-only metrics have been removed
    - malformed synthesis residue such as `Report satisfied by ...` and
      `...-style baselines` is now cleaned
    - targeted verification:
      `python -m pytest tests/test_literature_grounding.py tests/test_agent_backend.py tests/test_engine.py -q`
      -> `67 passed`
  - next gate:
    - `R009D`: completed
    - mean local overall / alignment / graph:
      `5.26 / 3.48 / 9.28`
    - mean native:
      `8.07 / 10`
    - interpretation:
      this is a better compromise than `R009C`: native quality recovers while
      retaining part of the local grounding gain, but the packet still does not
      cleanly dominate the earlier `R009B` checkpoint
    - current launch decision:
      do not launch the full `24`-case slice yet
    - next required revision before Phase 3:
      one more EIG-focused pass on utility, mature-subgraph selection, and
      final synthesis quality

## Phase 3: Full 24-Case Generation

If smoke passes, run the full slice:

```powershell
$indices = @(13, 1308, 3001, 3883, 4756, 6415, 7909, 8589, 8747, 8905, 9062, 9215, 9376, 9534, 9690, 9849, 10011, 10175, 10336, 10498, 10659, 10821, 11496, 13250)
$plans = @(
  @{ baseline = 'direct'; max_rounds = 1; external = $false },
  @{ baseline = 'self-refine'; max_rounds = 1; external = $false },
  @{ baseline = 'ai-researcher'; max_rounds = 1; external = $true },
  @{ baseline = 'ours-eig'; max_rounds = 5; external = $false }
)

foreach ($idx in $indices) {
  foreach ($plan in $plans) {
    $cmd = @(
      'scripts/run_pipeline.py',
      '--benchmark', 'ai_idea_bench_2025',
      '--benchmark-index', "$idx",
      '--baseline', $plan.baseline,
      '--agent-backend', 'openai-compatible',
      '--llm-config', 'configs/openai_compatible.example.json',
      '--output-dir', 'outputs/m2_aiib_r009_main',
      '--max-rounds', "$($plan.max_rounds)"
    )
    if ($plan.external) {
      $cmd += @('--external-baseline-config', 'configs/external_baselines.qwen.json')
    }
    & python @cmd
  }
}
```

Generation decision gate:

- at least `90 / 96` runs finish successfully
- no baseline has a systematic wrapper failure
- `ours-eig` outputs remain benchmark-grounded on a 4-run spot check
- stop and inspect before the judge pass if there is obvious generic collapse

## Phase 4: Full Native Reevaluation

After generation looks clean, reevaluate the successful saved runs:

```powershell
Get-ChildItem 'outputs/m2_aiib_r009_main' -Directory | ForEach-Object {
  python scripts/evaluate_run.py --run-dir $_.FullName --native-eval --llm-config configs/openai_compatible.example.json
}
```

This staged structure keeps generation and judge cost separated. If generation
quality is obviously bad, stop before paying for the full judge pass.

## Phase 5: Aggregation And Reporting

Immediate reporting target:

- aggregate per-run native metrics:
  - `I2T`
  - `I2I`
  - `IMCQ`
  - `NA`
  - `FA`
  - `FPS`

Important note on `IC`:

- `IC` is a matched-pool batch metric
- the current per-run native scorer correctly marks it as unavailable on single
  runs
- therefore `R009` should treat `IC` as a post-batch aggregation task, not as a
  blocker for generation

## Recommended Interpretation Policy

After `R009`, prioritize these questions:

1. Is `ours-eig` still clearly strongest on AIIB benchmark alignment?
2. Does the native AIIB table now also favor `ours-eig`, or is
   `self-refine` still stronger on the benchmark-native side?
3. Are the gains large enough to justify moving straight to `R010`?
4. Which ablation should be prioritized first if the native table is still
   mixed?

## Recommended Next Action After R009

- If `ours-eig` is competitive or clearly strongest on the AIIB native table:
  move to `R010` after the weak-context decision.
- If `ours-eig` is still locally strong but natively mixed:
  run the highest-value `M5` ablation before the full `LiveIdeaBench` slice.
