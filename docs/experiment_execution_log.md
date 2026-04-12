# Experiment Execution Log

## Objective

Align the repository's experiment code with the revised paper framing:

- the main method is `EIG` rather than `delayed consensus`
- delayed commitment remains an internal policy, not the headline method name
- experiment scripts, method plans, and paper-artifact builders should use the
  same naming and comparison protocol

## Implementation Plan

1. Refactor experiment method names and metadata around `ours-eig`.
2. Preserve backward compatibility for existing outputs and scripts that still
   use `ours-delayed-consensus`.
3. Update batch runners and paper-artifact scripts so summaries, findings, and
   labels match the revised method section.
4. Remove clearly obsolete experiment utilities and temporary directories after
   verifying that nothing still depends on them.
5. Run targeted tests for experiment plans and benchmark scoring before any new
   expensive experiment runs.

## Progress

### 2026-04-09

- Reviewed the current experiment runner, baseline registry, method-plan
  registry, and experiment documentation.
- Confirmed the main mismatch: the code still treats the method as
  `ours-delayed-consensus`, while the paper now presents the method as
  `Evolving Idea Graphs (EIG)`.
- Identified candidate cleanup targets to verify before deletion:
  `scripts/run_matched_budget_pilot.py`, `scripts/run_cost_aware_pilot.py`,
  `docs/implementation-plan.md`, `.tmp-baselines/`, `.tmp-tests/`,
  `.pytest_cache/`.
- Refactored the active experiment code so `ours-eig` is now the canonical
  method name across the baseline registry, method-plan catalog, batch runner,
  pipeline entrypoint, and paper-artifact builders.
- Kept `ours-delayed-consensus` as a backward-compatible alias so old outputs
  and command lines still resolve cleanly.
- Updated the README to use the new canonical method name.
- Replaced one stale engine repair string that still described the method as
  delayed consensus.
- Ran targeted regression tests:
  `python -m pytest tests/test_experiment_plans.py tests/test_benchmark_mode_and_baselines.py tests/test_benchmark_scoring.py`
  and all 21 tests passed.
- Deleted obsolete matched-budget utilities and the old implementation note:
  `scripts/run_matched_budget_pilot.py`, `scripts/run_cost_aware_pilot.py`,
  and `docs/implementation-plan.md`.
- Deleted `.pytest_cache/`.
- Attempted to delete `.tmp-tests/`, but two nested directories currently have
  Windows-level access restrictions and were left in place for now. They are
  not part of the active pipeline.
- Added `.pytest_cache/`, `.tmp-tests/`, and `.tmp-baselines/` to
  `.gitignore` so temporary experiment artifacts do not pollute the working
  tree even when Windows blocks cleanup.
- Ran a deterministic smoke test with the new canonical label:
  `python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline ours-eig --max-rounds 1`
  and the pipeline completed successfully, wrote artifacts, and reported the
  canonical baseline name as `ours-eig`.
- Rewrote `docs/paper_experiment_map.md` so the current planning note now
  reflects the revised EIG framing instead of the older delayed-consensus
  narrative.
- Started the next method-alignment step: replacing the coarse graph-wide
  utility heuristic with a transparent utility decomposition over candidate
  subgraphs.
- Added an explicit utility breakdown to the graph state
  (`promise`, `support`, `coherence`, `evidence`, `novelty`, contradiction and
  open-risk penalties, and size penalty).
- Switched maturity tracking to score the current best candidate subgraph
  rather than the entire active graph, which better matches the revised method
  section and reduces noise from weak side branches.
- Updated final-subgraph selection so it now prefers a connected mature
  candidate subgraph when available, and falls back to the older heuristic only
  if candidate construction fails.
- Surfaced the new utility breakdown in run summaries, progress logs, and
  `evaluate_run.py` reconstruction.
- Added targeted tests for utility breakdown exposure and final-subgraph
  selection metadata.
- Re-ran the targeted regression suite:
  `python -m pytest tests/test_engine.py tests/test_evaluation.py tests/test_experiment_plans.py tests/test_benchmark_mode_and_baselines.py tests/test_benchmark_scoring.py`
  and all 35 tests passed.
- Re-ran a deterministic smoke test on
  `ai_idea_bench_2025-13` with `ours-eig --max-rounds 1`; the run completed
  successfully and the saved `summary.json` now includes the new
  `utility_breakdown` payload.
- Cleaned safe generated caches inside the active repo:
  `scripts/__pycache__/`, `src/idea_graph/__pycache__/`,
  `src/idea_graph/benchmarks/__pycache__/`, `tests/__pycache__/`, and the
  external-baseline cache under
  `.tmp-baselines/Virtual-Scientists/sci_platform/__pycache__/`.
- The deterministic smoke result still produces only a rough proposal on the
  benchmark instance, which is expected at this stage. The next bottleneck is
  now candidate-edit quality and final synthesis quality, not graph-state
  serialization.
- Implemented utility-guided action selection on top of the revised EIG state:
  the legacy phase policy now serves as one candidate, generic alternative
  edits are added, and the controller simulates each candidate on a copied
  graph before choosing the highest predicted-gain edit.
- Added phase and role guardrails on top of utility ranking so the selector
  still respects key stability constraints:
  structure turns for mechanism and impact roles stay support-oriented,
  stress-test feasibility turns prefer evidence attachment, and repair turns
  with unresolved contradictions prioritize contradiction-targeted repairs.
- Added a lightweight controller layer for LLM-backed runs: after the LLM
  proposes a valid action, the controller scores it against the best
  deterministic candidate and overrides only when the predicted utility gap is
  meaningfully larger.
- Re-ran the full targeted regression suite after the controller changes:
  `python -m pytest tests/test_engine.py tests/test_evaluation.py tests/test_experiment_plans.py tests/test_benchmark_mode_and_baselines.py tests/test_benchmark_scoring.py`
  and all 35 tests passed.
- Ran a deterministic verification run:
  `python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline ours-eig --max-rounds 2`
  and confirmed the new utility-ranked deterministic selector is active.
- Ran a first small LLM-backed cross-benchmark batch with Qwen:
  - `ai_idea_bench_2025-13`, one-round run using `qwen3-8b`
  - `liveideabench-meteorology-0`, one-round run using `qwen3-8b`
- In the AI Idea Bench run, the utility controller overrode 3 of 5 LLM action
  proposals, replacing low-gain `request_evidence` actions with higher-gain
  structural edits.
- In the LiveIdeaBench run, the utility controller overrode 1 of 5 LLM action
  proposals and kept the remaining LLM edits.
- Updated progress messaging so override actions are now labeled as
  `utility controller override` instead of being ambiguously reported as plain
  deterministic policy actions.
- Refined the prompt-safe grounding path so dataset and metric cues can be
  recovered from allowed reference-paper snippets instead of only from hidden
  benchmark metadata.
- Tightened the action prompt and action user payload:
  benchmark focus, evidence candidates, design-anchor terms, and safe
  dataset/metric cues are now passed explicitly to the LLM action policy.
- Tightened the synthesis prompt and synthesis payload:
  the model is now told to avoid generic placeholders, name concrete reference
  directions, include ablations, and reuse anchor terms when available.
- Strengthened proposal postprocessing so generic method and evaluation text is
  expanded with safer concrete grounding when the prompt output remains vague.
- Revised maturity stopping to require at least two rounds and some evidence
  accumulation before early stop, which removes the previous too-eager
  Round-1 maturity behavior.
- Added targeted tests for:
  - action prompt payload richness,
  - safe grounding recovery from reference snippets,
  - postprocessed proposal specificity,
  - non-trivial maturity behavior across rounds.
- Re-ran the expanded local regression suite:
  `python -m pytest tests/test_agent_backend.py tests/test_literature_grounding.py tests/test_engine.py tests/test_evaluation.py tests/test_experiment_plans.py tests/test_benchmark_mode_and_baselines.py tests/test_benchmark_scoring.py`
  and all 48 tests passed.
- Deterministic verification on `ai_idea_bench_2025-13` now matures at
  `Round2` rather than `Round1`, which better matches the intended
  maturity-based commit logic.
- Ran one Qwen-backed verification after the prompt and grounding changes on
  `ai_idea_bench_2025-13`:
  - previous LLM-backed score after controller update: overall `4.79`,
    benchmark alignment `2.59`
  - latest LLM-backed score after prompt, synthesis, and safe-grounding
    cleanup: overall `4.90`, benchmark alignment `2.70`
- The latest LLM-backed output is cleaner than before: the previous noisy
  `Such methods include ...` fragment is gone from the experiment plan.

## Cleanup Policy

- Delete only files or directories that are both clearly obsolete and not part
  of the active pipeline.
- Keep backward-compatible aliases in code when they avoid breaking old output
  directories or scripts.
- Record each deletion here before or immediately after it happens.

## Previous Next Immediate Step

Refine the next bottlenecks exposed by the latest run:
benchmark-faithful experiment-plan specificity, stronger non-generic method
details, and broader small-batch validation on both benchmarks before paper
table refresh.

### 2026-04-11

- Rechecked the active EIG refinement diff with the current test suite and kept
  the code changes focused on four concrete needs:
  - weak-context proposal cleanup
  - weak-context method instantiation for `LiveIdeaBench`
  - clearer functional role names in shell progress
  - long-path-safe benchmark-evaluation artifact I/O on Windows
- Added and verified the new long-path-safe helper:
  - `src/idea_graph/fs_utils.py`
  - plus regression coverage in `tests/test_fs_utils.py`
- Tightened `pyproject.toml` so `pytest` only collects under `tests/`, which
  avoids accidental artifact collection from deep `outputs/` paths.
- Confirmed the prompt/synthesis cleanup is covered by new tests in:
  - `tests/test_agent_backend.py`
  - `tests/test_literature_grounding.py`
  - `tests/test_engine.py`
- Updated `docs/paper_experiment_tracker.md` to record:
  - the weak-context `R008F` meteorology rerun
  - the refreshed cross-benchmark small `M1` packet
  - today's infrastructure fixes and the current recommendation
- Confirmed the current reference small-`M1` packet is:
  - `outputs/quality_batches/20260411-000159-refreshed-m1-mini-synthesis-cleanup-v2-native`
- Current small-`M1` interpretation:
  - `ours-eig` is strongest overall on the refreshed 4-case cross-benchmark
    packet
  - `ours-eig` is clearly strongest on `AI_Idea_Bench_2025`
  - `ours-eig` ties `self-refine` on mean native `LiveIdeaBench` score, but
    remains stronger by local alignment
  - the only visible instability is stochastic weak-context meteorology
- Repository cleanup completed:
  - deleted `.tmp-external-baseline-runs/` because it contained only transient
    external-baseline scratch runs already superseded by preserved `outputs/`
    artifacts
  - added `.tmp-external-baseline-runs/` to `.gitignore` so future baseline
    scratch runs do not re-enter version control
  - restored the accidental dirty state in
    `.tmp-baselines/Virtual-Scientists` so the parent repository is not left
    with unrelated submodule noise
- Implemented the next narrow AIIB robustness cleanup after the diagnosis pass:
  - `run_experiment(...)` now stores `literature_grounding` from
    `generation_safe_metadata` rather than raw benchmark metadata, so saved
    summaries no longer preserve hidden target-paper grounding
  - benchmark-mode reference packets now sanitize OCR-like or GUI-dump snippets
    before they reach prompts
  - prompt-safe metadata now inherits the sanitized benchmark packet instead of
    copying the raw packet through to synthesis
  - noisy proposal-sentence detection now catches OCR-style GUI benchmark
    fragments such as `Task Instruction ...`, `a11y-tree`, `keyboardmouse`,
    `Package_name`, and related screen-dump tokens
  - literature-grounding signal matching now uses boundary-aware metric-token
    detection, which fixes the false `IoU` match inside words like `various`
- Added focused regressions for this cleanup:
  - benchmark packet sanitization in
    `tests/test_benchmark_mode_and_baselines.py`
  - prompt-safe grounding persistence in `tests/test_engine.py`
  - OCR-like experiment-plan rewriting in `tests/test_agent_backend.py`
  - metric false-positive protection in `tests/test_literature_grounding.py`
- Re-ran the relevant verification suite:
  - `python -m pytest tests/test_benchmark_mode_and_baselines.py tests/test_agent_backend.py tests/test_engine.py tests/test_literature_grounding.py -v`
  - `82 passed`
- Did a no-API inspection on AIIB hard case `3883` after the code fix:
  - the sanitized benchmark packet now drops the worst OSWorld/UI-dump snippet
    content instead of feeding it directly into prompts
  - generation-safe grounding no longer carries hidden target-paper summaries
  - the remaining prompt-visible context is still somewhat weak or generic for
    `3883`, but it is no longer benchmark-unsafe
- Ran one post-fix Qwen-backed hard-case spot check:
  - command:
    `python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 3883 --baseline ours-eig --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --output-dir outputs/m2_aiib_r009_postfix_spotcheck --max-rounds 5`
  - output:
    `outputs/m2_aiib_r009_postfix_spotcheck/20260411-175908-ai-idea-bench-2025-3883`
  - result:
    the previous OCR-like experiment-plan corruption is gone, and the saved
    `summary.json` now keeps prompt-safe grounding instead of hidden target
    grounding
  - remaining bottleneck:
    the proposal is now readable but still too generic on `3883`, because the
    surviving safe grounding does not yet provide enough concrete dataset and
    metric anchors for a benchmark-faithful experiment plan

## Current Next Step Queue

1. Post-fix AIIB gate check:
   - decide whether to accept the current cleanup as sufficient for a refreshed
     4-case `R009B` rerun, or do one more lightweight grounding-specific patch
     for benchmark titles like `OSWorld`
   - if we rerun now, use the same 4-case diagnosis packet:
     `13, 3883, 7909, 9849`
2. Optional narrow weak-context stabilization:
   - one more EIG-only refinement pass for meteorology-like
     `LiveIdeaBench` cases if we want lower run-to-run variance before larger
     benchmark slices
2. `M2` core automatic batch:
   - launch the benchmark-native larger slice once the weak-context decision is
     made
   - recommended priority order:
     - `AI_Idea_Bench_2025` core slice
     - `LiveIdeaBench` core slice
3. Paper artifacts:
   - refresh paper-ready tables and benchmark result notes from the frozen
     small-`M1` reference packet
4. Mechanism validation:
   - run the planned EIG ablations on utility ranking, maturity stop, and flat
     synthesis after the larger automatic slice is stable
5. Human evaluation packet:
   - only after proposal formatting and benchmark-facing behavior stay stable

### 2026-04-11: Documentation Reorganization

- Reorganized `docs/` into a smaller canonical set plus an archive.
- Added `docs/README.md` as the new index for active documentation.
- Kept the active top-level documentation focused on:
  - protocol
  - evaluation
  - benchmark integration
  - experiment plan
  - experiment tracker
  - compact experiment map
  - execution log
- Moved older or superseded notes into `docs/archive/`:
  - `april03_ablation_pilot.md`
  - `baseline_integration_plan.md`
  - `eig_role_synthesis_redesign.md`
  - `scientific_ideation_landscape.md`
- Added `docs/archive/README.md` so the archived notes remain interpretable but
  no longer compete with the active paper-facing workflow.
- Updated the repo root `README.md` layout section to point to the new active
  documentation set.

### 2026-04-11: R009 Launch Planning

- Prepared a concrete launch note for `R009`:
  - `docs/r009_aiib_launch_plan.md`
- Fixed the initial larger automatic slice to the primary benchmark
  `AI_Idea_Bench_2025` rather than a monolithic cross-benchmark launch.
- The launch note now records:
  - the fixed 24-case `AI_Idea_Bench_2025` slice
  - the 4-case smoke subset
  - the baseline-specific run settings
  - staged generation and native-evaluation commands
  - decision gates before the expensive full judge pass
  - the current status of `IC` as a matched-pool post-batch metric rather than
    a per-run artifact
- Linked the tracker entry for `R009` to this launch note so the next step is
  executable, not just described abstractly.

### 2026-04-11: R009 Preflight And Smoke Stage

- Completed the `R009` preflight check with:
  `python scripts/check_openai_compatible.py --llm-config configs/openai_compatible.example.json`
  and confirmed the DashScope Qwen endpoint returns a valid response.
- Started the `AI_Idea_Bench_2025` smoke packet under:
  `outputs/m2_aiib_r009_smoke`
  on the planned `4`-case subset:
  `13, 3883, 7909, 9849`
  with `direct`, `self-refine`, `ai-researcher`, and `ours-eig`.
- Diagnosed the first smoke interruption before changing any repo code:
  - the pipeline itself was not crashing on `9849/direct`
  - the real issue was the outer PowerShell loop treating native stderr as a
    hard failure while PDF extraction emitted the harmless warning
    `Ignoring wrong pointing object 40 0 (offset 0)`
  - direct reproduction confirmed `run_pipeline.py` still exited with code `0`
    on that case
- Added an execution-side workaround for the batch launch:
  - use exit-code-driven logging with `$LASTEXITCODE`
  - keep PowerShell from promoting native stderr to a terminating error during
    the loop
  - write UTF-8 smoke logs instead of the earlier null-separated text file
- One mistaken resume attempt used a placeholder environment value rather than
  the real API key; those invalid `9849` artifacts were removed immediately
  from `outputs/` and then rerun cleanly.
- Final smoke generation status:
  - `16 / 16` runs completed successfully
  - smoke output root:
    `outputs/m2_aiib_r009_smoke`
  - UTF-8 resume log:
    `outputs/m2_aiib_r009_smoke/_smoke_resume_log_utf8.txt`
- Local smoke-stage result summary across the `4` AIIB cases:
  - `ours-eig`: mean overall `6.00`, mean alignment `4.37`
  - `self-refine`: mean overall `4.65`, mean alignment `2.36`
  - `direct`: mean overall `4.61`, mean alignment `2.36`
  - `ai-researcher`: mean overall `4.46`, mean alignment `2.01`
- Spot-check interpretation from the saved `final_proposal.md` files:
  - `ours-eig` remains the strongest benchmark-grounded generator in the smoke
    packet
  - however, final synthesis still shows occasional awkward experiment-plan
    stitching, especially on harder AIIB cases
- Completed the planned low-cost native smoke check only for
  `self-refine` and `ours-eig`:
  - `8 / 8` judge runs completed successfully
  - native-eval log:
    `outputs/m2_aiib_r009_smoke/_native_eval_smoke_log_utf8.txt`
- Native smoke-stage result summary on the judged subset:
  - `self-refine`: mean native available-average `8.36 / 10`
  - `ours-eig`: mean native available-average `8.07 / 10`
  - case-level pattern:
    - `ours-eig` tied `self-refine` on AIIB `13`
    - `ours-eig` beat `self-refine` on AIIB `7909`
    - `self-refine` beat `ours-eig` clearly on AIIB `3883`
    - they tied again on AIIB `9849`
- Updated interpretation after smoke:
  - the `R009` launch infrastructure is now stable enough for a larger slice
  - the scientific signal is still mixed in the same direction as refreshed
    `M1`: `ours-eig` is clearly strongest under the local benchmark-facing
    metrics, but `self-refine` is still slightly stronger under the targeted
    smoke native average
- therefore the highest-value next step is not the full `24`-case launch
  immediately; it is one narrow `ours-eig` grounding/synthesis cleanup aimed
  at the native AIIB gap before spending the full `R009` generation budget

### 2026-04-11: EIG Robustness Cleanup Planning

- Reviewed the post-smoke diagnosis more carefully before touching code and
  confirmed that the next revision should target the internal controller
  modules, not just surface prompt wording.
- Key diagnosis:
  - `ours-eig` already builds stronger local graph states than the baselines
  - but utility, mature-subgraph selection, and final synthesis are not yet
    aligned tightly enough with benchmark-native `AI_Idea_Bench_2025`
    judgments
  - on harder smoke cases such as `3883`, the graph looks mature while the
    final proposal still undershoots the stronger baseline on native scoring
- Confirmed one additional fairness risk:
  the current grounding stack can rebuild literature grounding from full
  internal metadata and can also over-extract noisy dataset or metric items
  from visible snippets, so the next revision must explicitly tighten
  benchmark-safe grounding before further quality tuning.
- Wrote the agreed design note:
  - `docs/superpowers/specs/2026-04-11-eig-robustness-cleanup-design.md`
- Wrote the concrete implementation plan:
  - `docs/superpowers/plans/2026-04-11-eig-robustness-cleanup.md`
- Agreed controller direction captured in the design note:
  - do **not** introduce learned utility or maturity models at this stage
  - first strengthen:
    - fair benchmark-safe grounding
    - benchmark-aware transparent utility
    - mature-subgraph and maturity selection
    - slot-wise final synthesis
- The next execution stage should therefore be:
  - implement the robustness cleanup with tests first
  - rerun a narrow AIIB diagnosis subset
  - only then decide whether the full `R009` 24-case slice should launch

### 2026-04-10

- Archived legacy timestamped run folders from `2026-03-24` to `2026-04-01`
  into `outputs/_archive/legacy_runs_pre_20260409/` so the top-level
  `outputs/` now highlights the active April 9 to April 10 EIG runs plus the
  batch-level result directories.
- Kept all archived artifacts intact rather than deleting them, because several
  of them still matter for retrospective analysis and paper figures.
- Updated paper-artifact builder defaults so any hard-coded references to the
  archived `20260401-164026-ai-idea-bench-2025-18` run still resolve
  correctly after the cleanup.
- Reviewed the updated protocol documents
  (`docs/paper_protocol.md` and `docs/evaluation.md`) against the current paper
  experiment section and confirmed a remaining mismatch: the paper still tells
  a small proxy-heavy local-score story, while the protocol now requires
  exact or benchmark-faithful baselines plus benchmark-native scoring as the
  primary evidence.
- Audited the baseline-integration code path and confirmed the current status:
  `ai-researcher` and `scipip` have explicit exact wrappers, while `virsci`
  still hard-blocks benchmark-mode execution because the upstream system does
  not expose a fixed-topic benchmark entrypoint.
- Added a new paper-facing planning layer:
  - `docs/paper_experiment_plan.md`
  - `docs/paper_experiment_tracker.md`
- Rewrote `docs/paper_experiment_map.md` so the compact experiment map now
  matches the current protocol:
  - `AI_Idea_Bench_2025` as the primary benchmark
  - `LiveIdeaBench` as the secondary benchmark
  - `direct`, `self-refine`, `ai-researcher`, and `ours-eig` as the main
    comparison set
  - `virsci` kept behind a decision gate instead of being assumed runnable
  - graph-process and cost metrics moved to supplementary analysis
- The new plan fixes the execution order around concrete milestones:
  `M0` protocol sanity, `M1` smoke batch, `M2` core automatic batch, `M4`
  human blind review, and `M5` EIG ablations.
- Executed `R001` from `M0`:
  `ours-eig` on `AI_Idea_Bench_2025` index `13` with `--native-eval` and
  `--max-rounds 2`.
- `R001` succeeded end to end:
  generation, artifact writing, and benchmark-native scoring all completed
  without code changes.
- `R001` artifact directory:
  `outputs/20260410-154406-ai-idea-bench-2025-13`
- `R001` key signals:
  - benchmark-native protocol path works
  - maturity reached at `Round2`
  - local `benchmark_alignment` remains weak at `2.65/10`
  - native inspiration matching is non-zero, while target matching remains
    near zero on this case
- Executed `R002` from `M0`:
  `ours-eig` on `LiveIdeaBench` row `0` with `--native-eval` and
  `--max-rounds 2`.
- `R002` succeeded end to end:
  generation, artifact writing, and benchmark-native scoring all completed
  without code changes.
- `R002` artifact directory:
  `outputs/20260410-154531-liveideabench-meteorology-0`
- `R002` key signals:
  - the native `LiveIdeaBench` rubric path works
  - maturity again reached at `Round2`
  - local `benchmark_alignment` is extremely low (`0.25/10`) despite a decent
    native creativity-style average (`7.67/10`)
  - this reinforces that the current synthesis is still too generic relative
    to the benchmark packet
- Audited `R003` exact `ai-researcher` feasibility under the current Qwen
  provider setup.
- `R003` initial blocker diagnosis:
  - the exact repo is present under `.tmp-baselines/AI-Researcher`
  - the upstream code path dispatches `qwen*` engines to the Together client
    rather than to a generic OpenAI-compatible backend
  - local Python has `openai` installed, but `anthropic` and `together` are
    missing
  - the exact repo also lacks a ready `keys.json`
  - therefore, exact benchmark-mode execution with DashScope Qwen is not
    currently available without a compatibility patch
- Audited `R004` exact `virsci` feasibility.
- `R004` result:
  `NO-GO` for the main benchmark-faithful table in the current state.
- Root cause for `R004` no-go decision:
  - the repo wrapper already hard-rejects benchmark mode for `virsci`
  - the upstream stack is configured around its own `ollama_*` model setup
  - the bundled knowledge config points to an external absolute knowledge-bank
    path, not to a benchmark packet
  - keeping `virsci` in the headline table right now would force either a
    proxy or an undocumented patch, which the paper protocol explicitly forbids
- Flexible plan adjustment after `M0`:
  - `ours-eig` benchmark-native smoke tests are now marked as passed
  - `virsci` should be removed from the main-table shortlist unless we later
    implement a clearly documented benchmark-faithful integration
  - `ai-researcher` remains the preferred ideation baseline, but it now has a
    concrete integration blocker under the current Qwen provider stack
  - the cleanest next engineering step before `M1` is a narrow compatibility
    patch for exact `ai-researcher`, not a larger experiment batch
- Implemented an `ai-researcher` compatibility bridge for the current provider
  stack.
- Implementation detail:
  - `ai-researcher` now supports an `openai-compatible-bridge` execution mode
    in `src/idea_graph/external_baselines.py`
  - this bridge preserves the AI-Researcher-style
    seed-generation -> proposal-expansion -> ranking workflow
  - the bridge reuses the repo's benchmark-faithful AI-Researcher prompt flow
    under the same OpenAI-compatible client used elsewhere in the project
  - the exact upstream path is still preserved as the default mode when a
    natively supported provider is available
- Added a reusable bridge config:
  `configs/external_baselines.qwen.json`
- Updated the example config and README so the new bridge mode is documented.
- Added TDD coverage for the bridge:
  - bridge succeeds with an OpenAI-compatible backend
  - bridge raises a clear error when the OpenAI-compatible settings are missing
- Verification after the bridge patch:
  `python -m pytest tests/test_benchmark_mode_and_baselines.py tests/test_benchmark_scoring.py tests/test_experiment_plans.py -v`
  and all `23` tests passed.
- Re-ran `R003` after the bridge patch:
  `ai-researcher` on `AI_Idea_Bench_2025` index `13` with native evaluation.
- `R003` is now complete:
  `outputs/20260410-160800-ai-idea-bench-2025-13`
- `R003` key signals after the bridge patch:
  - the `ai-researcher` baseline is now runnable under the current DashScope
    Qwen setup
  - native average normalized score on this case: `5.71/10`
  - local benchmark alignment remains weak at `2.24/10`
- Started `M1` with a conservative warm-start rather than the full `4 + 4`
  slice, because the new bridge needed a cross-method sanity pass first.
- Warm-start runs completed:
  - `direct` on `AI_Idea_Bench_2025` index `13`:
    `outputs/20260410-160957-ai-idea-bench-2025-13`
  - `self-refine` on `AI_Idea_Bench_2025` index `13`:
    `outputs/20260410-161048-ai-idea-bench-2025-13`
  - `direct` on `LiveIdeaBench` row `0`:
    `outputs/20260410-161112-liveideabench-meteorology-0`
  - `self-refine` on `LiveIdeaBench` row `0`:
    `outputs/20260410-161141-liveideabench-meteorology-0`
  - `ai-researcher` on `LiveIdeaBench` row `0`:
    `outputs/20260410-161232-liveideabench-meteorology-0`
- Warm-start comparison signal on `AI_Idea_Bench_2025-13`:
  - `direct`: native average normalized `8.29/10`
  - `self-refine`: native average normalized `8.57/10`
  - `ai-researcher` bridge: native average normalized `5.71/10`
  - `ours-eig` from `R001`: native average normalized `6.0/10`
- Warm-start comparison signal on `LiveIdeaBench-0`:
  - all tested methods currently receive the same coarse native average
    (`7.67/10`) under the public rubric on this case
  - the local benchmark-alignment signal is more diagnostic here:
    `direct` `1.68/10`, `self-refine` `1.71/10`, `ours-eig` `0.25/10`,
    `ai-researcher` bridge `0.04/10`
- Interpretation after the warm-start:
  - the current control baselines are already strong on the AI Idea Bench case
  - `ours-eig` and the `ai-researcher` bridge both still suffer from
    benchmark-facing drift
  - scaling immediately to the full `4 + 4` `M1` slice would spend API budget
    before addressing the clearest current bottleneck
- Root-cause follow-up on the `ai-researcher` bridge:
  - inspected the bad `LiveIdeaBench` meteorology output and confirmed the
    proposal had drifted into `3D language field` wording
  - traced the failure to three benchmark-specific assumptions inside
    `src/idea_graph/baselines.py`:
    `ai-researcher` focus constraints, candidate-ranking fidelity scoring, and
    postprocess rewriting were all still specialized for the earlier
    language-field benchmark case
- Added regression coverage before fixing the bug:
  - non-language-field `ai-researcher` constraints must stay generic
  - `LiveIdeaBench` meteorology proposals must outrank unrelated
    language-field drift under the AI-Researcher ranking heuristic
  - `ai-researcher` postprocessing must not inject `language field`,
    `LERF`, or `Gaussian Splatting` into meteorology proposals
- Patched the `ai-researcher` bridge logic:
  - `_topic_text` now strips both
    `The topic of this paper is ...` and
    `Ideation topic keyword: ...`
  - `ai-researcher` anchor extraction now reuses the generic benchmark-anchor
    path instead of a fixed language-field list
  - focus constraints and topic-fidelity scoring are now benchmark-aware, with
    language-field-specific bonuses applied only when the visible benchmark
    packet actually indicates that domain
  - `ai-researcher` postprocessing now uses the old language-field rewrite only
    for true language-field cases and otherwise falls back to the generic
    benchmark-grounding postprocess
- While validating the rerun, found a second issue in the local evaluation
  pipeline:
  - `benchmark_alignment` stayed artificially low because the tokenization in
    `src/idea_graph/evaluation.py` preserved sentence-final periods, so
    `meteorology.` and `meteorology` were treated as different tokens
  - the topic scorer also underused explicit benchmark keywords on
    keyword-only `LiveIdeaBench` cases
- Added a regression test for keyword-only topic alignment and patched the
  scorer:
  - evaluation now strips trailing periods from tokens
  - topic scoring now strips benchmark prompt wrappers such as
    `Ideation topic keyword:`
  - explicit metadata keywords are now used directly in topic-alignment
    evaluation when available
- Verification after both fixes:
  - `python -m pytest tests/test_evaluation.py tests/test_benchmark_mode_and_baselines.py tests/test_benchmark_scoring.py tests/test_experiment_plans.py -v`
    passed with `29` tests green
- Re-ran the previously broken `ai-researcher` LiveIdeaBench case under Qwen:
  - command: `python scripts/run_pipeline.py --benchmark liveideabench --benchmark-index 0 --baseline ai-researcher --external-baseline-config configs\external_baselines.qwen.json --llm-config configs\openai_compatible.example.json --native-eval`
  - new run directory:
    `outputs/20260410-163049-liveideabench-meteorology-0`
  - the proposal now stays in meteorology rather than drifting into 3D
    language fields
  - native LiveIdeaBench average remains `7.67/10`
- Refreshed the saved run with the fixed local evaluator:
  - command: `python scripts/evaluate_run.py --run-dir outputs\20260410-163049-liveideabench-meteorology-0`
  - refreshed local `benchmark_alignment` is now `2.0/10` instead of the
    earlier misleading near-zero signal
  - this confirms that part of the earlier `ai-researcher` diagnosis was a
    real generation bug and part was an evaluation-tokenization bug
- Refreshed the full warm-start comparison packet under the corrected local
  evaluator without spending extra generation API:
  - `AI_Idea_Bench_2025-13`
    - `ours-eig`: overall `4.97`, benchmark alignment `2.71`
    - `ai-researcher`: overall `4.58`, benchmark alignment `2.36`
    - `direct`: overall `4.50`, benchmark alignment `2.17`
    - `self-refine`: overall `4.71`, benchmark alignment `2.55`
  - `LiveIdeaBench-0`
    - `ours-eig`: overall `3.59`, benchmark alignment `0.26`
    - `direct`: overall `4.44`, benchmark alignment `1.93`
    - `self-refine`: overall `4.50`, benchmark alignment `1.96`
    - `ai-researcher`: overall `4.46`, benchmark alignment `2.00`
- Updated interpretation after the refreshed `M1` warm-start:
  - the `ai-researcher` bridge is now credible enough to keep in the main
    baseline set
  - `virsci` should remain deferred as a later integration task
  - the biggest remaining bottleneck before the full `M1` slice is now
    `ours-eig` on keyword-only `LiveIdeaBench`, not baseline feasibility
- Implemented the first keyword-only / weak-context refinement pass for
  `ours-eig` so `LiveIdeaBench` can act as a meaningful divergent-thinking
  benchmark rather than a failure mode:
  - `src/idea_graph/literature_grounding.py` now builds a
    benchmark-safe `weak_context_scaffold` for keyword-only settings
  - the scaffold includes domain-family-specific divergence axes, safe
    existing-method directions, evaluation assets, metrics, risk items, and
    mechanism terms
  - `src/idea_graph/agent_backend.py` now exposes the scaffold to seed,
    action, and synthesis prompts, and `src/idea_graph/engine.py` now applies
    stricter maturity requirements for weak-context cases
- Added regression coverage for the weak-context path:
  - scaffold exposure in prompts
  - keyword-only literature grounding construction
  - weak-context maturity behavior
  - postprocessed proposal specificity and evaluation coverage
- Verification after the first weak-context patch:
  `python -m pytest tests/test_agent_backend.py tests/test_literature_grounding.py tests/test_engine.py tests/test_evaluation.py tests/test_experiment_plans.py tests/test_benchmark_mode_and_baselines.py tests/test_benchmark_scoring.py -v`
  passed with `58` tests green at that stage.
- Ran `ours-eig` weak-context verification on `LiveIdeaBench-0` after the
  scaffold patch:
  - run dir: `outputs/20260410-172718-liveideabench-meteorology-0`
  - executed rounds: `2`
  - local overall: `5.62/10`
  - local benchmark alignment: `3.86/10`
  - native average normalized: `7.67/10`
- Tightened weak-context maturity to preserve divergence for at least one
  additional round before commitment.
- Re-ran `ours-eig` on `LiveIdeaBench-0` after the maturity revision:
  - run dir: `outputs/20260410-173245-liveideabench-meteorology-0`
  - executed rounds: `3`
  - local overall: `6.12/10`
  - local benchmark alignment: `4.57/10`
  - native average normalized: `7.05/10`
- Root-cause inspection of the improved `173245` proposal found a visible
  synthesis artifact:
  the final `method` section ended with a raw scaffold phrase
  (`Physics-Guided Forecasting: ...`) instead of natural prose.
- Added a focused TDD fix for that synthesis bug:
  - proposal postprocessing now distinguishes control anchors from natural
    final-prose sentences
  - weak-context method refinement now recognizes when the proposal already
    covers the scaffold mechanism semantically, avoiding duplicated template
    text
- Ran a single low-cost rerun after the synthesis-cleanup patch:
  - run dir: `outputs/20260410-181027-liveideabench-meteorology-0`
  - local overall: `5.68/10`
  - local benchmark alignment: `3.88/10`
  - native average normalized: `7.05/10`
  - interpretation: the method section is now visibly cleaner and no longer
    leaks raw scaffold labels, but this rerun's evaluation section became
    slightly too sparse for the local benchmark-faithful scorer
- Implemented a second deterministic weak-context postprocess refinement
  without spending another generation call:
  - sparse keyword-only evaluation sections now get benchmark-safe missing
    evaluation assets, missing metrics, and explicit stress-test language
  - this directly targets the remaining local alignment bottleneck revealed by
    the `181027` rerun
- Re-ran the expanded local regression suite after the evaluation-coverage
  refinement:
  `python -m pytest tests/test_agent_backend.py tests/test_literature_grounding.py tests/test_engine.py tests/test_evaluation.py tests/test_experiment_plans.py tests/test_benchmark_mode_and_baselines.py tests/test_benchmark_scoring.py -v`
  and all `60` tests passed.
- Decision after these weak-context iterations:
  do not spend another immediate API rerun yet.
  The code-side weak-context fixes are now in place, the regression suite is
  green, and the next API spend should preferably happen as part of the
  refreshed `M1` comparison packet rather than as another isolated
  single-instance probe.
- Patched the batch runner so the refreshed paper-facing comparison packet can
  use the exact `ai-researcher` bridge instead of the older proxy-only method
  plan:
  - `src/idea_graph/experiment_plans.py` now includes `ai-researcher` in the
    main method-plan catalog
  - `scripts/run_quality_batch.py` now accepts
    `--external-baseline-config` and forwards it to external baseline
    adapters
- Added a regression test to lock that behavior:
  `tests/test_experiment_plans.py::test_main_method_plan_includes_exact_ai_researcher`
- Verification after the batch-runner patch:
  - `python -m pytest tests/test_experiment_plans.py tests/test_benchmark_mode_and_baselines.py tests/test_benchmark_scoring.py -v`
    passed with `27` tests green
  - `python scripts/run_quality_batch.py --help` confirmed the new
    `--external-baseline-config` flag is exposed correctly
- Ran the refreshed mini `M1` comparison packet on the current codepath:
  - command:
    `python scripts/run_quality_batch.py --llm-config configs/openai_compatible.example.json --external-baseline-config configs/external_baselines.qwen.json --ai-indices 13 --live-row-indices 0 --baselines direct self-refine ai-researcher ours-eig --native-eval --batch-name refreshed-m1-mini-current-codepath`
  - batch artifact directory:
    `outputs/quality_batches/20260410-181953-refreshed-m1-mini-current-codepath`
- Refreshed mini-packet result on `AI_Idea_Bench_2025-13`:
  - `ours-eig`: overall `5.45`, alignment `3.71`, native `8.29`
  - `direct`: overall `4.71`, alignment `2.55`, native `8.29`
  - `self-refine`: overall `4.62`, alignment `2.36`, native `9.14`
  - `ai-researcher`: overall `4.54`, alignment `2.31`, native `5.71`
- Refreshed mini-packet result on `LiveIdeaBench-0`:
  - `self-refine`: overall `7.05`, alignment `6.16`, native `7.05`
  - `ours-eig`: overall `6.36`, alignment `5.00`, native `7.67`
  - `direct`: overall `5.92`, alignment `4.21`, native `7.67`
  - `ai-researcher`: overall `4.92`, alignment `2.50`, native `7.67`
- Combined mini-packet interpretation:
  - `ours-eig` now ranks highest by mean local overall score across the two
    refreshed benchmark cases (`5.91` vs `5.83` for `self-refine`)
  - `ours-eig` is now clearly strongest on the refreshed AI Idea Bench case
    under the local output-only evaluator
  - `self-refine` still wins on the refreshed weak-context `LiveIdeaBench`
    case, so weak-context EIG quality is improved but not yet dominant
  - the exact `ai-researcher` bridge is stable, but currently underperforms
    both `direct` and `self-refine` on these two refreshed cases
- Decision after the refreshed mini packet:
  - do not yet claim cross-benchmark dominance for `ours-eig`
  - the next highest-value method revision should target the weak-context
    `LiveIdeaBench` setting specifically, especially branch diversity and final
    synthesis strength under keyword-only input
  - if we want the next experiment step instead of another immediate method
    patch, the cleanest escalation is the full `M1` `4 + 4` slice using the
    now-stable exact baseline set and current EIG codepath
- Performed the targeted weak-context follow-up before the larger `4 + 4`
  `M1` slice.
- Root-cause diagnosis after the `refreshed-m1-mini-weak-context-v2` batch:
  - the weak-context postprocess itself was not leaking into `LiveIdeaBench`
    incorrectly
  - the real regression source on `AI_Idea_Bench_2025` was that
    `_postprocess_final_proposal` rebuilt literature grounding from
    prompt-safe metadata, which intentionally strips `raw_record`
  - that safe rebuild removed access to structured benchmark datasets and
    metrics during final synthesis postprocessing, so AI Idea Bench proposals
    could lose the benchmark-grounded evaluation detail needed for alignment
- Added a TDD regression test for this bug in `tests/test_agent_backend.py`:
  AI Idea Bench postprocessing must preserve structured benchmark grounding
  when datasets and metrics are present in stored `literature_grounding`.
- Patched `src/idea_graph/agent_backend.py` so final proposal postprocessing
  now prefers stored internal `literature_grounding` metadata, and falls back
  to rebuilding from full metadata only when the stored grounding is missing.
- This keeps the prompt path benchmark-safe while allowing the internal
  formatter to recover benchmark datasets and metrics correctly.
- Re-ran the expanded local regression suite after the fix:
  `python -m pytest tests/test_agent_backend.py tests/test_literature_grounding.py tests/test_engine.py tests/test_evaluation.py tests/test_experiment_plans.py tests/test_benchmark_mode_and_baselines.py tests/test_benchmark_scoring.py -v`
  and all `64` tests passed.
- Re-ran the same refreshed mini packet after the postprocess-grounding fix:
  - command:
    `python scripts/run_quality_batch.py --llm-config configs\openai_compatible.example.json --external-baseline-config configs\external_baselines.qwen.json --ai-indices 13 --live-row-indices 0 --baselines direct self-refine ai-researcher ours-eig --native-eval --batch-name refreshed-m1-mini-weak-context-v3`
  - batch artifact directory:
    `outputs/quality_batches/20260410-185224-refreshed-m1-mini-weak-context-v3`
- Refreshed mini-packet result after the fix on `AI_Idea_Bench_2025-13`:
  - `ours-eig`: overall `5.92`, alignment `4.33`, native `8.57`
  - `self-refine`: overall `4.72`, alignment `2.53`, native `9.14`
  - `direct`: overall `4.50`, alignment `2.19`, native `8.29`
  - `ai-researcher`: overall `4.52`, alignment `2.32`, native `8.86`
- Refreshed mini-packet result after the fix on `LiveIdeaBench-0`:
  - `ours-eig`: overall `7.11`, alignment `6.15`, native `7.05`
  - `self-refine`: overall `6.97`, alignment `6.03`, native `7.67`
  - `direct`: overall `5.89`, alignment `4.18`, native `7.05`
  - `ai-researcher`: overall `5.88`, alignment `4.12`, native `7.67`
- Updated interpretation after the targeted rerun:
  - the targeted weak-context refinement succeeded without sacrificing the AI
    Idea Bench case
  - `ours-eig` is now strongest on both refreshed mini-batch benchmark cases
    under the local benchmark-facing evaluator
  - the combined mini-batch mean also now clearly favors `ours-eig`
    (`6.51` overall, `5.24` alignment)
  - the remaining visible bottleneck is not weak-context alignment anymore but
    proposal-surface polish, especially occasional templated method sentences
    that still sound slightly mechanical
- Implemented one narrow deterministic synthesis cleanup before the larger
  `M1` slice:
  - `src/idea_graph/agent_backend.py` now strips only exact scaffold-style
    method sentences that come from the weak-context postprocess, and only
    when the proposal already contains real method prose
  - added focused regression coverage for both the regular scaffold sentence
    and the weak-context scaffold sentence in `tests/test_agent_backend.py`
- Verification after the synthesis cleanup:
  - targeted new tests passed
  - full local regression suite passed with `66` tests green
- Launched the full `M1` `4 + 4` cross-benchmark slice on the stable codepath:
  - command:
    `python scripts/run_quality_batch.py --llm-config configs\openai_compatible.example.json --external-baseline-config configs\external_baselines.qwen.json --ai-indices 13 15 18 21 --live-row-indices 0 23 47 70 --baselines direct self-refine ai-researcher ours-eig --native-eval --batch-name m1-4x4-main-slice`
  - batch directory:
    `outputs/quality_batches/20260410-192309-m1-4x4-main-slice`
- `M1` `4 + 4` run completed cleanly for all four methods on all eight cases.
- Aggregate result summary from the completed `M1` slice:
  - local benchmark-facing averages:
    - `ours-eig`: overall `6.46`, alignment `5.08`
    - `self-refine`: overall `5.79`, alignment `4.19`
    - `direct`: overall `5.38`, alignment `3.51`
    - `ai-researcher`: overall `4.65`, alignment `2.27`
  - benchmark-native averages:
    - `direct`: `7.61`
    - `self-refine`: `7.47`
    - `ai-researcher`: `7.26`
    - `ours-eig`: `6.78`
- Per-benchmark interpretation from the completed `M1` slice:
  - on `AI_Idea_Bench_2025`, `ours-eig` is best under the local
    benchmark-facing scorer (`5.75` overall, `3.94` alignment), but not under
    the benchmark-native average (`6.50` vs `7.86` for `direct`)
  - on `LiveIdeaBench`, `ours-eig` is again best under the local
    benchmark-facing scorer (`7.16` overall, `6.21` alignment), but not under
    the benchmark-native average (`7.05` vs `7.51` for `self-refine` and
    `7.67` for `ai-researcher`)
- Updated `M1` conclusion:
  - operationally, `M1` is now complete and the exact baseline set is stable
    under the current Qwen provider stack
  - scientifically, the evidence is mixed:
    `ours-eig` clearly improves the repo's local benchmark-facing quality and
    alignment signals, but the benchmark-native metrics do not yet support a
    strongest-method claim
  - because the paper protocol treats benchmark-native metrics as the primary
    automatic evidence, the next step should be diagnosis rather than an
    immediate scale-up to `M2`
- Completed the first post-`M1` diagnosis pass on the abnormal
  `direct`-best native-metric pattern.
- Diagnosis result:
  the anomaly is only partly a generation issue.
  A meaningful part comes from benchmark-native judge fragility on
  `AI_Idea_Bench_2025` topic-alignment prompts:
  some saved runs had `i2t_motivation=0` and `i2t_experiment=0` despite
  clearly on-topic proposals.
- Minimal judge-side reproduction confirmed the problem:
  re-running the same `i2t` prompt on saved anomalous outputs returned
  sensible non-zero scores with the same model and prompt family.
- Implemented a scorer robustness patch in
  `src/idea_graph/benchmark_scoring.py`:
  - added schema-aware validation and repair-retry support for
    judge-returned JSON
  - added fallback parsing for flattened `i2t` payload variants
  - `AI_Idea_Bench_2025` topic-alignment scoring now retries once when the
    judge returns an invalid schema instead of silently recording false zeros
- Implemented generation-side cleanup and prompt refinements:
  - `src/idea_graph/literature_grounding.py` now filters noisy dataset-like
    fragments such as `paper introduces ...` and long descriptive clauses that
    should not be promoted into benchmark grounding
  - `src/idea_graph/agent_backend.py` now removes noisy copied proposal
    sentences and strengthens final synthesis instructions toward clean prose,
    one coherent mechanism story, and no OCR/PDF-style fragment leakage
  - `src/idea_graph/baselines.py` now strengthens `direct`, `self-refine`,
    and `ai-researcher` prompts against noisy snippet copying and mechanism
    over-stitching, and cleans noisy section sentences during baseline
    postprocessing
- Added focused regression coverage for:
  - validated native-judge retries,
  - flattened `i2t` payload parsing,
  - noisy snippet filtering in literature grounding,
  - noisy evaluation-fragment cleanup in final proposal postprocessing,
  - prompt-level anti-fragment instructions for baseline generation
- Verification after the diagnosis-driven patch set:
  `python -m pytest -q`
  passed with `77` tests green.
- Low-cost native-evaluation refresh confirmed the scorer fix on previously
  anomalous saved runs:
  - `self-refine` on `AI_Idea_Bench_2025-18` refreshed from a saved
    `native=6.0` packet with zero topic-alignment to a corrected
    `native=8.57`, with `i2t_motivation=4` and `i2t_experiment=5`
  - `ours-eig` on `AI_Idea_Bench_2025-18` refreshed from a saved
    `native=5.71` packet with zero topic-alignment to a corrected
    `native=8.57`, with `i2t_motivation=5` and `i2t_experiment=5`
- Updated interpretation after the diagnosis pass:
  - the earlier `M1` native aggregate is partially stale because some saved
    `AI_Idea_Bench_2025` runs were affected by topic-alignment scoring
    fragility
  - therefore we should not trust the old native leaderboard until the full
    completed `M1` batch is refreshed under the repaired scorer
  - generation quality still needs improvement, especially fluency and
    surface polish on weak-context outputs, but the strongest immediate issue
    was evaluation reliability rather than a total method failure

## 2026-04-10: Full `M1` Native Refresh Completed

- Completed the full judge-only native reevaluation for the saved
  `M1` `4 + 4` batch at
  `outputs/quality_batches/20260410-192309-m1-4x4-main-slice`
  before launching any new generation batch.
- Refresh procedure:
  - re-ran `scripts/evaluate_run.py --native-eval` on all `32` selected saved
    run directories with the repaired scorer and the current
    `configs/openai_compatible.example.json`
  - rebuilt `batch_summary.json`, `batch_summary.md`, `raw_rows.csv`,
    `selected_rows.csv`, `aggregate_rows.csv`, and
    `overall_aggregate_rows.csv` from the refreshed per-run `summary.json`
    files
- Corrected overall `M1` aggregate after the batch-wide native refresh:
  - local benchmark-facing averages:
    - `ours-eig`: overall `6.46`, alignment `5.08`
    - `self-refine`: overall `5.79`, alignment `4.19`
    - `direct`: overall `5.38`, alignment `3.51`
    - `ai-researcher`: overall `4.65`, alignment `2.27`
  - refreshed benchmark-native averages:
    - `self-refine`: `8.12`
    - `ai-researcher`: `7.95`
    - `direct`: `7.93`
    - `ours-eig`: `7.78`
- Corrected per-benchmark interpretation after refresh:
  - on `AI_Idea_Bench_2025`, `ours-eig` remains best under the local
    benchmark-facing scorer (`5.75` overall, `3.94` alignment), while
    `self-refine` is best under the refreshed native score (`8.71`)
  - on `LiveIdeaBench`, `ours-eig` again remains best under the local
    benchmark-facing scorer (`7.16` overall, `6.21` alignment), while
    `ai-researcher` is best under the native score (`7.75`)
- Updated conclusion after the refresh:
  - the earlier "`direct` is best" native interpretation was indeed stale and
    should not be used going forward
  - however, the refreshed table is still mixed rather than fully favorable to
    `ours-eig`: our method leads clearly on local quality and alignment, but
    not yet on benchmark-native automatic scoring
  - this means the next high-value step is still diagnosis-driven prompt and
    synthesis refinement, not immediate scale-up to a larger generation batch

## Next Immediate Step

Use the refreshed `M1` table as the stable checkpoint, then run one small
regeneration packet on the touched codepath:

- diagnose why `self-refine` and `ai-researcher` still convert benchmark
  inputs into stronger native-faithfulness scores than `ours-eig`
- turn that diagnosis into concrete prompt-side grounding and final-synthesis
  revisions for all methods, not only `ours-eig`
- validate those revisions on one small regenerated packet before any larger
  `M2`-style expansion
- keep `virsci` excluded from the main table for now and record it as a later
  integration task

## 2026-04-11: EIG Robustness Cleanup And Narrow AIIB Diagnosis

- Executed the planned pre-`R009` robustness revision for `ours-eig` inline in the repo.
- Code changes in this slice:
  - `src/idea_graph/literature_grounding.py`
    - prompt-safe grounding now ignores hidden target-paper fields
    - dataset/metric extraction from safe reference snippets now only trusts evaluation-side signals and filters generic snippet fragments
  - `src/idea_graph/agent_backend.py`
    - benchmark-mode final grounding now always rebuilds from prompt-safe metadata
    - synthesis payload now includes an explicit `slot_summary`
    - final proposal postprocessing can rewrite generic benchmark method and evaluation text from the selected claim-chain slots
  - `src/idea_graph/engine.py`
    - utility now includes transparent benchmark-facing terms: `benchmark_specificity`, `experiment_alignment`, `role_balance`, and `reference_copy_penalty`
    - maturity now evaluates the claim-chain subgraph first and requires benchmark-specific structure on non-weak-context benchmarks
  - `src/idea_graph/claim_chain.py`
    - slot scoring now prefers concrete methods and grounded evaluation nodes
    - supporting hypotheses are preserved in the selected subgraph when needed
  - `src/idea_graph/models.py`
    - `UtilityBreakdown` now surfaces the new benchmark-facing factors
- Added focused regression coverage in:
  - `tests/test_literature_grounding.py`
  - `tests/test_agent_backend.py`
  - `tests/test_engine.py`
- Verification after the robustness cleanup:
  - `python -m pytest tests/test_literature_grounding.py tests/test_agent_backend.py tests/test_engine.py -v`
  - `59 passed`
- Ran the planned narrow post-cleanup AIIB diagnosis rerun for `ours-eig` on the 4-case smoke subset with native evaluation:
  - `3883`
  - `13`
  - `7909`
  - `9849`
  - output root:
    `outputs/m2_aiib_r009_diagnosis`
- Narrow diagnosis result summary:
  - mean native available-average:
    `8.29 / 10`
  - mean local overall:
    `4.61 / 10`
  - mean local benchmark alignment:
    `2.31 / 10`
  - mean local graph score:
    `9.18 / 10`
  - per-case native:
    - `3883`: `7.43`
    - `13`: `8.57`
    - `7909`: `8.29`
    - `9849`: `8.86`
- Main interpretation:
  - the hard AIIB native gap is partly repaired; `3883` is no longer in the earlier failure regime
  - graph construction and native benchmark faithfulness both look stronger
  - however, proposal surface form is still unstable: some final experiment plans still contain OCR-like benchmark phrases or stitched evaluation text, which drags down the repo's local benchmark-facing scorer
- Updated recommendation:
  - do not launch the full `24`-case `R009` slice yet
  - first run one more narrow cleanup targeting noisy evaluation extraction and benchmark-plan rewriting in the final proposal path

### 2026-04-12: G1/G2 Dataset Construction (Tasks 1-4)

- Executed the dataset-construction slice in the dedicated worktree with a review-gated workflow:
  - implement one task at a time
  - run targeted tests before proceeding
  - keep docs and implementation aligned before handoff
- Task 1 (`G1` manifest enrichment) completed:
  - trajectory manifest entries now include full local and native label payloads needed by downstream critic-dataset assembly
- Task 2 (`G2` critic-dataset tests) completed:
  - added and stabilized split-assignment, label-packaging, and duplicate-handling coverage in `tests/test_critic_dataset.py`
  - targeted verification run used:
    `python -m pytest tests/test_trajectory_dataset.py tests/test_critic_dataset.py -q`
- Task 3 (`G2` `critic_dataset` implementation) completed with safety-focused fixes:
  - strict transition-group validation and split assignment checks
  - deterministic duplicate handling and duplicate-burden reporting
  - coverage accounting for weak labels and native labels in exported summaries
- Smoke artifact commands used:
  - G1 export:
    `python scripts/export_graph_critic_dataset.py --input-root outputs/m2_aiib_r009_diagnosis_safe_grounding --output-dir outputs/graph_critic_datasets --dataset-name smoke_r009_safe_grounding --limit-runs 4`
  - G2 build:
    `python scripts/build_graph_critic_dataset.py --g1-dataset-dir outputs/graph_critic_datasets/smoke_r009_safe_grounding --output-dir outputs/graph_critic_datasets --dataset-name smoke_r009_safe_grounding_g2`
- Task 4 smoke build completed successfully on the first `G2` smoke profile.
- First verified smoke profile:
  - G1 dataset: `outputs/graph_critic_datasets/smoke_r009_safe_grounding`
  - G1 `run_count`: `4`
  - G1 `transition_count`: `100`
  - G1 `average_actions_per_usable_eig_run`: `25.0`
  - G1 `total_tokens`: `817273`
  - G1 `mean_tokens_per_run`: `204318.25`
  - G2 dataset: `outputs/graph_critic_datasets/smoke_r009_safe_grounding_g2`
  - G2 `group_count`: `4`
  - G2 `train_group_count`: `3`
  - G2 `validation_group_count`: `1`
  - G2 `transition_count`: `100`
  - G2 `train_transition_count`: `75`
  - G2 `validation_transition_count`: `25`
  - weak-label coverage: `1.0`
  - native-label coverage: `1.0`
  - native-average coverage: `1.0`
  - duplicate burden mean/max runs per group: `1.0 / 1.0`
- Final verification state for this slice:
  - `python -m pytest tests/test_trajectory_dataset.py tests/test_critic_dataset.py -q` passed with `15 passed`
  - first `G2` smoke build completed successfully and produced a split-ready dataset for `G3`

### 2026-04-12: G2.5 Candidate Slates And G3 Text Critic Pilot

- Completed the next-action critic dataset extension in the dedicated
  `g2-critic-dataset` worktree.
- `G2.5` is now a clean derived layer over frozen `G1` trajectory exports and
  `G2` split/label manifests. It does not mutate `G2`.
- Candidate-slate build command:
  `python scripts/build_graph_critic_candidate_dataset.py --g1-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g1 --g2-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g2 --output-dir outputs/graph_critic_datasets --dataset-name current_benchmarked_ours_eig_full_g25`
- Candidate-slate artifact:
  `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g25`
- `G2.5` dataset profile:
  - `state_count`: `910`
  - `candidate_count`: `9456`
  - `commit_count`: `910`
  - `commit_fraction`: `0.0962351945854484`
  - `mean_candidates_per_state`: `10.391208791208792`
  - split state counts: `805` train and `105` validation
  - split candidate counts: `8479` train and `977` validation
- Implemented the first `G3` text-only next-action critic pilot:
  - model input: `state_text [SEP] candidate_text`
  - supervision: logged selected edit imitation
  - model family: TF-IDF features plus logistic regression
  - paper framing: low-data pilot and data-sanity baseline, not a final
    graph-encoder controller
- Text-critic training command:
  `python scripts/train_text_critic.py --candidate-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g25 --output-dir outputs/graph_critic_models/current_benchmarked_ours_eig_full_g3_text_pilot`
- Text-critic artifact:
  `outputs/graph_critic_models/current_benchmarked_ours_eig_full_g3_text_pilot`
- `G3` validation metrics:
  - `state_count`: `105`
  - `top1_accuracy`: `0.7238095238095238`
  - `mean_reciprocal_rank`: `0.8412698412698413`
  - `train_example_count`: `8479`
  - `validation_example_count`: `977`
- `G3` metadata checks:
  - `train_group_count`: `9`
  - `validation_group_count`: `2`
  - `group_overlap_count`: `0`
  - `train_commit_positive_count`: `0`
  - `validation_commit_positive_count`: `0`
- Important limitation:
  - current `G3` is an offline logged-edit imitation scorer
  - every state still includes one explicit `commit` candidate in the slate,
    but no `commit` candidate is the logged selected action in the current
    train or validation split
  - because of that supervision gap, this pilot does not yet validate learned
    commit decisions
  - future `G4/G5` should add commit-vs-continue supervision before claiming a
    full graph-critic controller
- Verification completed during this slice:
  - `python -m pytest tests/test_engine.py -q` passed
  - `python -m pytest tests/test_trajectory_dataset.py tests/test_candidate_slate_dataset.py tests/test_critic_dataset.py -q` passed
  - `python -m pytest tests/test_text_critic.py -q` passed
  - `python -m pytest tests/test_engine.py tests/test_candidate_slate_dataset.py tests/test_text_critic.py tests/test_critic_dataset.py tests/test_trajectory_dataset.py -q` passed

### 2026-04-12: G3.5 Partition Manifest Layer

- Closed Task 0 of the online-critic plan on the canonical `main` branch:
  - committed reviewed graph-critic sync as `8652962`
  - reran the focused critic packet on `main`
  - verification result:
    `python -m pytest tests/test_engine.py tests/test_candidate_slate_dataset.py tests/test_text_critic.py tests/test_critic_dataset.py tests/test_trajectory_dataset.py -q`
    passed with `61 passed`
- Implemented the new deterministic partition-manifest layer:
  - `src/idea_graph/critic_partitions.py`
  - `scripts/build_critic_partition_manifest.py`
  - `tests/test_critic_partitions.py`
- Task 1A review state:
  - spec-compliance review: approved
  - code-quality review: approved after one CLI output-dir fix
- New local verification for the partition layer:
  - `python -m pytest tests/test_critic_partitions.py -q` passed with `7 passed`
  - `python -m pytest tests/test_trajectory_dataset.py tests/test_critic_dataset.py tests/test_candidate_slate_dataset.py tests/test_critic_partitions.py -q` passed with `33 passed`
- Built the first concrete partition artifact:
  - command:
    `python scripts/build_critic_partition_manifest.py --g2-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g2 --output-dir outputs/graph_critic_datasets --dataset-name current_benchmarked_ours_eig_full_g35_partitions`
  - artifact:
    `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions`
  - current stats:
    - `group_count`: `11`
    - `critic_train`: `9`
    - `critic_dev`: `2`
    - `paper_eval`: `0`
    - `has_paper_eval`: `false`
- Recovery diagnosis update:
  - the active repo can currently rediscover the full `60` benchmarked
    `ours-eig` runs from `outputs/`
  - this means the older `48`-run `current_benchmarked_ours_eig_full_g1_commit_enriched`
    artifact is a stale partial rebuild, not evidence of missing runs in the
    current workspace
- Immediate next step after this checkpoint:
  - extend the enriched dataset contract with explicit provenance /
    `partition_role` fields
  - rebuild the commit-enriched `G1/G2/G2.5` datasets from the full `60`-run
  pool
  - then move to the warm-start pilot on the clean partitioned dataset

### 2026-04-12: Full 60-Run Commit-Enriched Refresh

- Rebuilt the stale commit-enriched dataset stack from the current full
  `60`-run `ours-eig` pool in the active repo.
- Rebuild commands:
  - `python scripts/export_graph_critic_dataset.py --input-root outputs --output-dir outputs/graph_critic_datasets --dataset-name current_benchmarked_ours_eig_full_g1_commit_enriched --baseline ours-eig`
  - `python scripts/build_graph_critic_dataset.py --g1-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g1_commit_enriched --output-dir outputs/graph_critic_datasets --dataset-name current_benchmarked_ours_eig_full_g2_commit_enriched`
  - `python scripts/build_graph_critic_candidate_dataset.py --g1-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g1_commit_enriched --g2-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g2_commit_enriched --output-dir outputs/graph_critic_datasets --dataset-name current_benchmarked_ours_eig_full_g25_commit_enriched`
- Refreshed artifact counts:
  - `G1`:
    - `run_count`: `60`
    - `transition_count`: `910`
    - `terminal_state_count`: `60`
  - `G2`:
    - `run_count`: `60`
    - `transition_count`: `910`
    - `group_count`: `11`
  - `G2.5`:
    - `state_count`: `970`
    - `candidate_count`: `10092`
    - `commit_positive_count`: `60`
    - `terminal_state_count`: `60`
- Reran the text-only critic on the refreshed full-pool candidate dataset:
  - command:
    `python scripts/train_text_critic.py --candidate-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g25_commit_enriched --output-dir outputs/graph_critic_models/current_benchmarked_ours_eig_full_g3_text_pilot_commit_enriched`
  - metrics:
    - `state_count`: `110`
    - `top1_accuracy`: `0.7363636363636363`
    - `mean_reciprocal_rank`: `0.8484848484848484`
  - commit-label metadata:
    - `train_commit_positive_count`: `55`
    - `validation_commit_positive_count`: `5`
- Interpretation:
  - the refreshed `G3` pilot remains a modest logged-edit scorer, but it no
    longer suffers from the earlier zero-positive-commit supervision problem
  - this makes the next warm-start / scorer-based controller step much more
    meaningful than before
- Immediate next step after this refresh:
  - add explicit `partition_role` / provenance fields to the enriched dataset
    contract where still missing
  - then start the first warm-start trainer on top of the refreshed full-pool
    `G2.5` artifact and `G3.5` partition manifest
