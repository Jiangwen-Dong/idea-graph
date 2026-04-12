# Paper Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| `R001` | `M0` | verify benchmark-native path on `AI_Idea_Bench_2025` | `ours-eig` | AI Idea Bench index `13` | native AIIB metrics + artifact writeout | MUST | DONE | run dir `outputs/20260410-154406-ai-idea-bench-2025-13`; benchmark-native path completed cleanly; native average normalized `6.0/10` |
| `R002` | `M0` | verify benchmark-native path on `LiveIdeaBench` | `ours-eig` | LiveIdeaBench row `0` | native LiveIdeaBench rubric + artifact writeout | MUST | DONE | run dir `outputs/20260410-154531-liveideabench-meteorology-0`; benchmark-native path completed cleanly; native average `7.67/10` |
| `R003` | `M0` | exact baseline smoke test | `ai-researcher` | AI Idea Bench index `13` | successful benchmark-mode generation | MUST | DONE | run dir `outputs/20260410-160800-ai-idea-bench-2025-13`; completed through the documented `openai-compatible-bridge`; exact upstream mode is still unsupported under the current DashScope/Qwen stack |
| `R004` | `M0` | baseline feasibility audit | `virsci` exact wrapper | code audit + optional dry run | pass/fail decision on benchmark-faithful usage | MUST | NO-GO | current wrapper explicitly rejects benchmark mode; upstream stack expects its own `ollama_*` configs and external knowledge-bank paths, so it should not stay in the headline benchmark table without a dedicated patch |
| `R005A` | `M1` | warm-start control run | `direct` | AI Idea Bench index `13` | native AIIB metrics + local quality | MUST | DONE | run dir `outputs/20260410-160957-ai-idea-bench-2025-13`; native average normalized `8.29/10` |
| `R006A` | `M1` | warm-start control run | `self-refine` | AI Idea Bench index `13` | native AIIB metrics + local quality | MUST | DONE | run dir `outputs/20260410-161048-ai-idea-bench-2025-13`; native average normalized `8.57/10` |
| `R007A` | `M1` | warm-start baseline run | `ai-researcher` | LiveIdeaBench row `0` | native LiveIdeaBench rubric + local quality | MUST | DONE | run dir `outputs/20260410-161232-liveideabench-meteorology-0`; native average `7.67/10`, but local benchmark alignment remained extremely low (`0.04/10`) |
| `R007B` | `M1` | post-fix verification rerun | `ai-researcher` | LiveIdeaBench row `0` | native LiveIdeaBench rubric + refreshed local quality | MUST | DONE | run dir `outputs/20260410-163049-liveideabench-meteorology-0`; fixed language-field leakage in the AI-Researcher bridge and corrected local topic-alignment tokenization/keyword handling; proposal now stays in meteorology; native average `7.67/10`; refreshed local benchmark alignment `2.0/10` |
| `R008A` | `M1` | warm-start control run | `direct` | LiveIdeaBench row `0` | native LiveIdeaBench rubric + local quality | MUST | DONE | run dir `outputs/20260410-161112-liveideabench-meteorology-0`; native average `7.67/10` |
| `R008B` | `M1` | warm-start control run | `self-refine` | LiveIdeaBench row `0` | native LiveIdeaBench rubric + local quality | MUST | DONE | run dir `outputs/20260410-161141-liveideabench-meteorology-0`; native average `7.67/10` |
| `R008C` | `M1` | weak-context scaffold verification | `ours-eig` | LiveIdeaBench row `0` | native LiveIdeaBench rubric + local quality | SHOULD | DONE | run dir `outputs/20260410-172718-liveideabench-meteorology-0`; first keyword-only scaffold pass; local overall `5.62/10`, local benchmark alignment `3.86/10`, native average normalized `7.67/10` |
| `R008D` | `M1` | weak-context maturity verification | `ours-eig` | LiveIdeaBench row `0` | native LiveIdeaBench rubric + local quality | SHOULD | DONE | run dir `outputs/20260410-173245-liveideabench-meteorology-0`; preserved divergence through `Round3`; local overall `6.12/10`, local benchmark alignment `4.57/10`, native average normalized `7.05/10`; best local weak-context run so far |
| `R008E` | `M1` | weak-context synthesis-cleanup rerun | `ours-eig` | LiveIdeaBench row `0` | native LiveIdeaBench rubric + local quality | SHOULD | DONE | run dir `outputs/20260410-181027-liveideabench-meteorology-0`; removed raw scaffold leakage in the method section; native average normalized `7.05/10`; local benchmark alignment `3.88/10`; exposed a remaining evaluation-coverage bottleneck now patched deterministically in code |
| `R008F` | `M1` | weak-context method-instantiation rerun | `ours-eig` | LiveIdeaBench row `0` | native LiveIdeaBench rubric + local quality | SHOULD | DONE | run dir `outputs/20260411-000104-liveideabench-meteorology-0`; added explicit weak-context method instantiation so meteorology proposals produce concrete mechanisms rather than only scaffolded themes; local overall `7.09/10`, local benchmark alignment `6.06/10`, native average normalized `7.67/10` |
| `R005-mini` | `M1` | refreshed current-codepath mini packet | `direct`, `self-refine`, `ai-researcher`, `ours-eig` | AI Idea Bench index `13` + LiveIdeaBench row `0` | native metrics + refreshed local quality | MUST | DONE | batch dir `outputs/quality_batches/20260410-181953-refreshed-m1-mini-current-codepath`; exact `ai-researcher` bridge now included in the batch runner; `ours-eig` is strongest on AI Idea Bench and highest by combined local overall, while `self-refine` remains strongest on the current LiveIdeaBench meteorology case |
| `R005-mini-v2` | `M1` | targeted weak-context regression-fix mini packet | `direct`, `self-refine`, `ai-researcher`, `ours-eig` | AI Idea Bench index `13` + LiveIdeaBench row `0` | native metrics + refreshed local quality | MUST | DONE | batch dir `outputs/quality_batches/20260410-185224-refreshed-m1-mini-weak-context-v3`; after fixing AI Idea Bench postprocess grounding, `ours-eig` recovered on AIIB (`5.92/4.33/8.57`) while staying strongest on LiveIdeaBench (`7.11/6.15/7.05`), making it the strongest method on both refreshed mini-batch cases under the local benchmark-facing evaluator |
| `R005-mini-v3` | `M1` | refreshed cross-benchmark small M1 packet with native judge refresh | `direct`, `self-refine`, `ai-researcher`, `ours-eig` | AI Idea Bench indices `13, 15` + LiveIdeaBench rows `0, 23` | native metrics + refreshed local quality | MUST | DONE | batch dir `outputs/quality_batches/20260411-000159-refreshed-m1-mini-synthesis-cleanup-v2-native`; `ours-eig` is strongest overall by local score (`6.62/10`) and strongest on `AI_Idea_Bench_2025` by both local and native scores (`8.86` native); on `LiveIdeaBench`, `ours-eig` ties `self-refine` on mean native score (`7.36`) while remaining stronger by local alignment (`6.08` vs `5.55`); meteorology is the remaining stochastic weak-context case |
| `R005` | `M1` | cross-benchmark smoke batch | `direct` | `4` AIIB + `4` Live | benchmark-native automatic metrics | MUST | DONE | shared batch dir `outputs/quality_batches/20260410-192309-m1-4x4-main-slice`; aggregate overall `5.38`, alignment `3.51`, native `7.61` |
| `R006` | `M1` | cross-benchmark smoke batch | `self-refine` | `4` AIIB + `4` Live | benchmark-native automatic metrics | MUST | DONE | shared batch dir `outputs/quality_batches/20260410-192309-m1-4x4-main-slice`; aggregate overall `5.79`, alignment `4.19`, native `7.47`; strongest native single-agent baseline in the final `M1` packet |
| `R007` | `M1` | cross-benchmark smoke batch | `ai-researcher` | `4` AIIB + `4` Live | benchmark-native automatic metrics | MUST | DONE | shared batch dir `outputs/quality_batches/20260410-192309-m1-4x4-main-slice`; aggregate overall `4.65`, alignment `2.27`, native `7.26`; bridge path remained stable throughout the full packet |
| `R008` | `M1` | cross-benchmark smoke batch | `ours-eig` | `4` AIIB + `4` Live | benchmark-native automatic metrics + graph diagnostics | MUST | DONE | shared batch dir `outputs/quality_batches/20260410-192309-m1-4x4-main-slice`; aggregate overall `6.46`, alignment `5.08`, graph `9.07`, native `6.78`; strongest by the local benchmark-facing scorer, but not by the benchmark-native averages |
| `R009A` | `M2` | preflight + smoke gate | `direct`, `self-refine`, `ai-researcher`, `ours-eig` | AIIB smoke subset `13, 3883, 7909, 9849` | local AIIB metrics + targeted native smoke check | MUST | DONE | output root `outputs/m2_aiib_r009_smoke`; preflight passed; smoke generation finished `16/16`; targeted native check finished `8/8` for `self-refine` and `ours-eig`; local smoke strongly favors `ours-eig` (`6.00/4.37`), while targeted native smoke is still slightly higher for `self-refine` (`8.36` vs `8.07`) |
| `R009B` | `M2` | post-cleanup narrow AIIB diagnosis rerun | `ours-eig` | AIIB smoke subset `13, 3883, 7909, 9849` | native AIIB metrics + local AIIB metrics | MUST | DONE | output root `outputs/m2_aiib_r009_diagnosis`; after the robustness cleanup, mean native rises to `8.29/10` and hard case `3883` recovers to `7.43/10`, but local overall/alignment fall to `4.61/2.31` because several final experiment plans still contain OCR-like or stitched evaluation fragments |
| `R009C` | `M2` | title-anchor narrow AIIB diagnosis rerun | `ours-eig` | AIIB smoke subset `13, 3883, 7909, 9849` | native AIIB metrics + local AIIB metrics | MUST | DONE | output root `outputs/m2_aiib_r009_diagnosis_title_anchor`; lightweight title-derived anchors (`Amex dataset`, `Osworld benchmark`) sharply improved local overall/alignment (`5.91/4.43`) and cleaned the `3883` proposal surface form, but mean native fell to `7.36/10`, with notable drops on `3883` and `9849` |
| `R009D` | `M2` | safe-grounding + synthesis narrow AIIB rerun | `ours-eig` | AIIB smoke subset `13, 3883, 7909, 9849` | native AIIB metrics + local AIIB metrics | MUST | DONE | output root `outputs/m2_aiib_r009_diagnosis_safe_grounding`; mean native recovered to `8.07/10` after the dataset-only title-anchor cleanup, while local overall/alignment settled at `5.26/3.48`; this is a better compromise than `R009C`, but still not clean enough to justify the full `24`-case launch |
| `R009` | `M2` | core automatic batch | shortlisted main baselines | at least `24` AIIB | native AIIB metrics including matched-pool `IC` | MUST | PAUSED | smoke gate is complete through `R009D`, but the full `24`-case launch is paused because the method track is shifting from heuristic maturity to a learned graph critic; archived launch note: `docs/archive/r009_aiib_launch_plan_pre_critic.md` |
| `R010` | `M2` | core automatic batch | shortlisted main baselines | at least `24` LiveIdeaBench | native LiveIdeaBench metrics | MUST | TODO | launch after the weak-context stability decision; keep the same shared output contract |
| `R011` | `M4` | human blind review packet | `direct`, `self-refine`, `ai-researcher`, `ours-eig` | balanced `8` to `12` cases per benchmark | human rubric | MUST | TODO | anonymize system names and standardize formatting |
| `R012` | `M5` | pre-critic heuristic ablation | `ours-eig` without utility-guided ranking | balanced `6 + 6` subset | native metrics + graph metrics | SHOULD | PAUSED | keep as optional heuristic ablation only if the critic track needs a fallback comparison |
| `R013` | `M5` | pre-critic heuristic ablation | `ours-eig` without heuristic stop | balanced `6 + 6` subset | native metrics + commit/round diagnostics | SHOULD | PAUSED | superseded by learned commit-vs-continue evaluation in the graph-critic track |
| `R014` | `M5` | pre-critic heuristic ablation | `ours-eig` with flat final synthesis | balanced `6 + 6` subset | native metrics + graph metrics | SHOULD | PAUSED | keep as optional synthesis ablation after the critic pilot stabilizes |
| `R015` | `M5` | appendix analysis | `ours-eig` plus main baselines | reuse `M2` outputs | process metrics, cost, runtime, fallback statistics | SHOULD | PAUSED | will be refreshed after graph-critic outputs exist |
| `G001` | `G0` | graph-critic method planning | docs only | no new generation | plan completeness | MUST | DONE | canonical forward plan added at `docs/eig_graph_critic_plan.md`; pre-critic records remain available as pilot data |
| `G002` | `G1` | trajectory export | saved EIG runs | small `M1` + `R009` pilots | exported state-action-score examples | MUST | DONE | spec: `docs/superpowers/specs/2026-04-12-graph-critic-trajectory-export-design.md`; implementation: `src/idea_graph/trajectory_dataset.py` + `scripts/export_graph_critic_dataset.py`; smoke dataset: `outputs/graph_critic_datasets/smoke_r009_safe_grounding` with `4` runs, `100` transition examples, and full trace-coverage profiling |
| `G003` | `G2` | critic dataset construction | exported trajectories | train/validation by benchmark instance | leakage-free splits | MUST | TODO | next target after `G1`; keep local weak labels separate from benchmark-native labels and add explicit overhead/profile reporting for the final training corpus |
| `G004` | `G3` | text critic baseline | flattened graph state | held-out trajectory split | action ranking + commit-vs-continue accuracy | MUST | TODO | sanity-check whether saved labels are learnable before implementing the graph encoder |
| `G005` | `G4` | graph critic | structured graph state | held-out trajectory split | action ranking + commit calibration | MUST | TODO | compare against the text critic to test whether graph structure helps |
| `G006` | `G5` | critic-controlled pilot | `ours-eig-critic` | AIIB smoke subset `13, 3883, 7909, 9849` | native AIIB + local quality + commit diagnostics | MUST | TODO | run only after critic validation is stable |

## 2026-04-10: EIG Role/Synthesis Revision

- implemented a new `src/idea_graph/claim_chain.py` utility for selecting a synthesis-ready scientific claim chain
- `select_final_subgraph(...)` now prefers a complete claim chain over the old legacy heuristic when available
- prompt contracts for `ours-eig` were revised toward functional scientific roles:
  - `ImpactReframer` -> `TaskFramer`
  - `NoveltyExaminer` -> `LiteratureGrounder`
  - `MechanismProposer` -> `MethodArchitect`
  - `EvaluationDesigner` -> `ExperimentDesigner`
  - `FeasibilityCritic` -> `SkepticRepairer`
- baseline prompt families were left unchanged apart from prior generic hygiene; no baseline was retrofitted with our role decomposition
- maturity now requires a complete claim chain, not only structural coherence
- synthesis payloads now expose the selected claim chain explicitly so final proposal generation can stay anchored to the validated graph path
- weak-context `LiveIdeaBench` handling remains supported:
  - a keyword-only graph may relax the explicit literature-gap slot
  - but it still must contain a task anchor, mechanism, evaluation, and caveat to be synthesis-ready
- regression status after this revision slice:
  - `python -m pytest -q`
  - `84 passed`
- next recommended check:
  - run one small regenerated packet before any larger new batch:
    `python scripts/run_quality_batch.py --llm-config configs\\openai_compatible.example.json --external-baseline-config configs\\external_baselines.qwen.json --ai-indices 13 15 --live-row-indices 0 23 --baselines direct self-refine ai-researcher ours-eig --batch-name m1-role-synthesis-check`

## 2026-04-10: Weak-Context And Final-Synthesis Cleanup

- diagnosed two concrete post-synthesis failure modes from the regenerated mini packet:
  - weak-context leakage from deterministic scaffold text (`The central mechanism should stay explicitly targeted ...`, `Focus the method on ...`, `keyword-specific case studies`, `generic problem framing`)
  - literature-grounded final synthesis still allowing imperative method phrasing (`Use ...`) and fragmentary benchmark-plan tails
- implemented targeted fixes in `src/idea_graph/agent_backend.py`:
  - weak-context postprocess now rewrites scaffold-like existing-method, hypothesis, method, and evaluation text into paper-style sentences
  - keyword-only placeholder phrases are normalized into natural benchmark-facing language
  - proposal cleaning no longer drops legitimate `periodic table` sentences because the noisy-sentence filter now only removes real `Table 1` / `Figure 2`-style artifacts
  - literature-grounded synthesis now rewrites imperative `Use ...` method sentences and upgrades short `Evaluate on ... / Report ...` tails into a full grounded evaluation plan
- implemented targeted fixes in `src/idea_graph/literature_grounding.py`:
  - added a dedicated periodic-table weak-context scaffold instead of falling back to the generic `general_science` family
  - replaced generic scaffold residue terms with more natural placeholders even for fallback cases
- regression coverage added:
  - weak-context instruction leakage cleanup
  - periodic-table scaffold cleanup
  - AI Idea Bench imperative-method and fragmentary-evaluation cleanup
- verification status after this pass:
  - `python -m pytest -q`
  - `88 passed`
- small post-patch sanity runs:
  - `outputs/20260410-230705-liveideabench-periodic-table-23`
    - local overall `7.14/10`, alignment `6.32/10`, graph `9.17/10`
    - periodic-table proposal is now clean weak-context text with domain-specific evaluation assets
  - `outputs/20260410-231206-ai-idea-bench-2025-13`
    - local overall `6.31/10`, alignment `4.98/10`, graph `9.49/10`
    - imperative `Use ...` method phrasing is removed and the experiment plan is now synthesized as a single benchmark-grounded block
- next recommended step:
  - rerun a refreshed small `M1` packet on `AIIB {13,15}` and `Live {0,23}` for `direct`, `self-refine`, `ai-researcher`, and `ours-eig`, then refresh judge-only/native evaluation if the generation outputs look stable

## 2026-04-11: Small-M1 Refresh And Infrastructure Hardening

- shell progress output now surfaces the functional scientific role names:
  - `TaskFramer`
  - `LiteratureGrounder`
  - `MethodArchitect`
  - `ExperimentDesigner`
  - `SkepticRepairer`
- the underlying graph schema still keeps the legacy internal role ids for backward compatibility, but terminal progress and prompt-facing text now use the new functional names consistently
- Windows long nested output paths were causing flaky reads and writes for benchmark-native evaluation artifacts, so the repo now uses long-path-safe text I/O helpers in the evaluation path
- weak-context grounding was refined again:
  - dedicated `periodic table` scaffold family
  - more natural weak-context scaffold language
  - explicit `method_instantiation` support so weak-context synthesis emits concrete mechanisms rather than generic placeholder prose
- final-synthesis cleanup now removes residual scaffold leakage more aggressively and rewrites short imperative method/evaluation fragments into paper-style proposal text
- regression status after this slice:
  - `python -m pytest -q`
  - `91 passed`
- refreshed small `M1` packet:
  - batch dir `outputs/quality_batches/20260411-000159-refreshed-m1-mini-synthesis-cleanup-v2-native`
  - `ours-eig` aggregate: local overall `6.62/10`, local alignment `5.39/10`, native `8.11/10`
  - `self-refine` aggregate: local overall `5.66/10`, native `8.04/10`
  - `direct` aggregate: local overall `5.42/10`
  - `ai-researcher` aggregate: local overall `4.99/10`
  - on `AI_Idea_Bench_2025`, `ours-eig` is clearly strongest and reaches native `8.86/10`
  - on `LiveIdeaBench`, `ours-eig` ties `self-refine` on mean native score (`7.36/10`) and remains stronger by local alignment, but meteorology still shows some run-to-run variance
- current recommendation:
  - treat `R005-mini-v3` as the reference small-`M1` packet
  - only run another narrowly targeted weak-context stabilization pass if we decide meteorology variance is still too high before launching larger `M2` slices

## 2026-04-11: EIG Robustness Cleanup + AIIB Diagnosis

- implemented the pre-`R009` robustness cleanup for `ours-eig`:
  - benchmark-mode grounding now rebuilds from prompt-safe metadata instead of hidden target fields
  - literature grounding now filters generic snippet fragments that were being misread as datasets or metrics
  - utility now exposes benchmark-facing factors: `benchmark_specificity`, `experiment_alignment`, `role_balance`, and `reference_copy_penalty`
  - claim-chain selection is now slot-aware, prefers grounded evaluation nodes, and preserves a supporting hypothesis in the mature subgraph when needed
  - maturity now evaluates the claim-chain subgraph first and requires benchmark-specific structure on non-weak-context benchmarks
  - synthesis payloads now expose a `slot_summary`, and final proposal postprocessing can rewrite generic benchmark outputs from the selected mechanism/evaluation slots
- regression status after this slice:
  - `python -m pytest tests/test_literature_grounding.py tests/test_agent_backend.py tests/test_engine.py -v`
  - `59 passed`
- narrow AIIB diagnosis rerun completed for `ours-eig` on `13, 3883, 7909, 9849`:
  - output root:
    `outputs/m2_aiib_r009_diagnosis`
  - native average:
    `8.29/10`
  - local averages:
    overall `4.61/10`, alignment `2.31/10`, graph `9.18/10`
  - per-case native:
    - `13`: `8.57`
    - `3883`: `7.43`
    - `7909`: `8.29`
    - `9849`: `8.86`
- interpretation:
  - the hard AIIB case `3883` is materially better than the earlier failure regime, so the robustness cleanup is directionally correct
  - however, the final proposal surface form is still not stable enough for a full `R009` launch: several runs still show OCR-like evaluation fragments, noisy dataset names, or stitched benchmark plans even when the native score improves
- next recommended step:
  - do one more narrow grounding/synthesis cleanup aimed specifically at OCR-like evaluation extraction and noisy benchmark-plan rewriting before spending the full `24`-case `R009` budget

## 2026-04-11: Post-Diagnosis Benchmark-Safety Cleanup

- implemented the follow-up benchmark-safety fix after the narrow AIIB diagnosis:
  - benchmark reference packets now strip OCR-like GUI dump snippets before they
    reach prompts
  - prompt-safe metadata now inherits that sanitized benchmark packet
  - saved `literature_grounding` in `run_experiment(...)` now comes from
    `generation_safe_metadata`, so hidden target-paper summaries no longer leak
    into fresh benchmark-mode EIG artifacts
  - metric-signal matching is now boundary-aware, which removes the false
    `IoU` trigger inside words like `various`
- added targeted regression coverage for:
  - reference-packet sanitization
  - prompt-safe grounding persistence
  - OCR-like benchmark evaluation rewriting
  - false metric matches from substring collisions
- verification status after this slice:
  - `python -m pytest tests/test_benchmark_mode_and_baselines.py tests/test_agent_backend.py tests/test_engine.py tests/test_literature_grounding.py -v`
  - `82 passed`
- current recommendation:
  - hard-case spot check on `3883` has now been run at
    `outputs/m2_aiib_r009_postfix_spotcheck/20260411-175908-ai-idea-bench-2025-3883`
  - that run confirms the bug fix worked: the OCR-like experiment-plan
    corruption is gone, and saved grounding is prompt-safe
  - the next decision is whether to accept the still-generic but readable
    proposal surface form and rerun the 4-case `R009B` packet, or do one more
    lightweight grounding-specific patch for benchmark-title-derived evaluation
    anchors first

## 2026-04-11: Title-Derived Anchor Patch + Refreshed 4-Case Diagnosis

- implemented one lightweight grounding patch in `src/idea_graph/literature_grounding.py`:
  - when snippet-derived grounding is weak, visible reference titles can now
    contribute conservative evaluation anchors such as:
    - `Amex dataset`
    - `Osworld benchmark`
    - `grounding accuracy`
    - `success rate`
    - `error rate`
- added regression coverage for:
  - title-derived benchmark anchors in `tests/test_literature_grounding.py`
  - final-proposal use of those anchors in `tests/test_agent_backend.py`
- verification after the patch:
  - `python -m pytest tests/test_benchmark_mode_and_baselines.py tests/test_agent_backend.py tests/test_engine.py tests/test_literature_grounding.py -v`
  - `84 passed`
- reran the full 4-case `ours-eig` diagnosis packet with native evaluation:
  - output root:
    `outputs/m2_aiib_r009_diagnosis_title_anchor`
  - cases:
    `13, 3883, 7909, 9849`
- result summary:
  - mean local overall:
    `5.91 / 10`
  - mean local benchmark alignment:
    `4.43 / 10`
  - mean local graph score:
    `9.24 / 10`
  - mean native available-average:
    `7.36 / 10`
- main deltas versus `R009B`:
  - `13`: local improved; native unchanged at `8.57`
  - `3883`: local improved strongly (`4.79 -> 6.58`) and the proposal became much cleaner, but native fell (`7.43 -> 5.71`)
  - `7909`: local improved; native unchanged at `8.29`
  - `9849`: local improved strongly (`4.54 -> 6.40`), but native fell (`8.86 -> 6.86`)
- interpretation:
  - the title-anchor patch is useful for cleaning generic or under-anchored
    local benchmark behavior
  - but it is not yet sufficient for stable benchmark-native gains, and may
    currently over-steer some cases toward visible-title heuristics

## 2026-04-11: Safe-Grounding Dataset-Only Anchor Cleanup

- refined the benchmark-safe title-anchor fallback:
  - kept visible-title dataset anchors such as `Amex dataset` and
    `Osworld benchmark`
  - removed title-only metric invention so titles alone no longer inject
    guessed metrics such as `grounding accuracy`, `success rate`, or
    `error rate`
- strengthened safe-grounding cleanup for AIIB math-style OCR residue:
  - `Work in Progress Eval Dataset` and sample-count pseudo-datasets are now
    filtered before they reach experiment-plan synthesis
- refined final synthesis:
  - malformed fragments such as `Report satisfied by ...` are now stripped
  - baseline comparison sentences now use natural short reference labels rather
    than `...-style baselines`
- verification after this slice:
  - `python -m pytest tests/test_literature_grounding.py tests/test_agent_backend.py tests/test_engine.py -q`
  - `67 passed`
- current recommendation:
  - run `R009D` next on the same 4 AIIB smoke cases
  - use that refreshed packet to judge whether EIG is now ready for the full
    `24`-case `R009` launch

## 2026-04-11: R009D Completed

- reran the refreshed 4-case AIIB packet for `ours-eig` under:
  `outputs/m2_aiib_r009_diagnosis_safe_grounding`
- aggregate result:
  - local overall:
    `5.26 / 10`
  - local alignment:
    `3.48 / 10`
  - local graph:
    `9.28 / 10`

## 2026-04-12: Graph-Critic G1 Trajectory Export

- recorded the concrete `G1` exporter design at:
  `docs/superpowers/specs/2026-04-12-graph-critic-trajectory-export-design.md`
- recorded the implementation plan at:
  `docs/superpowers/plans/2026-04-12-graph-critic-trajectory-export.md`
- implemented the reusable exporter library:
  `src/idea_graph/trajectory_dataset.py`
- implemented the thin CLI:
  `scripts/export_graph_critic_dataset.py`
- added focused regression coverage:
  `tests/test_trajectory_dataset.py`
- verification:
  - `python -m pytest tests/test_trajectory_dataset.py -q`
  - `7 passed`
- smoke export:
  - command:
    `python scripts/export_graph_critic_dataset.py --input-root outputs/m2_aiib_r009_diagnosis_safe_grounding --output-dir outputs/graph_critic_datasets --dataset-name smoke_r009_safe_grounding --limit-runs 4`
  - output dataset:
    `outputs/graph_critic_datasets/smoke_r009_safe_grounding`
  - current smoke profile:
    - `4` runs
    - `100` transition examples
    - `817,273` total traced tokens
    - mean traced tokens per run: `204,318.25`
    - mean traced tokens per transition: `8,172.73`
    - full trace coverage for `agent_traces`, `final_synthesis_trace`, override traces, local evaluation, and benchmark-native evaluation on this smoke slice
- current interpretation:
  - the saved pre-critic `R009D` artifacts are rich enough to support offline
    graph-critic data construction
  - the exporter already records the reviewer-facing profiling burden needed for
    later overhead analysis
  - the next stage is `G2`: build benchmark-instance-level train/validation
    splits and define the first weak-label package for critic training
  - native:
    `8.07 / 10`
- comparison:
  - stronger native compromise than `R009C`
  - still slightly weaker native average than `R009B`
  - keeps a substantial part of the local grounding improvement over `R009B`
- current decision:
  - do not launch the full `24`-case `R009` batch yet
  - first revise the core EIG controller and synthesis path:
    - utility scoring
    - mature-subgraph / claim-chain selection
    - benchmark-faithful final synthesis

## 2026-04-12: Graph-Critic Track Opened

- reframed the next EIG method stage around a learned graph critic rather than
  more hand-designed maturity thresholds
- added the canonical forward plan:
  - `docs/eig_graph_critic_plan.md`
- paused the full `R009` larger AIIB launch until the critic data/export path
  is ready
- preserved pre-critic `M1` and `R009` artifacts as pilot data for offline
  trajectory construction
- added new tracker entries:
  - `G001`: graph-critic planning
  - `G002`: trajectory export
  - `G003`: critic dataset construction
  - `G004`: text critic baseline
  - `G005`: graph critic
  - `G006`: critic-controlled AIIB pilot
- current next implementation target:
  - export graph states, actions, scores, and commit-vs-continue labels from
    saved pre-critic runs
