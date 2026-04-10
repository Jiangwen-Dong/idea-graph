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
| `R005-mini` | `M1` | refreshed current-codepath mini packet | `direct`, `self-refine`, `ai-researcher`, `ours-eig` | AI Idea Bench index `13` + LiveIdeaBench row `0` | native metrics + refreshed local quality | MUST | DONE | batch dir `outputs/quality_batches/20260410-181953-refreshed-m1-mini-current-codepath`; exact `ai-researcher` bridge now included in the batch runner; `ours-eig` is strongest on AI Idea Bench and highest by combined local overall, while `self-refine` remains strongest on the current LiveIdeaBench meteorology case |
| `R005-mini-v2` | `M1` | targeted weak-context regression-fix mini packet | `direct`, `self-refine`, `ai-researcher`, `ours-eig` | AI Idea Bench index `13` + LiveIdeaBench row `0` | native metrics + refreshed local quality | MUST | DONE | batch dir `outputs/quality_batches/20260410-185224-refreshed-m1-mini-weak-context-v3`; after fixing AI Idea Bench postprocess grounding, `ours-eig` recovered on AIIB (`5.92/4.33/8.57`) while staying strongest on LiveIdeaBench (`7.11/6.15/7.05`), making it the strongest method on both refreshed mini-batch cases under the local benchmark-facing evaluator |
| `R005` | `M1` | cross-benchmark smoke batch | `direct` | `4` AIIB + `4` Live | benchmark-native automatic metrics | MUST | DONE | shared batch dir `outputs/quality_batches/20260410-192309-m1-4x4-main-slice`; aggregate overall `5.38`, alignment `3.51`, native `7.61` |
| `R006` | `M1` | cross-benchmark smoke batch | `self-refine` | `4` AIIB + `4` Live | benchmark-native automatic metrics | MUST | DONE | shared batch dir `outputs/quality_batches/20260410-192309-m1-4x4-main-slice`; aggregate overall `5.79`, alignment `4.19`, native `7.47`; strongest native single-agent baseline in the final `M1` packet |
| `R007` | `M1` | cross-benchmark smoke batch | `ai-researcher` | `4` AIIB + `4` Live | benchmark-native automatic metrics | MUST | DONE | shared batch dir `outputs/quality_batches/20260410-192309-m1-4x4-main-slice`; aggregate overall `4.65`, alignment `2.27`, native `7.26`; bridge path remained stable throughout the full packet |
| `R008` | `M1` | cross-benchmark smoke batch | `ours-eig` | `4` AIIB + `4` Live | benchmark-native automatic metrics + graph diagnostics | MUST | DONE | shared batch dir `outputs/quality_batches/20260410-192309-m1-4x4-main-slice`; aggregate overall `6.46`, alignment `5.08`, graph `9.07`, native `6.78`; strongest by the local benchmark-facing scorer, but not by the benchmark-native averages |
| `R009` | `M2` | core automatic batch | shortlisted main baselines | at least `24` AIIB | native AIIB metrics including matched-pool `IC` | MUST | TODO | only launch after `M1` is clean |
| `R010` | `M2` | core automatic batch | shortlisted main baselines | at least `24` LiveIdeaBench | native LiveIdeaBench metrics | MUST | TODO | keep the same shared output contract |
| `R011` | `M4` | human blind review packet | `direct`, `self-refine`, `ai-researcher`, `ours-eig` | balanced `8` to `12` cases per benchmark | human rubric | MUST | TODO | anonymize system names and standardize formatting |
| `R012` | `M5` | mechanism ablation | `ours-eig` without utility-guided ranking | balanced `6 + 6` subset | native metrics + graph metrics | MUST | TODO | tests whether utility-guided selection matters |
| `R013` | `M5` | mechanism ablation | `ours-eig` without maturity stop | balanced `6 + 6` subset | native metrics + rounds-to-maturity | MUST | TODO | tests whether current stopping is helping or harming |
| `R014` | `M5` | mechanism ablation | `ours-eig` with flat final synthesis | balanced `6 + 6` subset | native metrics + graph metrics | MUST | TODO | tests mature-subgraph synthesis directly |
| `R015` | `M5` | appendix analysis | `ours-eig` plus main baselines | reuse `M2` outputs | process metrics, cost, runtime, fallback statistics | MUST | TODO | report separately from the main quality table |

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
