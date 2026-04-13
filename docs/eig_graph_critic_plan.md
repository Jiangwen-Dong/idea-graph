# EIG Graph Critic Plan

This note is the canonical forward plan for the next EIG method track. It
supersedes the earlier heuristic-maturity refinement notes for new method
development, while preserving those runs as useful pilot data and historical
diagnosis.

Current local dataset layout guide:

- `docs/graph_critic_dataset_layout.md`

## Purpose

The paper studies scientific ideation as a structured, multi-step process. The
current heuristic EIG prototype already externalizes partial ideas as a graph,
but its utility and maturity rules are still hand-designed. That creates a weak
paper story: reviewers can reasonably ask whether the thresholds are tuned
engineering choices rather than a scalable method.

The next method direction is therefore:

> Learn a graph critic that scores candidate graph edits and a special
> `commit` action, so idea graph evolution becomes an adaptive structured
> decision process rather than a fixed maturity-threshold loop.

## Core Insight

Scientific ideas are not coherent from the outset. They emerge as fragmented
claims that must be connected, grounded, challenged, and repaired. A multi-agent
system should therefore preserve these fragments as a relational state instead
of collapsing them into plain text after each round.

The graph critic uses this relational state to answer two questions:

1. Which edit is most likely to improve the final proposal?
2. Is the graph ready to commit, or should it keep evolving?

This replaces the brittle question, "Has the graph passed a hand-designed
maturity threshold?"

## Optimization View

Let the benchmark-visible input be \(x\), and let the idea graph at round \(t\)
be:

\[
G_t = (V_t, E_t).
\]

Each candidate edit \(a\) belongs to a feasible edit set
\(\mathcal{A}(G_t)\). The set also contains a special `commit` action:

\[
a \in \mathcal{A}(G_t) \cup \{\mathrm{commit}\}.
\]

The graph critic estimates:

\[
Q_\theta(G_t, a, x),
\]

the expected downstream quality of taking action \(a\) from graph state
\(G_t\). The controller selects:

\[
a_t^* = \arg\max_{a \in \mathcal{A}(G_t) \cup \{\mathrm{commit}\}}
Q_\theta(G_t, a, x).
\]

If `commit` is selected, the system synthesizes the final proposal from the
selected claim chain. Otherwise, the selected edit is applied and the graph
continues evolving.

This casts maturity as an optimal-stopping decision rather than a thresholded
score.

## Model Sketch

The first graph critic should be lightweight and modular.

Inputs:

- node text embedding from a frozen sentence encoder or LLM embedding model
- node type embedding
- agent role embedding
- evidence/support/challenge indicators
- edge relation embedding
- edge resolved/unresolved flag
- benchmark input embedding
- candidate action type and target-node features

Architecture:

- encode the idea graph with a relational GNN or small graph transformer
- encode each candidate edit from its action type and target nodes
- score each candidate edit with a small MLP or cross-attention head
- include `commit` as a normal candidate action with its own embedding

The critic should learn decision quality, not generate scientific prose. LLM
agents still generate candidate edits and final text.

## Supervision

The repository already produces trajectories that can become offline training
data:

- graph states after each round
- generated candidate actions
- selected actions
- final proposal artifacts
- benchmark-native scores
- local development scores
- cost and round-count metadata

Use three complementary supervision signals.

### Return Regression

Train the critic to predict downstream proposal quality from a state-action
pair:

\[
Q_\theta(G_t, a_t, x) \approx R_{\mathrm{final}}.
\]

The target can combine benchmark-native score and human score when available.
For development, local evaluator scores can be used as weak labels, but they
must not become the final paper outcome.

### Pairwise Ranking

When two actions or two trajectories are available for the same or comparable
inputs, train:

\[
Q_\theta(G, a^+, x) > Q_\theta(G, a^-, x).
\]

Pairwise learning is useful because scientific-idea scores can be noisy, while
relative preference is often more stable.

### Commit-Vs-Continue

For intermediate graph states, synthesize from the current graph and compare
that score with the score after additional edits. This creates supervision for
the special `commit` action:

- if immediate synthesis is already better, `commit` should win
- if later graph evolution improves the proposal, an edit action should win

This directly replaces heuristic maturity stopping.

## Calibration

The commit head needs calibration because premature stopping and excessive
editing are both harmful.

Planned calibration checks:

- hold out a small validation set of graph states
- compare predicted commit preference with observed commit-vs-continue outcome
- report early-stop error and late-stop error
- optionally apply temperature scaling or a small validation-set threshold only
  to the commit margin

The paper should frame calibration as reliability control, not as benchmark
tuning.

## Evaluation And Ablations

The graph critic track should keep the existing benchmark protocol:

- benchmark-native automatic metrics are the primary automatic evidence
- human blind review remains the strongest final quality check
- graph-process metrics are supplementary mechanism analysis

Minimum critic ablations:

- `ours-eig-heuristic`
  current heuristic utility/maturity controller
- `ours-eig-critic-text`
  learned critic using flattened graph text without graph structure
- `ours-eig-critic-graph`
  learned critic using the structured graph encoder
- `ours-eig-critic-no-commit`
  graph critic selects edits, but stopping uses a fixed round budget
- `ours-eig-critic-calibrated`
  graph critic with calibrated commit decision

Key reviewer-facing tests:

- graph structure matters if `critic-graph` outperforms `critic-text`
- learned control matters if `critic-graph` outperforms `heuristic`
- adaptive stopping matters if `critic-graph` outperforms `critic-no-commit`
- calibration matters if it reduces premature and late stopping

## Staged Roadmap

### Stage G0: Documentation And Data Audit

- freeze the current heuristic EIG runs as pre-critic pilot data
- record which output artifacts contain graph states, actions, and final scores
- update active docs to mark the full `R009` launch as paused

### Stage G1: Trajectory Export

- add a trajectory-export script that converts saved runs into state-action
  examples
- include graph JSON, candidate action metadata, selected action, final score,
  and commit-vs-continue labels when available
- verify export on the existing small `M1` and `R009` pilot outputs

### Stage G2: Critic Dataset Construction

- build train/validation splits by benchmark instance, not by individual state
- prevent leakage across trajectories from the same benchmark case
- store weak labels separately from benchmark-native and human labels

### Stage G2.5: Candidate-Slate Dataset

- derive feasible candidate slates from frozen `G1` and `G2` artifacts
- keep `G2` immutable, so label/split construction remains auditable
- add `commit` as an explicit candidate action for every graph state
- store stable candidate IDs, model-facing candidate text, and state-level
  candidate counts
- current full derived dataset:
  `outputs/graph_critic_datasets/03_archive/current_benchmarked_ours_eig_full_g25`
  with `910` states and `9456` candidates
- expanded development-pool dataset:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25`
  with `1267` states, `13004` candidates, and `72` commit-positive
  terminal states

### Stage G3: Text Critic Pilot

- train a modest text-only scorer over `state_text [SEP] candidate_text`
- evaluate ranking quality on held-out benchmark-instance groups
- treat this as a low-data logged-edit imitation pilot that serves as a simple
  baseline and data-sanity check, not as final learned controller evidence
- current pilot artifact:
  `outputs/graph_critic_models/current_benchmarked_ours_eig_full_g3_text_pilot`
  with validation top-1 accuracy `0.724` and mean reciprocal rank `0.841`
- important limitation: the current train and validation splits contain zero
  positive `commit` labels even though every candidate slate includes one
  explicit `commit` action, so this pilot does not yet demonstrate learned
  commit control and should be read as a supervision-limited baseline

### Stage G3.5: Partition Manifest Layer

- add a deterministic group-level partition manifest over the existing `G2`
  split rows
- keep the assignment unit at benchmark-instance group granularity, never
  individual state rows
- map source `train` / `validation` into `critic_train` / `critic_dev`
  explicitly, with optional `paper_eval` holdouts
- current artifact:
  `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g35_partitions`
  with `11` groups, `9` `critic_train`, `2` `critic_dev`, and no `paper_eval`
  groups yet
- current diagnostic conclusion:
  the repo can rediscover the full current `60`-run `ours-eig` pool from the
  active `outputs/` tree, so the older `48`-run commit-enriched export should
  be treated as a stale partial rebuild rather than a true missing-run problem

### Stage G3.6: Full-Pool Commit-Enriched Refresh

- regenerate the commit-enriched `G1 / G2 / G2.5` stack from the current full
  `60`-run `ours-eig` pool
- refreshed artifacts:
  - `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g1_commit_enriched`
  - `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g2_commit_enriched`
  - `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g25_commit_enriched`
- refreshed counts:
  - `G1` run count `60`, transition count `910`, terminal commit states `60`
  - `G2` run count `60`, transition count `910`
  - `G2.5` state count `970`, candidate count `10092`, commit-positive count
    `60`
- rerun the text critic on the refreshed `G2.5` artifact:
  `outputs/graph_critic_models/current_benchmarked_ours_eig_full_g3_text_pilot_commit_enriched`
- refreshed `G3` pilot result:
  - validation top-1 accuracy `0.736`
  - mean reciprocal rank `0.848`
  - crucially, `train_commit_positive_count = 55` and
    `validation_commit_positive_count = 5`
- practical conclusion:
  the text critic still remains a modest pilot, but the supervision gap that
  previously blocked commit-learning evidence is now materially reduced

### Stage G3.7: Split Registry And Frozen Development Pool

- freeze the current 11-group partition artifact as:
  - `development_pool_v1`
- generate a canonical split registry:
  - `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g35_partitions/split_registry.jsonl`
  - `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g35_partitions/split_registry_stats.json`
- current registry status:
  - `11` rows
  - pool name `development_pool_v1`
  - `critic_train = 9`
  - `critic_dev = 2`
  - `paper_eval = 0`
- practical conclusion:
  the current graph-critic pool is now explicitly documented as development
  data only, which makes future learned-controller evaluation easier to defend

### Stage G3.8: First Untouched Paper-Eval Candidate Pool

- define the first proposed untouched benchmark list:
  - `outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v1/candidate_instances.json`
- current candidate composition:
  - `AI_Idea_Bench_2025`: 6 proposed untouched instances
  - `LiveIdeaBench`: 4 proposed untouched instances
- practical conclusion:
  the repo now distinguishes development-pool instances from future frozen
  paper-eval instances, even before those held-out runs are launched

### Stage G4: Offline Warm-Start Text Critic

- train the first partition-aware offline warm-start scorer over the refreshed
  full-pool `G2.5` commit-enriched dataset
- enforce clean split usage:
  - `critic_train` for training
  - `critic_dev` for validation
  - ignore `paper_eval`
- warm-start artifact:
  `outputs/graph_critic_models/current_benchmarked_ours_eig_full_g4_text_warmstart`
- current warm-start result:
  - validation top-1 accuracy `0.755`
  - mean reciprocal rank `0.860`
  - `train_commit_positive_count = 55`
  - `validation_commit_positive_count = 5`
- practical conclusion:
  this is still a lightweight text scorer, but it is a stronger and cleaner
  offline controller baseline than the earlier plain `G3` pilot

### Stage G4.5: Controller-Safety Layer

- add a replay buffer that accepts only `critic_train` online episodes
- add a conservative policy wrapper that:
  - blocks early `commit`
  - requires a positive commit margin and confidence threshold
  - falls back to the heuristic when critic edit margins are small
- current implementation state:
  - `src/idea_graph/critic_replay.py`
  - `src/idea_graph/critic_policy.py`
  - verified by `tests/test_critic_replay.py` and
    `tests/test_critic_policy.py`
- practical conclusion:
  the learned controller now has a safe decision shell, so the next step is a
  tiny batched online adaptation pilot rather than more heuristic-policy work

### Stage G4.6: Tiny Online Adaptation Runner

- add a minimal runner that consumes:
  - offline `critic_train` candidate rows
  - `critic_train` replay rows in candidate-slate format
  - frozen `critic_dev` evaluation rows
- current implementation:
  - `scripts/run_online_text_critic_adaptation.py`
  - adaptation helpers in `src/idea_graph/online_text_critic.py`
- important replay refinement:
  online episodes must preserve candidate-slate rows, not only chosen actions,
  because the next-action critic requires state-local negatives during online
  retraining
- first smoke status:
  - bootstrap replay from recycled offline train rows runs successfully end to
    end
  - but it degrades dev performance relative to the offline warm-start
- practical conclusion:
  the runner plumbing is ready, but meaningful online adaptation now depends on
  collecting **new** train-group episodes rather than reusing old offline
  candidate rows as pseudo-online data

### Stage G4.7: Critic-Train Episode Collection Runner

- add a thin collection layer above the frozen split registry:
  - `src/idea_graph/critic_episode_collection.py`
  - `scripts/collect_critic_train_episodes.py`
- keep the contract narrow:
  - read `split_registry.jsonl`
  - select only `development_pool_v1 / critic_train`
  - require `train_online_critic` inside `allowed_usages`
  - emit a human-auditable launch manifest before execution
- current verified artifacts:
  - dry-run manifest smoke:
    `outputs/graph_critic_online_episodes/development_pool_v1_critic_train_manifest_smoke`
  - deterministic execute smoke:
    `outputs/graph_critic_online_episodes/development_pool_v1_critic_train_execute_smoke_det`
- current smoke conclusions:
  - dry run resolves the intended frozen train pool exactly:
    - `9` groups total
    - `6` `AI_Idea_Bench_2025`
    - `3` `LiveIdeaBench`
  - deterministic execute smoke completes end to end on one case, writing:
    - `launch_manifest.jsonl`
    - `collection_config.json`
    - `collection_summary.json`
    - `execution_results.jsonl`
    - per-run stdout/stderr logs
    - dedicated `runs/` artifacts
  - successful execute rows are now profiled with:
    - token counts when traces exist
    - estimated cost when available
    - final local score
    - final native score when available
- practical conclusion:
  the repo now has a reviewer-safe bridge from the frozen development split to
  fresh online episodes; the next step is a real openai-compatible
  `critic_train` collection slice, then replay export and frozen `critic_dev`
  comparison

### Stage G4.8: Real Critic-Train Collection And Online Adaptation

- launched the first real openai-compatible collection packet over the frozen
  `development_pool_v1 / critic_train` groups:
  - artifact:
    `outputs/graph_critic_online_episodes/development_pool_v1_critic_train_qwen_v1`
  - completed groups:
    - `9 / 9`
    - `6` `AI_Idea_Bench_2025`
    - `3` `LiveIdeaBench`
  - traced token total:
    - `1,639,113`
- converted the collected runs into a replay-ready candidate-slate online
  buffer using the existing `G1` export plus forced-`critic_train`
  reconstruction:
  - `G1` export:
    `outputs/graph_critic_datasets/01_active_text_critic/development_pool_v1_critic_train_qwen_v1_g1`
  - replay buffer:
    `outputs/graph_critic_online_episodes/development_pool_v1_critic_train_qwen_v1/online_replay_buffer.jsonl`
  - replay stats:
    - `204` states
    - `1851` candidate rows
    - `9` positive terminal commit states
    - all rows tagged `partition_role=critic_train`
- reran online adaptation on frozen `critic_dev` using the real online buffer:
  - output artifact:
    `outputs/graph_critic_models/current_benchmarked_ours_eig_full_g46_text_online_real_train_v1`
  - adaptation config:
    - `offline_fraction = 0.85`
    - `random_seed = 0`
  - baseline warm-start dev metrics:
    - top-1 `0.7545`
    - MRR `0.8597`
  - adapted dev metrics:
    - top-1 `0.7727`
    - MRR `0.8697`
- practical conclusion:
  unlike the earlier pseudo-online bootstrap smoke, real `critic_train`
  episodes improve frozen `critic_dev` ranking quality for the lightweight text
  critic; this is the first positive end-to-end result for the online
  adaptation line

### Stage G4.9: Development-Pool Expansion Infrastructure

- status:
  - implemented and synced into the main checkout
  - API-backed train/dev expansion collection completed
  - combined G1/G2/G2.5 datasets materialized
- purpose:
  - address the main critic-dataset bottleneck: too few leakage-safe
    benchmark-instance groups
  - keep `development_pool_v1` frozen while creating a separate
    development-only expansion pool
- added tooling:
  - role-aware episode selection:
    `select_pool_rows(...)`
  - `collect_critic_train_episodes.py` flags:
    - `--partition-role`
    - `--required-usage`
  - split override support:
    - `build_graph_critic_dataset.py --split-overrides`
  - overlap-safe expansion pool builder:
    `scripts/build_critic_expansion_pool.py`
- new development-only candidate pool:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_candidate_pool_v1`
- pool size:
  - `12` unique groups
  - `8` `critic_train`
  - `4` `critic_dev`
  - `8` AIIB groups
  - `4` LiveIdeaBench groups
- verified dry-run manifests:
- completed real collections:
  - `outputs/graph_critic_online_episodes/development_pool_v2_critic_train_qwen_v1`
    - `8` selected groups
    - `8` completed groups
    - `1,623,590` traced tokens
  - `outputs/graph_critic_online_episodes/development_pool_v2_critic_dev_qwen_v1`
    - `4` selected groups
    - `4` completed groups
    - `777,552` traced tokens
- combined datasets:
  - G1:
    `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g1`
    with `72` runs and `1195` transitions
  - G2:
    `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g2`
    with `23` groups, `17` train groups, `6` validation groups, and
    `1195` transitions
  - G2.5:
    `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25`
    with `1267` states and `13004` candidate rows
  - readiness report:
    `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_readiness/training_readiness_report.md`
- current conclusion:
  the expanded dataset is now adequate for a first offline graph-feature
  scorer comparison against the text scorer, but it remains development-only
  and should not yet support final learned-controller benchmark claims

### Stage G5: Graph Critic

- implement a first graph-feature action scorer on the frozen
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25` split
- compare against the refreshed text scorer on held-out development groups
- keep learned `commit` out of runtime until edit-action ranking is stable
- only promote graph critic into controller-in-the-loop generation after it
  beats the text scorer offline on the same candidate slates

### Stage G5.1: First Offline Graph-Feature Baseline Gate

- added the first lightweight graph-feature scorer implementation:
  - `src/idea_graph/graph_feature_critic.py`
  - `scripts/train_graph_feature_critic.py`
  - `tests/test_graph_feature_critic.py`
- frozen comparison protocol:
  - candidate-slate root:
    `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25`
  - snapshot root:
    `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g1`
  - partition manifest:
    `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g2_partitions/partition_manifest.jsonl`
  - same `critic_train` / `critic_dev` groups, same candidate slates, same
    positive commit weighting as the refreshed text scorer
- refreshed text scorer artifact:
  `outputs/graph_critic_models/development_pool_v2_text_warmstart_v1`
  - top-1 `0.7081`
  - MRR `0.8147`
- first graph-feature scorer artifact:
  `outputs/graph_critic_models/development_pool_v2_graph_feature_v1`
  - top-1 `0.5024`
  - MRR `0.6824`
- practical conclusion:
  - the first pure graph-feature baseline does **not** clear the offline gate
  - graph-critic runtime testing should remain blocked
  - the current graph line is still development-only evidence, not controller
    replacement evidence
- implication for the roadmap:
  - do **not** spend new benchmark-generation budget on a graph-critic runtime
    controller yet
  - next graph-critic work should strengthen representation quality first,
    likely through either:
    - hybrid text-plus-graph features, or
    - a richer learned graph encoder with node-text integration
  - keep the long-term controller framing intact: the graph critic still aims
    to rank full next-action candidates, including `commit`, but the current
    offline result shows that the first structured baseline is not yet ready

### Stage G6: Controlled Generation Pilot

- plug the critic into action selection and commit decisions
- run the same 4-case AIIB gate before launching larger batches
- first implemented variant:
  - `ours-eig-critic-text`
  - learned text critic used only as a conservative edit reranker
  - heuristic stop / maturity kept unchanged
- first gate artifact:
  `outputs/m2_aiib_g48_controller_gate_v1`
- first gate summary:
  `outputs/m2_aiib_g48_controller_gate_v1/paired_summary.md`
- first gate result:
  - mean local overall:
    - `ours-eig = 5.27`
    - `ours-eig-critic-text = 5.20`
  - mean local benchmark alignment:
    - `ours-eig = 3.46`
    - `ours-eig-critic-text = 3.37`
  - mean AIIB native average:
    - `ours-eig = 7.93`
    - `ours-eig-critic-text = 7.79`
- practical conclusion:
  the learned reranker is a real end-to-end pilot now, but not yet a stable
  win over heuristic EIG. The strongest warning sign is that on case `3883`
  the altered edit ranking caused the unchanged heuristic maturity rule to stop
  at `Round2` instead of `Round4`, slightly improving local scores while
  hurting native score by `1.15`.
- post-gate cleanup:
  - future LLM-backed controller runs now persist `runtime_controller_log`
    entries, with regression coverage in `tests/test_engine.py`
- immediate next requirement before any larger controller comparison:
  - use the patched runtime-controller trace path in future LLM-backed runs
  - use the maturity-sensitive safety guard around reranked edits
  - rerun the same frozen 4-case AIIB gate before any broader paper packet
- current safety update:
  - `ScoredCandidate` carries gain metadata from the engine simulation path
  - the runtime text critic receives current support and contradiction
    features from the reference subgraph
  - critic edit overrides that would make the graph newly mature are blocked
    when they do not add enough support evidence
  - learned `commit` remains disabled, so this patch only makes edit
    reranking safer near the heuristic maturity boundary
- verified regression packet:
  `python -m pytest tests/test_engine.py tests/test_runtime_critic.py tests/test_benchmark_mode_and_baselines.py tests/test_online_text_critic.py tests/test_critic_policy.py tests/test_critic_replay.py tests/test_critic_episode_collection.py tests/test_critic_split_registry.py -q`
  passed with `81 passed`
- second frozen gate artifact:
  `outputs/m2_aiib_g48_controller_gate_v2`
- second frozen gate result:
  - the `3883` early-stop symptom improved from `Round2` in V1 to `Round4`
    in V2
  - mean local overall changed from `5.23` for `ours-eig` to `5.29` for
    `ours-eig-critic-text`
  - mean local benchmark alignment changed from `3.39` to `3.56`
  - mean AIIB native average changed from `8.00` to `7.93`
  - controller traces are now present for all critic runs
- practical conclusion:
  the safety patch is useful, but the text critic remains a mixed pilot rather
  than a main-system replacement; the next learned-controller progress should
  prioritize trace diagnosis, development-pool expansion, and offline
  graph-critic comparison

### Stage G5.2: Relation-Aware Offline Graph Critic

- artifact:
  `outputs/graph_critic_models/development_pool_v2_relation_graph_sanitized_v1`
- frozen protocol:
  - candidate slates:
    `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25`
  - state snapshots:
    `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g1`
  - partition manifest:
    `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g2_partitions/partition_manifest.jsonl`
- representation note:
  - relation-aware two-layer message passing over typed nodes and typed edges
  - frozen sentence embeddings for node text, state text, and leakage-safe
    candidate text
  - state-local candidate ranking loss with edit-only and all-candidate
    validation views
- trusted metrics:
  - all-candidate top-1: `0.8373`
  - all-candidate MRR: `0.8951`
  - edit-only top-1: `0.8550`
  - edit-only MRR: `0.9089`
- comparison:
  - refreshed text scorer:
    `outputs/graph_critic_models/development_pool_v2_text_warmstart_v1`
    - top-1 `0.7081`
    - MRR `0.8147`
  - first graph-feature scorer:
    `outputs/graph_critic_models/development_pool_v2_graph_feature_v1`
    - top-1 `0.5024`
    - MRR `0.6824`
- cleanup note:
  - discard `outputs/graph_critic_models/development_pool_v2_relation_graph_v1`
    from decision-making because it was trained before the candidate-text
    leakage cleanup and is therefore not trustworthy
- decision:
  - the relation-aware graph critic now clears the offline gate against the
    frozen development split
  - next step can move to a narrow controller gate with the learned graph
    scorer, while keeping learned `commit` conservative and benchmark spending
    small until end-to-end quality is revalidated

### Stage G7: Paper Experiments

- rerun the main comparison only after the critic pilot is stable
- include critic ablations and calibration analysis

## Open Risks

- Saved trajectories may not contain enough alternative actions for strong
  pairwise learning. Mitigation: generate additional candidate actions offline
  without running full new proposals.
- Benchmark-native scores are noisy. Mitigation: use ranking losses and cache
  judge outputs carefully.
- The critic could learn evaluator artifacts rather than scientific quality.
  Mitigation: keep human review and cross-benchmark validation.
- A graph encoder may not beat a strong text critic at small data scale.
  Mitigation: report this honestly; the graph representation can still be
  justified through process transparency and ablations.
- Implementation cost may delay paper experiments. Mitigation: keep
  `ours-eig-heuristic` as a documented fallback and ablation, not as the main
  final method if the critic succeeds.
