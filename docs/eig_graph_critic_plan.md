# EIG Graph Critic Plan

This note is the canonical forward plan for the next EIG method track. It
supersedes the earlier heuristic-maturity refinement notes for new method
development, while preserving those runs as useful pilot data and historical
diagnosis.

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
  `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g25`
  with `910` states and `9456` candidates

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

### Stage G4: Graph Critic

- implement the graph encoder and action scorer
- compare against the text critic on held-out trajectories

### Stage G5: Controlled Generation Pilot

- plug the critic into action selection and commit decisions
- run the same 4-case AIIB gate before launching larger batches

### Stage G6: Paper Experiments

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
