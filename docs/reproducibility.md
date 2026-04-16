# Reproducibility And Release Hygiene

This note records the current paper-facing repository policy for the parallel
EIG v2 line. It is intended for the GitHub supplementary-code snapshot, not as
a replacement for experiment logs under `outputs/`.

## Active Method

- The active EIG runtime is `parallel_graph_v2`.
- The old sequential EIG loop is retained only for historical comparison and
  backward-compatible tests.
- The current teacher for critic-label curation is the parallel-v2 heuristic
  controller.
- The learned critic path should use the two-head graph critic:
  - one shared graph encoder
  - an edit/action head for role-local action selection
  - a commit head for post-round commit prediction

## Frozen Splits

Tracked split manifests live in `data/splits/parallel_v2`.

- `critic_train_dev_registry.jsonl`
  contains 400 development groups: 300 `critic_train` and 100 `critic_dev`.
- `critic_train_dev_split_overrides.jsonl`
  maps `critic_train -> train` and `critic_dev -> validation`.
- `paper_eval_v2_registry.jsonl`
  contains 256 final paper-eval groups.
- `paper_eval_v2_disjointness_audit.json`
  records zero overlap between the paper-eval pool and blocked development
  groups used for critic training and calibration.

The `critic_train` rows may be used for heuristic-teacher replay collection and
critic training. The `critic_dev` rows may be used for checkpoint selection and
commit-head calibration. The `paper_eval_v2` rows must remain untouched until
the method and baselines are frozen.

## Generated Artifacts

Large generated files are intentionally ignored by Git.

- `outputs/` stores run outputs, replay exports, model checkpoints, and
  evaluation packets.
- `data/benchmarks/` stores downloaded benchmark assets.
- `C:\eig_p2v2_harvest` may store local full-harvest copies.
- `.worktrees/` stores temporary development worktrees.

For release, preserve lightweight manifests and code in Git, but publish large
result artifacts separately if needed.

## Baseline Policy

The main paper table should use true or benchmark-faithful baselines.

- `direct` and `self-refine` are controlled local baselines and may be reported.
- `ai-researcher`, `scipip`, and `virsci` are external baseline entrypoints.
  They require separately installed upstream repositories and a configured
  `configs/external_baselines.example.json` derivative.
- `ai-researcher-proxy`, `scipip-proxy`, and `virsci-proxy` are diagnostic
  local approximations. They must not silently replace exact baselines in the
  headline paper table.
- The `openai-compatible-bridge` for AI-Researcher-style runs is useful for
  DashScope/Qwen development, but it should be labeled as a bridge setting
  unless it exactly follows the upstream paper implementation.

The repository does not track `.tmp-baselines/*` upstream clones. Users should
install external baselines locally and point the example config to those paths.

## Worktree Policy

Current branch hygiene policy:

- Merge `parallel-runtime-v2-exec` into `main`; this is the forward branch.
- Keep dirty historical worktree changes backed up as local patch files before
  removing any old worktree.
- Do not merge old divergent critic-dataset branches directly into `main`.
- After `main` verifies successfully, old clean worktrees may be removed as
  local cleanup because their history remains available through Git branches.

The local patch-backup folder is ignored by Git so safety patches do not pollute
the supplementary-code snapshot.

## Secret Policy

Do not commit API keys or provider-specific private config files.

- Use environment variables such as `DASHSCOPE_API_KEY`.
- Start from `configs/openai_compatible.example.json` and
  `configs/external_baselines.example.json`.
- Keep local provider files such as `configs/external_baselines.qwen.json`
  untracked.
