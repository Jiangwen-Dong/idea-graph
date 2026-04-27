from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from idea_graph.collaboration_protocol import ROUND_PHASES
from idea_graph.critic_curation import CuratedDatasetQuotas, curate_two_head_critic_dataset
from idea_graph.engine import build_seed_graphs, generic_candidate_action_specs, merge_seed_graphs
from idea_graph.models import IdeaGraph


def _jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _base_edit_row(
    *,
    state_id: str,
    candidate_index: int,
    kind: str,
    round_name: str,
    role: str = "MechanismProposer",
    selected: bool = False,
) -> dict[str, object]:
    return {
        "state_id": state_id,
        "candidate_id": f"{state_id}::candidate:{candidate_index:04d}",
        "group_id": f"group::{state_id}",
        "split": "train",
        "benchmark": "unit",
        "instance_name": state_id,
        "run_dir": f"run::{state_id}",
        "step_index": candidate_index,
        "round_name": round_name,
        "role": role,
        "state_kind": "parallel_pre_action",
        "state_text": "nodes=3;edges=1;contradictions=0",
        "candidate_index": candidate_index,
        "candidate_count": 3,
        "candidate_kind": kind,
        "candidate_target_ids": [] if kind in {"skip", "freeze_branch"} else ["N001"],
        "candidate_payload": {"branch_id": "B001"},
        "candidate_source": "unit",
        "candidate_text": f"kind={kind}",
        "is_commit": False,
        "is_logged_selected": selected,
        "before_state_snapshot": "parallel_state_snapshots/unit.json",
    }


def _base_commit_row(
    *,
    state_id: str,
    round_name: str,
    label: int,
) -> dict[str, object]:
    return {
        "state_id": state_id,
        "group_id": f"group::{state_id}",
        "split": "train",
        "benchmark": "unit",
        "instance_name": state_id,
        "run_dir": f"run::{state_id}",
        "round_name": round_name,
        "role": "CommitController",
        "state_kind": "parallel_post_round",
        "state_text": "nodes=4;edges=2;contradictions=0",
        "before_state_snapshot": "post_round_commit_state_snapshots/unit.json",
        "support_coverage": 0.8,
        "unresolved_contradiction_ratio": 0.0,
        "utility": 0.7,
        "commit_label": label,
    }


def _write_small_source_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "source"
    dataset_dir.mkdir()
    edit_rows = [
        _base_edit_row(
            state_id="s-structure",
            candidate_index=0,
            kind="add_support_edge",
            round_name="Round1",
            selected=True,
        ),
        _base_edit_row(
            state_id="s-structure",
            candidate_index=1,
            kind="skip",
            round_name="Round1",
        ),
        _base_edit_row(
            state_id="s-structure",
            candidate_index=2,
            kind="freeze_branch",
            round_name="Round1",
        ),
        _base_edit_row(
            state_id="s-repair",
            candidate_index=0,
            kind="propose_repair",
            round_name="Round3",
            selected=True,
        ),
        _base_edit_row(
            state_id="s-repair",
            candidate_index=1,
            kind="skip",
            round_name="Round3",
        ),
        _base_edit_row(
            state_id="s-repair",
            candidate_index=2,
            kind="freeze_branch",
            round_name="Round3",
        ),
    ]
    commit_rows = [
        _base_commit_row(state_id="c-continue", round_name="Round2", label=0),
        _base_commit_row(state_id="c-commit", round_name="Round3", label=1),
    ]
    (dataset_dir / "edit_head_rows.jsonl").write_text(_jsonl(edit_rows), encoding="utf-8")
    (dataset_dir / "commit_head_rows.jsonl").write_text(_jsonl(commit_rows), encoding="utf-8")
    return dataset_dir


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_active_round_phases_do_not_include_freeze_branch() -> None:
    for phase in ROUND_PHASES.values():
        assert "freeze_branch" not in phase.allowed_actions


def test_generic_candidates_do_not_emit_freeze_branch() -> None:
    graph = IdeaGraph(topic="Test topic", literature=["paper"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)
    branch = next(branch for branch in graph.branches.values() if branch.role == "MechanismProposer")

    candidates = generic_candidate_action_specs(graph, "Round3", "MechanismProposer", branch)

    assert candidates
    assert all(candidate["kind"] != "freeze_branch" for candidate in candidates)


def test_curator_relabels_one_positive_per_edit_state_and_filters_freeze(tmp_path: Path) -> None:
    source = _write_small_source_dataset(tmp_path)
    result = curate_two_head_critic_dataset(
        dataset_dir=source,
        output_dir=tmp_path / "out",
        dataset_name="curated",
        quotas=CuratedDatasetQuotas(
            edit_phase_kind={("structure", "add_support_edge"): 1, ("repair", "skip"): 1},
            commit_round_label={("Round2", 0): 1, ("Round3", 1): 1},
        ),
    )

    rows = _read_jsonl(result.dataset_dir / "edit_head_rows.jsonl")
    positive_counts = Counter(row["candidate_kind"] for row in rows if row["is_logged_selected"])

    assert result.edit_label_count == 2
    assert all(row["candidate_kind"] != "freeze_branch" for row in rows)
    assert positive_counts == {"add_support_edge": 1, "skip": 1}
    for state_id in {row["state_id"] for row in rows}:
        state_rows = [row for row in rows if row["state_id"] == state_id]
        assert sum(1 for row in state_rows if row["is_logged_selected"]) == 1


def test_curator_writes_audit_counts(tmp_path: Path) -> None:
    source = _write_small_source_dataset(tmp_path)
    result = curate_two_head_critic_dataset(
        dataset_dir=source,
        output_dir=tmp_path / "out",
        dataset_name="curated",
        quotas=CuratedDatasetQuotas(
            edit_phase_kind={("structure", "add_support_edge"): 1, ("repair", "skip"): 1},
            commit_round_label={("Round2", 0): 1, ("Round3", 1): 1},
        ),
    )

    audit = json.loads((result.dataset_dir / "curation_audit.json").read_text(encoding="utf-8"))

    assert audit["edit_label_count"] == 2
    assert audit["commit_label_count"] == 2
    assert audit["edit_positive_counts"]["add_support_edge"] == 1
    assert audit["edit_positive_counts"]["skip"] == 1
    assert audit["commit_label_counts"]["0"] == 1
    assert audit["commit_label_counts"]["1"] == 1
