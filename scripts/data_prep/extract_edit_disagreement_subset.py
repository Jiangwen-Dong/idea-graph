from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import read_text_file, write_text_file


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(read_text_file(path))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in read_text_file(path).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain JSON objects per line.")
        rows.append(dict(payload))
    return rows


def _resolve_run_dir(base_dir: Path, repo_root: Path | None, run_dir_text: str) -> Path:
    run_dir = Path(run_dir_text)
    if run_dir.is_absolute():
        return run_dir
    manifest_relative = (base_dir / run_dir).resolve()
    if manifest_relative.exists():
        return manifest_relative
    if repo_root is not None:
        repo_relative = (repo_root / run_dir).resolve()
        if repo_relative.exists():
            return repo_relative
    return manifest_relative


def extract_edit_disagreements(
    *,
    run_manifest_path: Path,
    heuristic_baseline: str,
    critic_baseline: str,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    manifest_rows = _load_jsonl(run_manifest_path)
    manifest_dir = run_manifest_path.parent
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in manifest_rows:
        group_id = str(row.get("group_id", "")).strip()
        baseline_name = str(row.get("baseline_name", "")).strip()
        if not group_id or not baseline_name:
            continue
        grouped.setdefault(group_id, {})[baseline_name] = row

    disagreements: list[dict[str, Any]] = []
    for group_id, baselines in sorted(grouped.items()):
        heuristic_row = baselines.get(heuristic_baseline)
        critic_row = baselines.get(critic_baseline)
        if heuristic_row is None or critic_row is None:
            continue

        heuristic_run_dir = _resolve_run_dir(
            manifest_dir,
            repo_root,
            str(heuristic_row.get("run_dir", "")).strip(),
        )
        critic_run_dir = _resolve_run_dir(
            manifest_dir,
            repo_root,
            str(critic_row.get("run_dir", "")).strip(),
        )
        heuristic_summary = _load_json(heuristic_run_dir / "summary.json")
        critic_summary = _load_json(critic_run_dir / "summary.json")
        critic_graph = _load_json(critic_run_dir / "graph.json")
        critic_log = critic_graph.get("metadata", {}).get("runtime_controller_log", [])
        if not isinstance(critic_log, list):
            continue
        for entry in critic_log:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("selected_source", "")).strip() != "critic":
                continue
            disagreements.append(
                {
                    "group_id": group_id,
                    "instance_name": str(critic_row.get("instance_name", "")).strip(),
                    "round": str(entry.get("round", "")).strip(),
                    "role": str(entry.get("role", "")).strip(),
                    "override_margin": float(entry.get("override_margin", 0.0) or 0.0),
                    "heuristic_candidate_id": str(entry.get("heuristic_candidate_id", "")).strip(),
                    "critic_candidate_id": str(entry.get("selected_candidate_id", "")).strip(),
                    "heuristic_kind": str(entry.get("heuristic_kind", "")).strip()
                    or str(entry.get("heuristic_candidate", {}).get("kind", "")).strip(),
                    "critic_kind": str(entry.get("selected_kind", "")).strip()
                    or str(entry.get("selected_candidate", {}).get("kind", "")).strip(),
                    "heuristic_stop_reason": str(heuristic_summary.get("stop_reason", "")).strip(),
                    "critic_stop_reason": str(critic_summary.get("stop_reason", "")).strip(),
                    "heuristic_executed_round_count": int(heuristic_summary.get("executed_round_count", 0) or 0),
                    "critic_executed_round_count": int(critic_summary.get("executed_round_count", 0) or 0),
                    "critic_run_dir": str(critic_run_dir),
                    "heuristic_run_dir": str(heuristic_run_dir),
                }
            )
    return disagreements


def _write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    write_text_file(
        output_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract critic-vs-heuristic edit disagreements from saved controller packet runs.",
    )
    parser.add_argument("--run-manifest", required=True, help="Path to packet run_manifest.jsonl.")
    parser.add_argument("--output-path", required=True, help="Destination JSONL path.")
    parser.add_argument("--heuristic-baseline", default="ours-eig")
    parser.add_argument("--critic-baseline", default="ours-eig-critic-graph-twohead")
    parser.add_argument(
        "--repo-root",
        default="",
        help="Optional repository root used to resolve repo-relative run_dir values.",
    )
    args = parser.parse_args()

    disagreements = extract_edit_disagreements(
        run_manifest_path=Path(args.run_manifest),
        heuristic_baseline=str(args.heuristic_baseline).strip(),
        critic_baseline=str(args.critic_baseline).strip(),
        repo_root=Path(args.repo_root).resolve() if str(args.repo_root).strip() else None,
    )
    _write_jsonl(disagreements, Path(args.output_path))
    print(f"Wrote {len(disagreements)} disagreement rows to {Path(args.output_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
