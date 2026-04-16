from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_critic_train_episodes, run_controller_eval_packet, run_pipeline, run_quality_batch


class ScriptDefaultPathTests(unittest.TestCase):
    def test_run_pipeline_default_benchmark_root_prefers_shared_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / "idea-graph"
            worktree_root = repo_root / ".worktrees" / "parallel-runtime-v2-exec"
            expected = repo_root / "data" / "benchmarks"
            expected.mkdir(parents=True, exist_ok=True)
            worktree_root.mkdir(parents=True, exist_ok=True)

            with patch("scripts.run_pipeline.ROOT", worktree_root):
                parser = run_pipeline.build_parser()

            args = parser.parse_args([])
            self.assertEqual(args.benchmark_root, expected)

    def test_collect_critic_train_episodes_default_benchmark_root_prefers_shared_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / "idea-graph"
            worktree_root = repo_root / ".worktrees" / "parallel-runtime-v2-exec"
            expected = repo_root / "data" / "benchmarks"
            expected.mkdir(parents=True, exist_ok=True)
            worktree_root.mkdir(parents=True, exist_ok=True)

            with patch("scripts.collect_critic_train_episodes.ROOT", worktree_root):
                parser = collect_critic_train_episodes.build_parser()

            args = parser.parse_args(["--split-registry", "dummy.jsonl", "--output-dir", "out"])
            self.assertEqual(args.benchmark_root, expected)

    def test_run_quality_batch_defaults_prefer_shared_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / "idea-graph"
            worktree_root = repo_root / ".worktrees" / "parallel-runtime-v2-exec"
            expected_base = repo_root / "data" / "benchmarks"
            (expected_base / "ai_idea_bench_2025").mkdir(parents=True, exist_ok=True)
            (expected_base / "liveideabench").mkdir(parents=True, exist_ok=True)
            worktree_root.mkdir(parents=True, exist_ok=True)

            with patch("scripts.run_quality_batch.ROOT", worktree_root):
                parser = run_quality_batch.build_parser()

            args = parser.parse_args(["--llm-config", "dummy.json"])
            self.assertEqual(args.ai_benchmark_root, expected_base / "ai_idea_bench_2025")
            self.assertEqual(args.live_benchmark_root, expected_base / "liveideabench")

    def test_run_controller_eval_packet_default_benchmark_root_prefers_shared_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / "idea-graph"
            worktree_root = repo_root / ".worktrees" / "parallel-runtime-v2-exec"
            expected = repo_root / "data" / "benchmarks"
            expected.mkdir(parents=True, exist_ok=True)
            worktree_root.mkdir(parents=True, exist_ok=True)

            with patch("scripts.run_controller_eval_packet.ROOT", worktree_root):
                parser = run_controller_eval_packet.build_parser()

            args = parser.parse_args(["--packet-manifest", "packet.jsonl", "--baselines", "ours-eig", "--output-root", "out"])
            self.assertEqual(args.benchmark_root_base, expected)


if __name__ == "__main__":
    unittest.main()
