from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.check_external_baselines import check_external_baseline_config


class ExternalBaselinePreflightTests(unittest.TestCase):
    def test_preflight_marks_missing_repos_as_not_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "external.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ai-researcher": {
                            "enabled": True,
                            "execution_mode": "upstream",
                            "repo_path": str(Path(tmp) / "missing-ai-researcher"),
                        },
                        "scipip": {
                            "enabled": True,
                            "repo_path": str(Path(tmp) / "missing-scipip"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = check_external_baseline_config(config_path)

        by_name = {row["baseline"]: row for row in report["baselines"]}
        self.assertFalse(by_name["ai-researcher"]["ready"])
        self.assertFalse(by_name["scipip"]["ready"])
        self.assertIn("missing", " ".join(by_name["ai-researcher"]["issues"]).lower())

    def test_preflight_accepts_minimal_fake_upstream_layouts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ai_repo = root / "AI-Researcher"
            ai_runner = ai_repo / "ai_researcher" / "src"
            ai_runner.mkdir(parents=True)
            for name in ("grounded_idea_gen.py", "experiment_plan_gen.py", "tournament_ranking.py"):
                (ai_runner / name).write_text("print('ok')\n", encoding="utf-8")

            scipip_repo = root / "SciPIP"
            (scipip_repo / "src").mkdir(parents=True)
            (scipip_repo / "src" / "generator.py").write_text("print('ok')\n", encoding="utf-8")
            config_file = scipip_repo / "configs" / "datasets.yaml"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("datasets: []\n", encoding="utf-8")

            config_path = root / "external.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ai-researcher": {
                            "enabled": True,
                            "execution_mode": "upstream",
                            "repo_path": str(ai_repo),
                        },
                        "scipip": {
                            "enabled": True,
                            "repo_path": str(scipip_repo),
                            "config_path": str(config_file),
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = check_external_baseline_config(config_path)

        by_name = {row["baseline"]: row for row in report["baselines"]}
        self.assertTrue(by_name["ai-researcher"]["ready"])
        self.assertEqual(by_name["ai-researcher"]["adapter_status"], "exact-upstream")
        self.assertTrue(by_name["scipip"]["ready"])
        self.assertEqual(by_name["scipip"]["adapter_status"], "paper-faithful-adapter")


if __name__ == "__main__":
    unittest.main()
