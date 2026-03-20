from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--input",
        str(ROOT / "data" / "sample_instance.json"),
        "--output-dir",
        str(ROOT / "outputs"),
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
