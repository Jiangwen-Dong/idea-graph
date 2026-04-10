from __future__ import annotations

import sys
from pathlib import Path
import shutil
from tempfile import mkdtemp
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import _windows_safe_path, read_text_file, write_text_file


class FsUtilsTests(unittest.TestCase):
    def test_write_and_read_text_file_on_deep_nested_path(self) -> None:
        tmp_dir = mkdtemp()
        try:
            base = Path(tmp_dir)
            nested = base
            while len(str(nested / "benchmark_native_evaluation.json")) < 252:
                nested = nested / ("deep_segment_" + str(len(str(nested))))
            target = nested / "benchmark_native_evaluation.json"

            write_text_file(target, '{"ok": true}\n')

            self.assertEqual(read_text_file(target), '{"ok": true}\n')
        finally:
            shutil.rmtree(_windows_safe_path(tmp_dir), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
