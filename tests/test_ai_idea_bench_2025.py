from __future__ import annotations

import io
import sys
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
from zipfile import ZipFile

from pypdf import PdfWriter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.benchmarks.ai_idea_bench_2025 import (  # noqa: E402
    AIIdeaBench2025Record,
    ai_idea_bench_2025_instance_from_record,
)


def _blank_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class AIIdeaBench2025PaperAccessTests(unittest.TestCase):
    def _make_temp_root(self) -> Path:
        temp_root_base = ROOT / "tests" / "_tmp_ai_idea_bench_2025"
        temp_root_base.mkdir(parents=True, exist_ok=True)
        path = temp_root_base / f"case-{uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_instance_from_record_uses_nested_extracted_papers_dir(self) -> None:
        root = self._make_temp_root()
        try:
            papers_dir = root / "Idea_bench_data" / "papers_data"
            papers_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = papers_dir / "sample reference paper.pdf"
            pdf_path.write_bytes(_blank_pdf_bytes())

            record = AIIdeaBench2025Record(
                benchmark_index=0,
                topic="topic",
                revised_topic="topic",
                target_paper="",
                target_paper_path="",
                motivation="",
                method_summary="",
                reference_titles=["Sample Reference Paper"],
                reference_local_paths=["./papers_data/sample reference paper.pdf"],
                raw_record={},
            )
            instance = ai_idea_bench_2025_instance_from_record(record, benchmark_root=root)

            self.assertEqual(instance.metadata["reference_local_paths"][0], str(pdf_path.resolve()))
            snippets = instance.metadata["paper_grounding"]["reference_paper_snippets"]
            self.assertEqual(snippets[0]["source_kind"], "file")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_instance_from_record_can_read_papers_from_archive(self) -> None:
        root = self._make_temp_root()
        try:
            archive_path = root / "Idea_bench_data.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "Idea_bench_data/papers_data/sample archive paper.pdf",
                    _blank_pdf_bytes(),
                )

            record = AIIdeaBench2025Record(
                benchmark_index=1,
                topic="topic",
                revised_topic="topic",
                target_paper="",
                target_paper_path="",
                motivation="",
                method_summary="",
                reference_titles=["Sample Archive Paper"],
                reference_local_paths=["./papers_data/sample archive paper.pdf"],
                raw_record={},
            )
            instance = ai_idea_bench_2025_instance_from_record(record, benchmark_root=root)

            snippets = instance.metadata["paper_grounding"]["reference_paper_snippets"]
            self.assertEqual(snippets[0]["source_kind"], "zip_entry")
            self.assertIn("sample archive paper.pdf", snippets[0]["entry_name"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
