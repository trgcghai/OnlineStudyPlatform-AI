import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from docx import Document
from fastapi.testclient import TestClient
from pptx import Presentation

from app.core.config import AiSettings, get_ai_settings
from app.main import app
from app.services.knowledge_documents import (
    KnowledgeDocumentError,
    SafeDocumentRedirectHandler,
    chunk_transcript,
    extract_docx,
    extract_pdf,
    extract_pptx,
    split_text,
    validate_document_url,
)


class KnowledgeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_ai_settings] = lambda: AiSettings(
            AI_TOKEN=None
        )

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_indexes_transcript_and_returns_chunk_count(self) -> None:
        with patch(
            "app.api.knowledge.index_chunks",
            new_callable=AsyncMock,
            return_value=2,
        ) as index:
            response = TestClient(app).post(
                "/knowledge/index",
                json={
                    "sourceId": "source-1",
                    "courseId": "course-1",
                    "lessonId": "lesson-1",
                    "lessonTitle": "Bài học",
                    "sourceType": "transcript",
                    "contentHash": "a" * 64,
                    "transcript": "Nội dung bài giảng đầy đủ.",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chunkCount"], 2)
        index.assert_awaited_once()

    def test_material_requires_source_url(self) -> None:
        response = TestClient(app).post(
            "/knowledge/index",
            json={
                "sourceId": "source-1",
                "courseId": "course-1",
                "lessonId": "lesson-1",
                "lessonTitle": "Bài học",
                "sourceType": "material",
                "contentHash": "a" * 64,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_delete_source_calls_vector_store(self) -> None:
        with patch(
            "app.api.knowledge.delete_source", new_callable=AsyncMock
        ) as delete:
            response = TestClient(app).delete("/knowledge/sources/source-1")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["deleted"])
        delete.assert_awaited_once()


class KnowledgeDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AiSettings(
            KNOWLEDGE_CHUNK_SIZE=300,
            KNOWLEDGE_CHUNK_OVERLAP=40,
        )

    def test_rejects_legacy_and_unapproved_document_urls(self) -> None:
        with self.assertRaisesRegex(KnowledgeDocumentError, "not supported"):
            validate_document_url("http://minio/lesson.ppt", {"minio"})
        with self.assertRaisesRegex(KnowledgeDocumentError, "not allowed"):
            validate_document_url("https://example.com/lesson.pdf", {"minio"})

    def test_rejects_redirect_before_following_an_unapproved_host(self) -> None:
        handler = SafeDocumentRedirectHandler({"minio"})

        with self.assertRaisesRegex(KnowledgeDocumentError, "not allowed"):
            handler.redirect_request(
                None,
                None,
                302,
                "Found",
                {},
                "https://example.com/private.pdf",
            )

    def test_extracts_pdf_by_page(self) -> None:
        reader = SimpleNamespace(
            is_encrypted=False,
            pages=[
                SimpleNamespace(extract_text=lambda: "Trang một"),
                SimpleNamespace(extract_text=lambda: "Trang hai"),
            ],
        )
        with patch(
            "app.services.knowledge_documents.PdfReader", return_value=reader
        ):
            parts = extract_pdf("lesson.pdf")

        self.assertEqual([part.locator for part in parts], ["Trang 1", "Trang 2"])
        self.assertEqual(parts[1].text, "Trang hai")

    def test_extracts_docx_paragraphs_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lesson.docx"
            document = Document()
            document.add_paragraph("Khái niệm dependency injection")
            table = document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "Provider"
            table.cell(0, 1).text = "Service"
            document.save(path)

            parts = extract_docx(str(path), self.settings)

        self.assertIn("dependency injection", parts[0].text)
        self.assertIn("Provider | Service", parts[0].text)

    def test_extracts_pptx_by_slide(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lesson.pptx"
            presentation = Presentation()
            slide = presentation.slides.add_slide(presentation.slide_layouts[1])
            slide.shapes.title.text = "Dependency Injection"
            slide.placeholders[1].text = "Provider và service"
            presentation.save(path)

            parts = extract_pptx(str(path), self.settings)

        self.assertEqual(parts[0].locator, "Slide 1")
        self.assertIn("Provider và service", parts[0].text)

    def test_chunks_long_transcript_with_overlap(self) -> None:
        chunks = chunk_transcript("nội dung bài giảng " * 100, self.settings)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].locator, "Bản ghi video")
        self.assertEqual(
            chunks,
            sorted(chunks, key=lambda chunk: chunk.chunk_index),
        )

    def test_rejects_overlap_that_cannot_advance_the_chunk_window(self) -> None:
        with self.assertRaisesRegex(KnowledgeDocumentError, "smaller"):
            split_text("mot hai ba bon", size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
