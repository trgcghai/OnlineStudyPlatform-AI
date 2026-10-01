import os
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docx import Document
from pptx import Presentation
from pypdf import PdfReader

from app.core.config import AiSettings


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx"}
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000


class KnowledgeDocumentError(RuntimeError):
    pass


class SafeDocumentRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str]) -> None:
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_document_url(newurl, self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True)
class DocumentPart:
    locator: str | None
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    locator: str | None
    text: str
    chunk_index: int


def normalize_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def validate_document_url(url: str, allowed_hosts: set[str]) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise KnowledgeDocumentError("Document URL scheme is not allowed")
    if not parsed.hostname or parsed.hostname.casefold() not in allowed_hosts:
        raise KnowledgeDocumentError("Document URL host is not allowed")
    extension = Path(urllib.parse.unquote(parsed.path)).suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        raise KnowledgeDocumentError("Document format is not supported")
    return extension


def download_document(url: str, settings: AiSettings) -> str:
    extension = validate_document_url(url, settings.knowledge_allowed_host_set)
    opener = urllib.request.build_opener(
        SafeDocumentRedirectHandler(settings.knowledge_allowed_host_set)
    )
    descriptor, temporary_path = tempfile.mkstemp(suffix=extension)
    os.close(descriptor)
    total = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "kltn-ai/1.0"})
        with opener.open(
            request, timeout=settings.knowledge_download_timeout_seconds
        ) as response:
            validate_document_url(
                response.geturl(), settings.knowledge_allowed_host_set
            )
            with open(temporary_path, "wb") as destination:
                while True:
                    block = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not block:
                        break
                    total += len(block)
                    if total > settings.knowledge_max_bytes:
                        raise KnowledgeDocumentError("Document exceeds the size limit")
                    destination.write(block)
        if total == 0:
            raise KnowledgeDocumentError("Document is empty")
        return temporary_path
    except Exception:
        Path(temporary_path).unlink(missing_ok=True)
        raise


def validate_office_archive(path: str, settings: AiSettings) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise KnowledgeDocumentError("Office archive has too many entries")
            total = sum(member.file_size for member in members)
            if total > settings.knowledge_max_archive_bytes:
                raise KnowledgeDocumentError("Office archive expands beyond the limit")
            if any(member.flag_bits & 0x1 for member in members):
                raise KnowledgeDocumentError(
                    "Encrypted Office documents are not supported"
                )
    except zipfile.BadZipFile as error:
        raise KnowledgeDocumentError("Office document is corrupt") from error


def extract_pdf(path: str) -> list[DocumentPart]:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise KnowledgeDocumentError("Encrypted PDF documents are not supported")
        parts: list[DocumentPart] = []
        for index, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                parts.append(DocumentPart(locator=f"Trang {index}", text=text))
        return parts
    except KnowledgeDocumentError:
        raise
    except Exception as error:
        raise KnowledgeDocumentError("PDF document could not be read") from error


def extract_docx(path: str, settings: AiSettings) -> list[DocumentPart]:
    validate_office_archive(path, settings)
    try:
        document = Document(path)
        values = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            values.extend(
                " | ".join(normalize_text(cell.text) for cell in row.cells)
                for row in table.rows
            )
        text = normalize_text("\n".join(values))
        return [DocumentPart(locator=None, text=text)] if text else []
    except Exception as error:
        raise KnowledgeDocumentError("DOCX document could not be read") from error


def extract_pptx(path: str, settings: AiSettings) -> list[DocumentPart]:
    validate_office_archive(path, settings)
    try:
        presentation = Presentation(path)
        parts: list[DocumentPart] = []
        for index, slide in enumerate(presentation.slides, start=1):
            values = [
                shape.text
                for shape in slide.shapes
                if hasattr(shape, "text") and shape.text
            ]
            try:
                notes = slide.notes_slide.notes_text_frame.text
                if notes:
                    values.append(notes)
            except (AttributeError, ValueError):
                pass
            text = normalize_text("\n".join(values))
            if text:
                parts.append(DocumentPart(locator=f"Slide {index}", text=text))
        return parts
    except Exception as error:
        raise KnowledgeDocumentError("PPTX document could not be read") from error


def extract_document(path: str, settings: AiSettings) -> list[DocumentPart]:
    extension = Path(path).suffix.casefold()
    if extension == ".pdf":
        parts = extract_pdf(path)
    elif extension == ".docx":
        parts = extract_docx(path, settings)
    elif extension == ".pptx":
        parts = extract_pptx(path, settings)
    else:
        raise KnowledgeDocumentError("Document format is not supported")

    character_count = sum(len(part.text) for part in parts)
    if character_count == 0:
        raise KnowledgeDocumentError("Document has no extractable text")
    if character_count > settings.knowledge_max_characters:
        raise KnowledgeDocumentError("Extracted document text exceeds the limit")
    return parts


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise KnowledgeDocumentError(
            "Chunk overlap must be smaller than the chunk size"
        )
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        current_length = 0
        while end < len(words):
            added = len(words[end]) + (1 if current_length else 0)
            if current_length + added > size and end > start:
                break
            current_length += added
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        overlap_length = 0
        next_start = end
        while next_start > start and overlap_length < overlap:
            next_start -= 1
            overlap_length += len(words[next_start]) + 1
        start = next_start if next_start < end else end
    return chunks


def chunk_parts(parts: list[DocumentPart], settings: AiSettings) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for part in parts:
        for text in split_text(
            part.text,
            settings.knowledge_chunk_size,
            settings.knowledge_chunk_overlap,
        ):
            chunks.append(
                DocumentChunk(
                    locator=part.locator,
                    text=text,
                    chunk_index=len(chunks),
                )
            )
    if not chunks:
        raise KnowledgeDocumentError("Document has no extractable text")
    return chunks


def extract_and_chunk_url(url: str, settings: AiSettings) -> list[DocumentChunk]:
    path = download_document(url, settings)
    try:
        return chunk_parts(extract_document(path, settings), settings)
    finally:
        Path(path).unlink(missing_ok=True)


def chunk_transcript(transcript: str, settings: AiSettings) -> list[DocumentChunk]:
    text = normalize_text(transcript)
    if not text:
        raise KnowledgeDocumentError("Transcript is empty")
    if len(text) > settings.knowledge_max_characters:
        raise KnowledgeDocumentError("Transcript exceeds the text limit")
    return chunk_parts([DocumentPart(locator="Bản ghi video", text=text)], settings)
