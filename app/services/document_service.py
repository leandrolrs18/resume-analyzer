import asyncio
import logging
from pathlib import Path

import fitz
from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import ApplicationError
from app.schemas import ResumeDocument
from app.services.ocr_service import OcrService

logger = logging.getLogger(__name__)

MIN_NATIVE_PAGE_CHARS = 40
ALLOWED_PDF_EMBEDDED_FILES = 0


class DocumentService:
    def __init__(self, ocr_service: OcrService, settings: Settings):
        self.ocr_service = ocr_service
        self.settings = settings

    async def extract_documents(self, files: list[UploadFile]) -> list[ResumeDocument]:
        return await asyncio.gather(*(self._extract_single(upload) for upload in files))

    async def _extract_single(self, upload: UploadFile) -> ResumeDocument:
        content = await upload.read()
        if len(content) > self.settings.max_upload_size_bytes:
            raise ApplicationError(
                "Uploaded file exceeds maximum allowed size",
                status_code=413,
                details={"filename": upload.filename},
            )
        candidate = self._candidate_name(upload.filename or "unknown")
        if (upload.filename or "").lower().endswith(".pdf"):
            extracted_text = await self._extract_pdf(content)
        else:
            extracted_text = await self.ocr_service.extract_from_image_bytes(content)
        if not extracted_text.strip():
            raise ApplicationError(
                "No text could be extracted from file",
                status_code=422,
                details={"filename": upload.filename},
            )
        return ResumeDocument(
            candidate=candidate,
            source_filename=upload.filename or candidate,
            extracted_text=extracted_text.strip(),
        )

    async def _extract_pdf(self, content: bytes) -> str:
        try:
            return await self._extract_pdf_hybrid(content)
        except ApplicationError:
            raise
        except Exception as exc:
            raise ApplicationError(
                "Failed to parse PDF", status_code=400, details={"error": str(exc)}
            ) from exc

    async def _extract_pdf_hybrid(self, content: bytes) -> str:
        doc = fitz.open(stream=content, filetype="pdf")
        try:
            self._validate_pdf(doc)
            page_count = min(doc.page_count, self.settings.max_pages_per_document)
            page_texts: list[str | None] = []
            ocr_tasks = []
            ocr_positions = []

            for page_index in range(page_count):
                page = doc.load_page(page_index)
                native_text = page.get_text("text").strip()
                if self._is_useful_native_text(native_text):
                    page_texts.append(native_text)
                    continue

                logger.info("falling_back_to_ocr_for_pdf_page", extra={"page": page_index + 1})
                image = self.ocr_service.render_pdf_page(page)
                ocr_positions.append(len(page_texts))
                page_texts.append(None)
                ocr_tasks.append(asyncio.to_thread(self.ocr_service._ocr_image_sync, image))

            if ocr_tasks:
                try:
                    ocr_pages = await asyncio.gather(*ocr_tasks)
                except Exception:
                    from app.observability.metrics import OCR_FAILURES_TOTAL

                    OCR_FAILURES_TOTAL.inc()
                    raise
                for position, text in zip(ocr_positions, ocr_pages, strict=False):
                    page_texts[position] = text

            if not ocr_tasks:
                logger.info("native_pdf_text_detected")
            elif any(text for text in page_texts):
                logger.info("hybrid_pdf_text_detected")

            return "\n".join(text.strip() for text in page_texts if text and text.strip()).strip()
        finally:
            doc.close()

    @staticmethod
    def _validate_pdf(doc: fitz.Document) -> None:
        if doc.is_encrypted:
            raise ApplicationError("PDFs criptografados não são suportados", status_code=415)
        embedded_files = getattr(doc, "embfile_count", lambda: 0)()
        if embedded_files > ALLOWED_PDF_EMBEDDED_FILES:
            raise ApplicationError(
                "PDFs com arquivos embutidos não são suportados",
                status_code=415,
                details={"embedded_files": embedded_files},
            )
        if doc.needs_pass:
            raise ApplicationError("PDFs protegidos por senha não são suportados", status_code=415)

    @staticmethod
    def _is_useful_native_text(text: str) -> bool:
        if len(text.strip()) < MIN_NATIVE_PAGE_CHARS:
            return False
        alpha_count = sum(character.isalpha() for character in text)
        visible_count = sum(not character.isspace() for character in text)
        if visible_count == 0:
            return False
        return (alpha_count / visible_count) >= 0.45

    @staticmethod
    def _candidate_name(filename: str) -> str:
        stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
        return stem.title() if stem else "Unknown Candidate"
