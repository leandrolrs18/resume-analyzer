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
        native_text = await asyncio.to_thread(self._extract_pdf_text, content)
        if native_text.strip():
            logger.info("native_pdf_text_detected")
            return native_text
        logger.info("falling_back_to_ocr_for_pdf")
        return await self.ocr_service.extract_from_pdf_bytes(
            content, self.settings.max_pages_per_document
        )

    def _extract_pdf_text(self, content: bytes) -> str:
        doc = fitz.open(stream=content, filetype="pdf")
        try:
            page_count = min(doc.page_count, self.settings.max_pages_per_document)
            pages = [doc.load_page(index).get_text("text") for index in range(page_count)]
            return "\n".join(page.strip() for page in pages if page.strip()).strip()
        except Exception as exc:
            raise ApplicationError(
                "Failed to parse PDF", status_code=400, details={"error": str(exc)}
            ) from exc
        finally:
            doc.close()

    @staticmethod
    def _candidate_name(filename: str) -> str:
        stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
        return stem.title() if stem else "Unknown Candidate"
