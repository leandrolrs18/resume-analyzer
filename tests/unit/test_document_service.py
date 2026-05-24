import pytest
from fastapi import UploadFile

from app.core.config import Settings
from app.services.document_service import DocumentService


class StubOcrService:
    def __init__(self):
        self.called = False

    async def extract_from_image_bytes(self, _: bytes) -> str:
        self.called = True
        return "ocr image text"

    async def extract_from_pdf_bytes(self, _: bytes, __: int) -> str:
        self.called = True
        return "ocr pdf text"


@pytest.mark.asyncio
async def test_document_service_skips_ocr_for_native_pdf(monkeypatch) -> None:
    settings = Settings()
    ocr = StubOcrService()
    service = DocumentService(ocr, settings)
    monkeypatch.setattr(service, "_extract_pdf_text", lambda _: "native pdf text")
    upload = UploadFile(filename="maria.pdf", file=object())

    async def fake_read() -> bytes:
        return b"fake-pdf"

    monkeypatch.setattr(upload, "read", fake_read)

    document = await service._extract_single(upload)

    assert document.extracted_text == "native pdf text"
    assert ocr.called is False
