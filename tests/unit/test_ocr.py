import pytest

from app.core.config import Settings
from app.services.document_service import DocumentService


class FallbackOcrService:
    def __init__(self):
        self.pdf_calls = 0

    async def extract_from_pdf_bytes(self, _: bytes, __: int) -> str:
        self.pdf_calls += 1
        return "ocr extracted text"


@pytest.mark.asyncio
async def test_pdf_without_native_text_falls_back_to_ocr(monkeypatch) -> None:
    service = DocumentService(FallbackOcrService(), Settings())
    monkeypatch.setattr(service, "_extract_pdf_text", lambda _: "")

    text = await service._extract_pdf(b"fake-pdf")

    assert text == "ocr extracted text"
    assert service.ocr_service.pdf_calls == 1
