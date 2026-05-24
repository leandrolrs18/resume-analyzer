import pytest
from fastapi import UploadFile
from PIL import Image

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

    def render_pdf_page(self, _) -> Image.Image:
        self.called = True
        return Image.new("RGB", (100, 40), "white")

    def _ocr_image_sync(self, _: Image.Image) -> str:
        self.called = True
        return "ocr page text"


@pytest.mark.asyncio
async def test_document_service_skips_ocr_for_native_pdf(monkeypatch) -> None:
    settings = Settings()
    ocr = StubOcrService()
    service = DocumentService(ocr, settings)

    class FakePage:
        def get_text(self, _: str) -> str:
            return "native pdf text with enough alphabetic content"

    class FakeDoc:
        is_encrypted = False
        needs_pass = False
        page_count = 1

        def embfile_count(self) -> int:
            return 0

        def load_page(self, _: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.services.document_service.fitz.open", lambda **_: FakeDoc())
    upload = UploadFile(filename="maria.pdf", file=object())

    async def fake_read() -> bytes:
        return b"fake-pdf"

    monkeypatch.setattr(upload, "read", fake_read)

    document = await service._extract_single(upload)

    assert document.extracted_text == "native pdf text with enough alphabetic content"
    assert ocr.called is False


@pytest.mark.asyncio
async def test_document_service_uses_hybrid_pdf_extraction(monkeypatch) -> None:
    settings = Settings()
    ocr = StubOcrService()
    service = DocumentService(ocr, settings)

    class FakePage:
        def __init__(self, text: str):
            self.text = text

        def get_text(self, _: str) -> str:
            return self.text

    class FakeDoc:
        is_encrypted = False
        needs_pass = False
        page_count = 2

        def embfile_count(self) -> int:
            return 0

        def load_page(self, index: int) -> FakePage:
            if index == 0:
                return FakePage("Formação em Engenharia de Software pela Universidade Federal")
            return FakePage("")

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.services.document_service.fitz.open", lambda **_: FakeDoc())

    text = await service._extract_pdf(b"fake-pdf")

    assert "Formação em Engenharia de Software" in text
    assert "ocr page text" in text
    assert ocr.called is True


@pytest.mark.asyncio
async def test_document_service_rejects_pdf_with_embedded_files(monkeypatch) -> None:
    service = DocumentService(StubOcrService(), Settings())

    class FakeDoc:
        is_encrypted = False
        needs_pass = False
        page_count = 1

        def embfile_count(self) -> int:
            return 1

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.services.document_service.fitz.open", lambda **_: FakeDoc())

    with pytest.raises(Exception, match="arquivos embutidos"):
        await service._extract_pdf(b"fake-pdf")
