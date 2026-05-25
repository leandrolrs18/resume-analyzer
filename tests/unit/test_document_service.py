import pytest

from app.core.config import Settings
from app.services.document_service import DocumentService


class StubOcrService:
    def render_pdf_page(self, _):
        return object()

    def _ocr_image_sync(self, _) -> str:
        return "ocr page text"


@pytest.mark.asyncio
async def test_pdf_without_native_text_falls_back_to_ocr(monkeypatch) -> None:
    service = DocumentService(StubOcrService(), Settings())

    class FakePage:
        def get_text(self, _: str) -> str:
            return ""

    class FakeDoc:
        is_encrypted = False
        needs_pass = False
        page_count = 1

        def embfile_count(self) -> int:
            return 0

        def load_page(self, _):
            return FakePage()

        def close(self):
            return None

    monkeypatch.setattr("app.services.document_service.fitz.open", lambda **_: FakeDoc())

    assert await service._extract_pdf(b"fake") == "ocr page text"
