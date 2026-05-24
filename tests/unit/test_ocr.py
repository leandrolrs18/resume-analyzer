from types import SimpleNamespace

import pytest
from PIL import Image

from app.core.config import Settings
from app.services.document_service import DocumentService
from app.services.ocr_service import OcrService


class FallbackOcrService:
    def __init__(self):
        self.page_calls = 0

    def render_pdf_page(self, _) -> Image.Image:
        return Image.new("RGB", (100, 40), "white")

    def _ocr_image_sync(self, _: Image.Image) -> str:
        self.page_calls += 1
        return "ocr extracted text from scanned page"


@pytest.mark.asyncio
async def test_pdf_without_native_text_falls_back_to_ocr(monkeypatch) -> None:
    service = DocumentService(FallbackOcrService(), Settings())

    class FakePage:
        def get_text(self, _: str) -> str:
            return ""

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

    text = await service._extract_pdf(b"fake-pdf")

    assert text == "ocr extracted text from scanned page"
    assert service.ocr_service.page_calls == 1


def test_ocr_tries_multiple_tesseract_layout_modes(monkeypatch) -> None:
    calls = []

    def fake_image_to_string(_, lang: str, config: str) -> str:
        calls.append((lang, config))
        if "--psm 11" in config:
            return "Experiência profissional com Python e AWS em currículo escaneado"
        return "???"

    monkeypatch.setattr(
        "app.services.ocr_service.pytesseract",
        SimpleNamespace(image_to_string=fake_image_to_string),
    )

    service = OcrService()
    text = service._ocr_image_sync(Image.new("RGB", (100, 40), "white"))

    assert "currículo escaneado" in text
    assert calls[0] == ("por+eng", "--oem 3 --psm 6")
    assert calls[-1] == ("por+eng", "--oem 3 --psm 11")
