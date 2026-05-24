import asyncio
from io import BytesIO

import fitz
from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover - exercised only in incomplete local installs
    pytesseract = None

from app.core.exceptions import ApplicationError
from app.observability.metrics import OCR_FAILURES_TOTAL


class OcrService:
    def __init__(self, languages: str = "por+eng") -> None:
        self.languages = languages

    def _ocr_image_sync(self, image: Image.Image) -> str:
        if pytesseract is None:
            raise ApplicationError(
                "Tesseract OCR dependency is not installed",
                status_code=500,
            )
        return pytesseract.image_to_string(image, lang=self.languages, config="--psm 6").strip()

    async def extract_from_image_bytes(self, content: bytes) -> str:
        image = Image.open(BytesIO(content)).convert("RGB")
        try:
            return await asyncio.to_thread(self._ocr_image_sync, image)
        except Exception:
            OCR_FAILURES_TOTAL.inc()
            raise

    async def extract_from_pdf_bytes(self, content: bytes, max_pages: int) -> str:
        doc = fitz.open(stream=content, filetype="pdf")
        try:
            page_count = min(doc.page_count, max_pages)
            tasks = []
            for page_index in range(page_count):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
                tasks.append(asyncio.to_thread(self._ocr_image_sync, image))
            pages = await asyncio.gather(*tasks)
            return "\n".join(filter(None, pages)).strip()
        except Exception:
            OCR_FAILURES_TOTAL.inc()
            raise
        finally:
            doc.close()
