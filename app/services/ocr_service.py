import asyncio
from io import BytesIO

import fitz
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import pytesseract
except ImportError:  # pragma: no cover - exercised only in incomplete local installs
    pytesseract = None

from app.core.exceptions import ApplicationError
from app.observability.metrics import OCR_FAILURES_TOTAL

MIN_USEFUL_OCR_CHARS = 40
MIN_ALPHA_RATIO = 0.45
PDF_RENDER_ZOOM = 2.0
TESSERACT_CONFIGS = (
    "--oem 3 --psm 6",
    "--oem 3 --psm 4",
    "--oem 3 --psm 11",
)


class OcrService:
    def __init__(self, languages: str = "por+eng") -> None:
        self.languages = languages

    def _ocr_image_sync(self, image: Image.Image) -> str:
        if pytesseract is None:
            raise ApplicationError(
                "Tesseract OCR dependency is not installed",
                status_code=500,
            )
        prepared = self._prepare_image(image)
        best_text = ""
        for config in TESSERACT_CONFIGS:
            text = pytesseract.image_to_string(prepared, lang=self.languages, config=config).strip()
            if self._is_useful_text(text):
                return text
            if len(text) > len(best_text):
                best_text = text
        return best_text.strip()

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
                image = self.render_pdf_page(doc.load_page(page_index))
                tasks.append(asyncio.to_thread(self._ocr_image_sync, image))
            pages = await asyncio.gather(*tasks)
            return "\n".join(filter(None, pages)).strip()
        except Exception:
            OCR_FAILURES_TOTAL.inc()
            raise
        finally:
            doc.close()

    @staticmethod
    def render_pdf_page(page: fitz.Page) -> Image.Image:
        pix = page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM), alpha=False)
        return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        grayscale = ImageOps.grayscale(image)
        grayscale = ImageOps.autocontrast(grayscale)
        grayscale = ImageEnhance.Contrast(grayscale).enhance(1.8)
        grayscale = grayscale.filter(ImageFilter.SHARPEN)
        threshold = grayscale.point(lambda pixel: 255 if pixel > 170 else 0)
        return threshold

    @staticmethod
    def _is_useful_text(text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < MIN_USEFUL_OCR_CHARS:
            return False
        alpha_count = sum(character.isalpha() for character in stripped)
        visible_count = sum(not character.isspace() for character in stripped)
        if visible_count == 0:
            return False
        return (alpha_count / visible_count) >= MIN_ALPHA_RATIO
