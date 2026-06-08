import asyncio
from io import BytesIO
import logging

import fitz
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import pytesseract
except ImportError:  # pragma: no cover - exercised only in incomplete local installs
    pytesseract = None

from app.core.exceptions import ApplicationError
from app.observability.metrics import OCR_FAILURES_TOTAL

logger = logging.getLogger(__name__)

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
        self._resolved_languages = None

    def _get_resolved_languages(self) -> str:
        if self._resolved_languages is not None:
            return self._resolved_languages

        if pytesseract is None:
            self._resolved_languages = self.languages
            return self._resolved_languages

        try:
            # Tesseract informa quais idiomas estão instalados no ambiente.
            available = pytesseract.get_languages()
        except Exception as e:
            logger.warning(f"Failed to get available Tesseract languages: {e}")
            available = []

        if available:
            requested = [lang.strip() for lang in self.languages.split("+") if lang.strip()]
            supported = [lang for lang in requested if lang in available]
            if supported:
                self._resolved_languages = "+".join(supported)
            else:
                if "eng" in available:
                    self._resolved_languages = "eng"
                else:
                    self._resolved_languages = available[0]
        else:
            self._resolved_languages = self.languages

        logger.info(f"Resolved OCR languages: {self._resolved_languages}")
        return self._resolved_languages

    def _ocr_image_sync(self, image: Image.Image) -> str:
        if pytesseract is None:
            raise ApplicationError(
                "Tesseract OCR dependency is not installed",
                status_code=500,
            )
        resolved_langs = self._get_resolved_languages()
        # Pré-processa imagem com Pillow antes de enviar ao Tesseract.
        prepared = self._prepare_image(image)
        best_text = ""
        for config in TESSERACT_CONFIGS:
            try:
                # Chamada principal do OCR: imagem -> texto.
                text = pytesseract.image_to_string(prepared, lang=resolved_langs, config=config).strip()
            except Exception as e:
                logger.warning(f"OCR with config {config} failed: {e}")
                continue
            if self._is_useful_text(text):
                return text
            if len(text) > len(best_text):
                best_text = text

        if not self._is_useful_text(best_text):
            logger.info("OCR on thresholded image did not yield useful text; trying grayscale fallback")
            grayscale = ImageOps.grayscale(image)
            grayscale = ImageOps.autocontrast(grayscale)
            for config in TESSERACT_CONFIGS:
                try:
                    # Segunda tentativa com imagem em tons de cinza se o threshold falhar.
                    text = pytesseract.image_to_string(grayscale, lang=resolved_langs, config=config).strip()
                except Exception as e:
                    logger.warning(f"OCR fallback with config {config} failed: {e}")
                    continue
                if self._is_useful_text(text):
                    return text
                if len(text) > len(best_text):
                    best_text = text

        return best_text.strip()

    async def extract_from_image_bytes(self, content: bytes) -> str:
        # Imagem enviada diretamente na API: bytes -> Pillow Image -> Tesseract.
        image = Image.open(BytesIO(content)).convert("RGB")
        try:
            return await asyncio.to_thread(self._ocr_image_sync, image)
        except Exception:
            OCR_FAILURES_TOTAL.inc()
            raise



    @staticmethod
    def render_pdf_page(page: fitz.Page) -> Image.Image:
        # Página PDF escaneada: PyMuPDF renderiza a página como PNG em memória.
        pix = page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM), alpha=False)
        # Pillow abre o PNG renderizado e entrega uma imagem RGB para o OCR.
        return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        # Normaliza a imagem para melhorar contraste e leitura do Tesseract.
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
