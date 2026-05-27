import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from app.services.ocr_service import OcrService
from app.core.exceptions import ApplicationError


def test_ocr_service_resolves_languages_successfully():
    with patch("app.services.ocr_service.pytesseract") as mock_tesseract:
        mock_tesseract.get_languages.return_value = ["eng", "por", "osd"]
        service = OcrService(languages="por+eng")
        assert service._get_resolved_languages() == "por+eng"


def test_ocr_service_filters_unsupported_languages():
    with patch("app.services.ocr_service.pytesseract") as mock_tesseract:
        mock_tesseract.get_languages.return_value = ["eng", "osd"]
        service = OcrService(languages="por+eng")
        assert service._get_resolved_languages() == "eng"


def test_ocr_service_fallback_when_none_supported():
    with patch("app.services.ocr_service.pytesseract") as mock_tesseract:
        mock_tesseract.get_languages.return_value = ["fra", "deu"]
        service = OcrService(languages="por+eng")
        # should fall back to first available if 'eng' is not there
        assert service._get_resolved_languages() == "fra"


def test_ocr_service_tesseract_not_installed():
    with patch("app.services.ocr_service.pytesseract", None):
        service = OcrService(languages="por+eng")
        with pytest.raises(ApplicationError, match="Tesseract OCR dependency is not installed"):
            service._ocr_image_sync(Image.new("RGB", (10, 10)))


def test_ocr_service_grayscale_fallback_on_empty_text():
    with patch("app.services.ocr_service.pytesseract") as mock_tesseract:
        service = OcrService(languages="eng")

        # Make the first attempts return empty string (not useful text)
        # And the fallback return a useful text
        mock_tesseract.image_to_string.side_effect = [
            "",
            "",
            "",  # For the prepared image (3 configs)
            "This is a fallback extracted text that meets the length requirement.",  # For the first config of grayscale fallback
        ]

        image = Image.new("RGB", (100, 100))
        extracted = service._ocr_image_sync(image)

        assert extracted == "This is a fallback extracted text that meets the length requirement."
        # Verify image_to_string was called 4 times (3 prepared, 1 grayscale)
        assert mock_tesseract.image_to_string.call_count == 4
