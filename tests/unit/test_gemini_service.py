import pytest
import urllib.request
import urllib.error
import json
from unittest.mock import patch, MagicMock
from app.services.llm_service import GeminiLlmService


def test_gemini_service_generate_success():
    service = GeminiLlmService(api_key="fake-key", model="gemini-2.5-flash")

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{"text": "Hello, this is a response from Gemini."}]
            }
        }]
    }).encode("utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = service._generate_sync("hello", 100)
        assert res == "Hello, this is a response from Gemini."

        # Check payload
        args, kwargs = mock_urlopen.call_args
        request_obj = args[0]
        assert request_obj.full_url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=fake-key"
        payload = json.loads(request_obj.data.decode("utf-8"))
        assert payload["contents"][0]["parts"][0]["text"] == "hello"
        assert payload["generationConfig"]["maxOutputTokens"] == 1000
        assert payload["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0


@pytest.mark.asyncio
async def test_gemini_service_generate_async():
    service = GeminiLlmService(api_key="fake-key", model="gemini-2.5-flash")

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{"text": "Hello, this is a response from Gemini."}]
            }
        }]
    }).encode("utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = await service.generate("hello", 100)
        assert res == "Hello, this is a response from Gemini."


def test_gemini_service_no_api_key():
    service = GeminiLlmService(api_key="")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY não configurada"):
        service._generate_sync("hello", 100)


def test_gemini_service_invalid_response():
    service = GeminiLlmService(api_key="fake-key")

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "invalid_key": "some_value"
    }).encode("utf-8")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        with pytest.raises(ValueError, match="Gemini response did not include generated text"):
            service._generate_sync("hello", 100)
