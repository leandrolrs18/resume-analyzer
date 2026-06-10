import pytest
import urllib.request
import urllib.error
import json
from unittest.mock import patch, MagicMock
from app.services.llm_service import GeminiLlmService


# Garante que a geração síncrona monta a requisição certa e extrai o texto da resposta.
def test_gemini_service_generate_success():
    service = GeminiLlmService(api_key="fake-key", model="gemini-3-flash-preview")

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
        assert request_obj.full_url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key=fake-key"
        payload = json.loads(request_obj.data.decode("utf-8"))
        assert payload["contents"][0]["parts"][0]["text"] == "hello"
        assert payload["generationConfig"]["maxOutputTokens"] == 1000
        assert payload["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0


# Garante que a versão assíncrona usa a mesma lógica de geração da versão síncrona.
@pytest.mark.asyncio
async def test_gemini_service_generate_async():
    service = GeminiLlmService(api_key="fake-key", model="gemini-3-flash-preview")

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


# Garante que a integração falha cedo quando a chave da API não está configurada.
def test_gemini_service_no_api_key():
    service = GeminiLlmService(api_key="")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY não configurada"):
        service._generate_sync("hello", 100)


# Garante que respostas sem o campo esperado do Gemini viram erro explícito.
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


# Garante que o cliente faz retry após 429 e recupera com sucesso na tentativa seguinte.
@patch("time.sleep")
def test_gemini_service_retry_success(mock_sleep):
    service = GeminiLlmService(api_key="fake-key")

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{"text": "Success after retry!"}]
            }
        }]
    }).encode("utf-8")

    # First call raises 429 HTTPError, second call succeeds
    mock_urlopen = MagicMock()
    mock_urlopen.return_value.__enter__.side_effect = [
        urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None),
        mock_response
    ]

    with patch("urllib.request.urlopen", mock_urlopen):
        res = service._generate_sync("hello", 100)
        assert res == "Success after retry!"
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(2.0)  # Initial delay


# Garante que o cliente desiste após as tentativas configuradas quando o 429 persiste.
@patch("time.sleep")
def test_gemini_service_retry_failure(mock_sleep):
    service = GeminiLlmService(api_key="fake-key")

    # All calls raise 429 HTTPError
    mock_urlopen = MagicMock()
    mock_urlopen.return_value.__enter__.side_effect = [
        urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None),
        urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None),
        urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None),
        urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None),
    ]

    with patch("urllib.request.urlopen", mock_urlopen):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            service._generate_sync("hello", 100)
        assert exc_info.value.code == 429
        assert mock_sleep.call_count == 3
        from unittest.mock import call
        mock_sleep.assert_has_calls([
            call(2.0),
            call(4.0),
            call(8.0)
        ], any_order=False)
