import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from app.observability.metrics import LLM_FAILURES_TOTAL

logger = logging.getLogger(__name__)


class LlmService:
    def __init__(self, model_name: str | None = None):
        # CORREÇÃO CRÍTICA: Se a string recebida não for um arquivo .gguf real,
        # nós forçamos o uso do caminho correto do volume/pasta onde baixamos o Qwen
        if model_name and model_name.endswith(".gguf") and os.path.exists(model_name):
            self.model_path = model_name
        else:
            # Fallback seguro para o caminho local onde o Qwen2.5 está montado no Docker
            self.model_path = os.getenv(
                "LLM_MODEL_PATH", "/home/user/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
            )

        self._model = None
        self._generation_lock = asyncio.Lock()

    def _load_model(self) -> Any:
        """
        Garante o carregamento único (Lazy Loading) do modelo GGUF na memória
        usando a biblioteca estável llama-cpp-python.
        """
        if self._model is None:
            started_at = time.perf_counter()
            logger.info(f"Carregando modelo Qwen GGUF centralizado a partir de: {self.model_path}")
            try:
                from llama_cpp import Llama

                self._model = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,  # Janela de contexto segura para os chunks de currículos
                    n_threads=4,  # Evita travar os núcleos da CPU do container
                    verbose=False,  # Desliga os logs poluídos de C++ no terminal
                )
            except Exception:
                LLM_FAILURES_TOTAL.inc()
                logger.exception("erro_critico_ao_carregar_arquivo_gguf")
                raise
            logger.info(
                "llm_model_loaded",
                extra={"latency_ms": round((time.perf_counter() - started_at) * 1000, 2)},
            )
        return self._model

    def _generate_sync(self, prompt: str, max_new_tokens: int) -> str:
        """
        Executa a inferência síncrona diretamente na CPU usando o formato ChatML (Qwen2.5).
        """
        model = self._load_model()
        started_at = time.perf_counter()

        # Aplica o Chat Template do Qwen2.5 para garantir que ele entenda o papel do sistema
        prompt_final = (
            "<|im_start|>system\n"
            "Você é um assistente de recrutamento técnico estrito e preciso."
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        response = model(
            prompt_final,
            max_tokens=max_new_tokens,
            temperature=0.1,  # Temperatura baixa para evitar qualquer tipo de alucinação
            stop=["<|im_end|>", "<|im_start|>"],
        )
        logger.info(
            "llm_generation_completed",
            extra={
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "max_new_tokens": max_new_tokens,
                "prompt_chars": len(prompt_final),
            },
        )
        return str(response["choices"][0]["text"]).strip()

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        """
        Roda a inferência pesada de CPU em uma thread separada para não bloquear
        o loop de eventos assíncronos do FastAPI.
        """
        try:
            self._load_model()
            async with self._generation_lock:
                return await asyncio.to_thread(self._generate_sync, prompt, max_new_tokens)
        except Exception:
            LLM_FAILURES_TOTAL.inc()
            logger.exception("llm_generation_failed")
            raise

class GeminiLlmService:
    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview"):
        self.api_key = api_key
        self.model = model

    def _generate_sync(self, prompt: str, max_new_tokens: int) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY não configurada")

        started_at = time.perf_counter()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {
                "parts": [{"text": "Você é um assistente de recrutamento técnico estrito e preciso."}]
            },
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max(max_new_tokens, 1000),
                "thinkingConfig": {
                    "thinkingBudget": 0
                }
            }
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        max_retries = 3
        backoff_factor = 2.0
        initial_delay = 2.0
        data = None

        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    delay = initial_delay * (backoff_factor ** attempt)
                    logger.warning(
                        f"Gemini API request failed with status {e.code}. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    continue
                logger.error(f"Gemini API request failed (status {e.code}): {e}")
                raise
            except Exception as e:
                if attempt < max_retries:
                    delay = initial_delay * (backoff_factor ** attempt)
                    logger.warning(
                        f"Gemini API request failed with error: {e}. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    continue
                logger.error(f"Gemini API request failed: {e}")
                raise

        if data is None:
            raise RuntimeError("Gemini API request failed to return data")

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Gemini response did not include generated text: {e}")

        logger.info(
            "gemini_generation_completed",
            extra={
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "max_new_tokens": max_new_tokens,
                "model": self.model,
            },
        )
        return str(text).strip()

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        try:
            return await asyncio.to_thread(self._generate_sync, prompt, max_new_tokens)
        except Exception:
            LLM_FAILURES_TOTAL.inc()
            logger.exception("gemini_generation_failed")
            raise

