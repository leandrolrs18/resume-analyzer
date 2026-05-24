import asyncio
import logging
import os
from llama_cpp import Llama
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
                "LLM_MODEL_PATH", "/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
            )
        
        self._model = None

    def _load_model(self) -> Llama:
        """
        Garante o carregamento único (Lazy Loading) do modelo GGUF na memória 
        usando a biblioteca estável llama-cpp-python.
        """
        if self._model is None:
            logger.info(f"Carregando modelo Qwen GGUF centralizado a partir de: {self.model_path}")
            try:
                self._model = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,       # Janela de contexto segura para os chunks de currículos
                    n_threads=4,      # Evita travar os núcleos da CPU do container
                    verbose=False     # Desliga os logs poluídos de C++ no terminal
                )
            except Exception:
                LLM_FAILURES_TOTAL.inc()
                logger.exception("erro_critico_ao_carregar_arquivo_gguf")
                raise
        return self._model

    def _generate_sync(self, prompt: str, max_new_tokens: int) -> str:
        """
        Executa a inferência síncrona diretamente na CPU usando o formato ChatML (Qwen2.5).
        """
        model = self._load_model()
        
        # Aplica o Chat Template do Qwen2.5 para garantir que ele entenda o papel do sistema
        prompt_final = (
            f"<|im_start|>system\nVocê é um assistente de recrutamento técnico estrito e preciso.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        response = model(
            prompt_final,
            max_tokens=max_new_tokens,
            temperature=0.1,  # Temperatura baixa para evitar qualquer tipo de alucinação
            stop=["<|im_end|>", "<|im_start|>"]
        )
        return str(response["choices"][0]["text"]).strip()

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        """
        Roda a inferência pesada de CPU em uma thread separada para não bloquear 
        o loop de eventos assíncronos do FastAPI.
        """
        try:
            return await asyncio.to_thread(self._generate_sync, prompt, max_new_tokens)
        except Exception:
            LLM_FAILURES_TOTAL.inc()
            logger.exception("llm_generation_failed")
            raise

    def __call__(self, prompt: str, max_tokens: int, temperature: float = 0.1, stop: list[str] = None) -> dict:
        """
        MÁGICA DE COMPATIBILIDADE: Permite que o LlmService seja chamado diretamente 
        como uma função (ex: self.llm_service(...)), resolvendo o erro de 
        'TypeError: LlmService object is not callable' que dava no SummarizationService.
        """
        model = self._load_model()
        raw_response = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["<|im_end|>"]
        )
        return raw_response