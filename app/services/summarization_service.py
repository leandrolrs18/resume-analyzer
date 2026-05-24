import logging
from typing import Any

from app.schemas import ResumeDocument

logger = logging.getLogger(__name__)

# Mantemos apenas os limites físicos de caracteres para não estourar o contexto do modelo
MAX_SUMMARY_SOURCE_CHARS = 4000
MAX_JUSTIFICATION_SOURCE_CHARS = 1200


class SummarizationService:
    def __init__(self, llm_service: Any | None, max_new_tokens: int):
        self.llm_service = llm_service
        self.max_new_tokens = max_new_tokens

    async def summarize(self, document: ResumeDocument) -> str:
        """
        Gera o sumário curto do currículo utilizando única e exclusivamente o LLM.
        """
        if self.llm_service is None:
            return "Serviço de LLM indisponível para gerar o sumário."

        # Corta o texto bruto do currículo para caber na janela do Qwen com segurança
        context = document.extracted_text[:MAX_SUMMARY_SOURCE_CHARS]

        prompt = (
            "Resuma o currículo abaixo em português do Brasil.\n"
            "Diretrizes estritas:\n"
            "1. Use SOMENTE informações presentes no texto fornecido. Não invente ou assuma dados.\n"
            "2. Remova dados sensíveis de contato direto (como número de telefone ou e-mail exato).\n"
            "3. Escreva um texto objetivo de 5 a 8 linhas abordando: nome/perfil, principais experiências, "
            "tecnologias dominadas, formação acadêmica e um destaque profissional.\n\n"
            f"Texto do Currículo:\n{context}\n\n"
            "Resumo Estruturado:"
        )

        prompt_final = (
            f"<|im_start|>system\nVocê é um assistente de RH especialista. Gere apenas o resumo solicitado de 5 a 8 linhas, sem introduções.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        try:
            # Executa a inferência síncrona/bloqueante na CPU de forma isolada
            response = self.llm_service(
                prompt_final, max_tokens=self.max_new_tokens, temperature=0.3, stop=["<|im_end|>"]
            )
            return response["choices"][0]["text"].strip()
        except Exception:
            logger.exception("failed_to_generate_llm_summary")
            return "Erro crítico ao processar o sumário com o modelo local."

    async def justify(
        self, query: str, candidate: str, citations: list[str], max_new_tokens: int
    ) -> str:
        """
        Sintetiza uma resposta natural e inteligente baseada exclusivamente
        nas evidências reais coletadas pelo RankingService sem alucinações.
        """
        if not citations:
            return f"O candidato {candidate} não apresentou evidências explícitas no currículo para responder à pergunta: '{query}'."

        context = "\n\n".join(citations)[:MAX_JUSTIFICATION_SOURCE_CHARS]

        if self.llm_service is None:
            return f"Serviço de LLM indisponível. Citações brutas: {context[:150]}..."

        prompt = (
            "Você é um recrutador técnico experiente e analítico.\n"
            "Sua tarefa é responder à pergunta do recrutador baseando-se estritamente nas evidências reais do currículo fornecido abaixo.\n"
            "Se a pergunta for factual (ex: onde estudou ou se sabe python), extraia e diga o local ou fato de forma direta.\n"
            "Se for sobre competências ou aderência, explique em até 3 frases curtas como o candidato atende os requisitos.\n"
            "Se a resposta não puder ser confirmada pelas evidências fornecidas, responda exatamente: 'Informação não disponível no documento'.\n"
            "Não invente dados fora do bloco de evidências.\n\n"
            f"Pergunta do Recrutador: {query}\n\n"
            f"Nome do Candidato: {candidate}\n\n"
            f"Evidências encontradas no currículo:\n{context}\n\n"
            "Resposta Sintetizada:"
        )

        prompt_final = (
            f"<|im_start|>system\nVocê é um assistente de recrutamento que nunca inventa informações e escreve em português claro.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        try:
            response = self.llm_service(
                prompt_final,
                max_tokens=max_new_tokens,
                temperature=0.1,  # Baixa temperatura para manter o modelo focado e factual
                stop=["<|im_end|>"],
            )
            return response["choices"][0]["text"].strip()
        except Exception:
            logger.exception("failed_to_generate_llm_justification")
            return f"Erro ao processar a justificativa para {candidate}."
