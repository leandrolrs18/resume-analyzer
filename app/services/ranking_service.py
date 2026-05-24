import logging
import re
from app.schemas import Citation, RankingEvidence, ResumeDocument

logger = logging.getLogger(__name__)


class RankingService:
    def __init__(self, top_k_citations: int = 3):
        self.top_k_citations = top_k_citations
        self.llm = None 

    async def rank(self, query: str, documents: list[ResumeDocument]) -> list[RankingEvidence]:
        """
        Avalia a aderência semântica analisando o texto corrido, garantindo que o 
        modelo local extraia a formação correta sem falhar por causa de formato JSON.
        """
        evidences: list[RankingEvidence] = []

        if self.llm is None:
            logger.warning("Instância do LLM não injetada no RankingService.")
            return [RankingEvidence(candidate=d.candidate, score=0.0, citations=[]) for d in documents]

        for document in documents:
            # Mantemos uma margem maior para garantir que a formação acadêmica (geralmente no fim) não seja cortada
            full_context = "\n".join([f"[Bloco {c.chunk_id}]: {c.text}" for c in document.chunks])
            full_context = full_context[:6000] 

            prompt_sistema = (
                "Você é um triador de currículos especialista.\n"
                "Analise o texto fornecido e encontre as respostas para a pergunta do recrutador.\n"
                "Sua resposta deve seguir obrigatoriamente este formato de duas linhas:\n"
                "NOTA: [insira aqui uma nota de 0.0 a 1.0]\n"
                "BLOCOS: [insira aqui os números dos blocos que comprovam a resposta, separados por vírgula]"
            )

            prompt_usuario = f"Pergunta: {query}\n\nCurrículo:\n{full_context}"

            prompt_final = (
                f"<|im_start|>system\n{prompt_sistema}<|im_end|>\n"
                f"<|im_start|>user\n{prompt_usuario}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            try:
                # Mudamos para max_tokens menor, acelerando drasticamente o processamento
                response = self.llm(
                    prompt_final,
                    max_tokens=60,
                    temperature=0.0, # Temperatura zero para evitar alucinações de nota
                    stop=["<|im_end|>"]
                )
                
                resposta_texto = response["choices"][0]["text"].strip()
                logger.info(f"Resposta bruta do Qwen para {document.candidate}:\n{resposta_texto}")

                # Extração robusta por Regex das linhas de NOTA e BLOCOS
                score_match = re.search(r"NOTA:\s*([0-9.]+)", resposta_texto, re.IGNORECASE)
                blocos_match = re.search(r"BLOCOS:\s*([0-9, ]+)", resposta_texto, re.IGNORECASE)

                score = float(score_match.group(1)) if score_match else 0.0
                
                blocos_ids = []
                if blocos_match:
                    # Coleta todos os números separados por vírgula ou espaço
                    blocos_ids = [int(x) for x in re.findall(r"\d+", blocos_match.group(1))]

                # Reconstroi as citações baseadas nos blocos retornados
                top_citations = []
                for chunk in document.chunks:
                    if chunk.chunk_id in blocos_ids:
                        top_citations.append(Citation(chunk_id=chunk.chunk_id, text=chunk.text))
                
                top_citations = top_citations[:self.top_k_citations]

            except Exception:
                logger.exception(f"Falha na análise semântica do candidato {document.candidate}")
                score = 0.0
                top_citations = []

            evidences.append(
                RankingEvidence(
                    candidate=document.candidate,
                    score=round(score, 4),
                    citations=top_citations,
                )
            )

        return sorted(evidences, key=lambda item: item.score, reverse=True)