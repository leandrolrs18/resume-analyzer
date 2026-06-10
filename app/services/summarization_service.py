import logging
from typing import Any

from app.schemas import RankingEvidence, ResumeDocument

logger = logging.getLogger(__name__)


class SummarizationService:
    def __init__(
        self,
        llm_service: Any | None,
        max_new_tokens: int,
        gemini_service: Any | None = None,
    ):
        # Recebe os providers já montados fora daqui.
        # llm_service costuma ser o modelo local; gemini_service é usado quando provider="gemini".
        self.llm_service = llm_service
        self.gemini_service = gemini_service
        self.max_new_tokens = max_new_tokens

    async def summarize(
        self,
        document: ResumeDocument,
        language: str = "pt",
        llm_provider: str = "local",
    ) -> str:
        # Fluxo de resumo:
        # 1) Normaliza o idioma pedido.
        # 2) Monta um fallback local a partir do texto extraído.
        # 3) Se houver LLM, gera um resumo controlado por prompt.
        # 4) Se a saída for válida, usa o texto do modelo; senão mantém o fallback.
        language = self._normalize_language(language)
        # Fallback entra primeiro para a API sempre ter resposta mesmo se o LLM falhar.
        fallback = self._fallback_summary(document, language, [])
        summary = fallback
        # Escolhe Gemini ou modelo local conforme llm_provider.
        llm = self._llm(llm_provider)
        if llm:
            try:
                # Entrada do LLM: texto extraído do currículo + instruções de estilo/segurança.
                prompt = self._summary_prompt(document, language)
                # Saída esperada: um parágrafo curto de resumo, sem JSON e sem markdown.
                generated = (await llm.generate(prompt, max(self.max_new_tokens, 220))).strip()
                # Se o modelo copiar contato/link do currículo, descarta e mantém fallback.
                if not self._looks_like_copied_resume(generated, document):
                    summary = generated
            except Exception:
                logger.exception("llm_summary_failed")
        return summary

    async def synthesize_ranked_results(
        self,
        query: str,
        language: str,
        llm_provider: str,
        evidence: list[RankingEvidence],
        documents_by_candidate: dict[str, ResumeDocument],
        max_new_tokens: int,
    ) -> dict[str, dict[str, str]]:
        # Fluxo de ranking sintetizado:
        # 1) Cria um fallback por candidato com resumo + justificativa.
        # 2) Filtra apenas evidências realmente ancoradas em citações.
        # 3) Gera prompt com trechos relevantes para o LLM.
        # 4) Faz parse da resposta e substitui o fallback só quando o texto é válido.
        # Entrada: ranking já calculado, citações escolhidas e documentos por candidato.
        # Saída: dict[candidato] com summary e justification para completar o RankingResult.
        fallback = {
            item.candidate: {
                "summary": self._fallback_summary(
                    documents_by_candidate[item.candidate],
                    language,
                    [citation.text for citation in item.citations],
                ),
                "justification": self._fallback_justification(query, language, item),
            }
            for item in evidence
        }
        llm = self._llm(llm_provider)
        # grounded remove candidatos sem score/citação, porque o LLM só deve explicar evidência real.
        grounded = [item for item in evidence if item.score > 0 and item.citations]
        if not llm or not grounded:
            return fallback

        try:
            # Prompt final recebe só candidatos com relevância alta; o score já veio do RankingService.
            llm_evidence = [item for item in grounded if item.score >= 0.5]
            if not llm_evidence:
                return fallback
            llm_candidate_names = {item.candidate for item in llm_evidence}
            prompt = self._prompt(query, language, llm_evidence)
            # LLM retorna texto em formato fixo; _blocks transforma esse texto em dict.
            raw_response = await llm.generate(prompt, max_new_tokens)
            parsed = self._blocks(raw_response)
            for candidate_key, item in parsed.items():
                matched_key = candidate_key if candidate_key in fallback else None
                if not matched_key:
                    logger.warning(
                        f"Could not match LLM candidate '{candidate_key}' to any fallback key: {list(fallback.keys())}"
                    )
                    continue
                if matched_key not in llm_candidate_names:
                    continue
                summary = item.get("summary", "").strip()
                # Summary só precisa existir; a limpeza pesada já foi retirada para confiar no prompt.
                if summary:
                    fallback[matched_key]["summary"] = summary
                justification = item.get("justification", "").strip()
                if justification:
                    fallback[matched_key]["justification"] = justification
        except Exception:
            logger.exception("single_llm_synthesis_failed")
        return fallback

    def _llm(self, provider: str):
        # Escolhe o provider de LLM configurado: Gemini quando disponível, senão local.
        if provider == "gemini" and self.gemini_service:
            return self.gemini_service
        return self.llm_service

    @staticmethod
    def _summary_prompt(document: ResumeDocument, language: str) -> str:
        # Prompt de resumo: usa só o texto extraído e pede saída curta, neutra e sem inventar.
        lang = "inglês" if language == "en" else "português do Brasil"
        evidence = document.extracted_text[:3000]
        return (
            f"Idioma obrigatório: {lang}. Use somente os dados do currículo abaixo, "
            "sem inventar. "
            "Retorne um único parágrafo natural com 5 a 8 frases curtas. Escreva em terceira "
            "pessoa, como avaliador de recrutamento. Cubra perfil profissional, formação, "
            "experiência, competências, projetos ou senioridade aparente. Não use inglês "
            "quando o idioma obrigatório for português. Não use markdown, numeração, bullets, "
            "rótulos, "
            "dados de contato ou cabeçalho copiado do currículo. Não diga 'perfil baseado nos "
            "dados extraídos'. Não repita a mesma tecnologia várias vezes.\n\n"
            f"Candidato: {document.candidate}\n\n"
            f"Trecho do currículo:\n{evidence}"
        )

    @staticmethod
    def _prompt(query: str, language: str, evidence: list[RankingEvidence]) -> str:
        # Prompt de ranking: passa query + evidências e força formato fixo por candidato.
        lang = "inglês" if language == "en" else "português do Brasil"
        required_candidates = [item.candidate for item in evidence]
        blocks = []
        for item in evidence:
            citations = "\n".join(
                f"- {' '.join(citation.text.split())[:220]}..." for citation in item.citations[:2]
            )
            blocks.append(f"Candidato: {item.candidate}\nEvidências:\n{citations}")
        return (
            f"Responda em {lang}. Use somente as evidências abaixo, sem inventar dados. "
            f"Você deve responder todos os {len(required_candidates)} candidatos listados. "
            "Não omita nenhum candidato da lista obrigatória. "
            "Escreva como avaliador de recrutamento, sempre em terceira pessoa. "
            "Não copie frases do currículo literalmente. Não use primeira pessoa, como "
            "'eu', 'meu', 'fui', 'atuei', 'apliquei', 'liderei' ou 'tenho'. "
            "Não inclua citações diretas, trechos entre aspas ou cópias literais do currículo na justificativa. "
            "A justificativa deve conter apenas referências indiretas, resumindo a experiência e a aderência conceitualmente. "
            "Na justificativa, comece pelo nome do candidato e explique em detalhes o critério de ranking, "
            "comparando de forma analítica a aderência, senioridade, duração, cargos, projetos "
            "ou complexidade quando houver evidência.\n\n"
            "Lista obrigatória de candidatos:\n"
            + "\n".join(f"- {candidate}" for candidate in required_candidates)
            + "\n\n"
            "Para cada candidato, retorne exatamente este formato, sem JSON e sem markdown:\n"
            "CANDIDATO: nome (use exatamente o identificador fornecido após 'Candidato: ')\n"
            "RESUMO: um parágrafo corrido com 3 frases curtas\n"
            "JUSTIFICATIVA: um parágrafo objetivo com 2 a 3 frases explicando detalhadamente o alinhamento do candidato\n"
            "FIM\n\n"
            f"Pergunta: {query}\n\n" + "\n\n".join(blocks)
        )

    @staticmethod
    def _blocks(raw: str) -> dict[str, dict[str, str]]:
        # Faz parse do retorno do LLM em blocos por candidato.
        # Entrada esperada:
        # CANDIDATO: nome
        # RESUMO: ...
        # JUSTIFICATIVA: ...
        # FIM
        parsed: dict[str, dict[str, str]] = {}
        current: str | None = None
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("CANDIDATO:"):
                # Abre um novo bloco para o candidato atual.
                current = line.split(":", maxsplit=1)[1].strip()
                parsed[current] = {"summary": "", "justification": ""}
            elif current and line.upper().startswith("RESUMO:"):
                # Guarda somente o texto depois do rótulo RESUMO.
                parsed[current]["summary"] = line.split(":", maxsplit=1)[1].strip()
            elif current and line.upper().startswith("JUSTIFICATIVA:"):
                # Guarda somente o texto depois do rótulo JUSTIFICATIVA.
                parsed[current]["justification"] = line.split(":", maxsplit=1)[1].strip()
        if not parsed:
            raise ValueError("LLM did not return candidate blocks")
        return parsed

    @classmethod
    def _fallback_summary(
        cls,
        document: ResumeDocument,
        language: str,
        citations: list[str],
    ) -> str:
        # Resumo de fallback: monta uma versão segura usando trecho limpo do texto extraído.
        words = []
        for word in document.extracted_text.split():
            lower = word.lower()
            if (
                "@" in lower
                or "http" in lower
                or "linkedin" in lower
                or "github" in lower
                or "tel" in lower
            ):
                continue
            words.append(word)
        snippet = " ".join(words[:25]) + "..."
        if language == "en":
            return " ".join(
                [
                    f"{document.candidate} presents a professional background from the resume.",
                    f"The extracted text snippet includes: {snippet}",
                    "No structured profile could be loaded for validation.",
                    "Extracted context shows relevant information.",
                    "Relevant projects or activities appear in the text.",
                    "The profile should be reviewed with the original resume evidence.",
                ]
            )
        return " ".join(
            [
                f"{document.candidate} apresenta trajetória profissional descrita no currículo.",
                f"O texto extraído do candidato inclui: {snippet}",
                "A formação e competências constam no arquivo original.",
                "O perfil apresenta aderência técnica para a área.",
                "Projetos e experiências adicionais aparecem no currículo.",
                "A avaliação deve considerar as evidências originais do currículo.",
            ]
        )

    @classmethod
    def _fallback_justification(cls, query: str, language: str, item: RankingEvidence) -> str:
        # Justificativa de fallback: explica aderência sem depender do LLM.
        del language
        if not item.citations:
            return f"{item.candidate} não apresentou evidências fortes para a pergunta: {query}."
        return (
            f'{item.candidate} se destacou para "{query}" porque há evidências de '
            "experiências e qualificações correlatas em seu histórico profissional."
        )

    @classmethod
    def _looks_like_copied_resume(cls, value: str, document: ResumeDocument) -> bool:
        # Bloqueia saída que pareça copiar contato/link do currículo.
        del document
        normalized_value = value.casefold()
        return any(
            marker in normalized_value for marker in ("@", "http", "linkedin.com", "github.com")
        )

    @staticmethod
    def _normalize_language(language: str) -> str:
        # Só aceita pt ou en para manter o comportamento previsível.
        return "en" if language == "en" else "pt"
