import logging
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from app.schemas import ResumeStructuredProfile

logger = logging.getLogger(__name__)

SECTION_TITLES = {
    "education": (
        "formacao",
        "formacao academica",
        "education",
        "academic background",
        "educational history",
    ),
    "experience": (
        "experiencia",
        "experiencia profissional",
        "experience",
        "work experience",
        "employment history",
        "professional experience",
    ),
    "skills": (
        "competencias tecnicas",
        "competencias",
        "habilidades",
        "core skills",
        "skills",
        "technical skills",
        "proficient skills",
    ),
    "certifications": (
        "certificacoes",
        "certificados",
        "certifications",
        "certificates",
        "courses",
        "workshops",
    ),
    "projects": ("projetos", "projects", "publications", "publication"),
    "languages": ("idiomas", "languages"),
}


class ResumeParserService:
    def __init__(self, spacy_module: Any | None = None):
        self._spacy = spacy_module
        self._nlp = None
        self._spacy_loaded = False

    async def parse(self, text: str, llm_provider: str = "local") -> ResumeStructuredProfile:
        del llm_provider
        lines = self._clean_lines(text)
        sections = self._split_sections(lines)
        name = self._name_from_spacy(text) or self._name_from_lines(lines)
        return ResumeStructuredProfile(
            name=name,
            education=self._items(sections["education"]),
            experience=self._items(sections["experience"]),
            skills=self._skills(sections["skills"]),
            certifications=self._items(sections["certifications"]),
            projects=self._items(sections["projects"]),
            languages=self._items(sections["languages"]),
        )

    def semantic_views(self, candidate: str, profile: ResumeStructuredProfile) -> list[str]:
        views = {
            "perfil": [profile.name or candidate],
            "formação": profile.education,
            "experiência": profile.experience,
            "competências": profile.skills,
            "certificações": profile.certifications,
            "projetos": profile.projects,
            "idiomas": profile.languages,
        }
        return [f"{label}: {'; '.join(values[:8])}" for label, values in views.items() if values]

    @classmethod
    def _split_sections(cls, lines: list[str]) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = defaultdict(list)
        current: str | None = None
        for line in lines:
            section, remainder = cls._section_header(line)
            if section:
                current = section
                if remainder:
                    sections[current].append(remainder)
                continue
            if current:
                sections[current].append(line)
        return sections

    @classmethod
    def _section_header(cls, line: str) -> tuple[str | None, str]:
        normalized = cls._normalize(line).strip(" :-|")
        for section, titles in SECTION_TITLES.items():
            for title in sorted(titles, key=len, reverse=True):
                if normalized == title:
                    return section, ""
                if " " in title and normalized.startswith(f"{title} ") and len(line.split()) <= 8:
                    return section, line[len(title) :].strip(" :-|")
        return None, ""

    @staticmethod
    def _clean_lines(text: str) -> list[str]:
        lines = []
        for raw in text.replace("\u200b", " ").splitlines():
            line = " ".join(raw.replace("•", " ").replace("●", " ").split())
            if line and not ResumeParserService._looks_like_contact(line):
                lines.append(line[:260])
        return lines

    @staticmethod
    def _looks_like_contact(line: str) -> bool:
        lower = line.lower()
        return "@" in lower or "http" in lower or "linkedin" in lower or "github" in lower

    @staticmethod
    def _items(values: Iterable[str], limit: int = 8) -> list[str]:
        items: list[str] = []
        for value in values:
            text = value.strip(" -–:;")
            if len(text) >= 4 and text not in items:
                items.append(text[:220])
            if len(items) >= limit:
                break
        return items

    @classmethod
    def _skills(cls, values: Iterable[str]) -> list[str]:
        items: list[str] = []
        for value in values:
            for part in value.replace("|", ",").replace(";", ",").split(","):
                skill = cls._clean_skill(part)
                normalized_items = {cls._normalize(item) for item in items}
                if 2 <= len(skill) <= 60 and cls._normalize(skill) not in normalized_items:
                    items.append(skill)
                if len(items) >= 12:
                    return items
        return items

    @classmethod
    def _clean_skill(cls, value: str) -> str:
        skill = value.split(":", maxsplit=1)[-1].strip(" -–:;.")
        normalized = cls._normalize(skill)
        for prefix in (
            "linguagens de programacao ",
            "programming languages ",
            "artificial intelligence & machine learning ",
            "inteligencia artificial & machine learning ",
            "backend & arquitetura ",
            "backend & architecture ",
            "cloud & devops ",
        ):
            if normalized.startswith(prefix):
                skill = skill[len(prefix) :].strip(" -–:;.")
                normalized = cls._normalize(skill)
        if normalized in {
            "tecnicas",
            "skills",
            "core skills",
            "technical skills",
            "linguagens de programacao",
            "programming languages",
            "backend & arquitetura",
            "backend & architecture",
            "cloud & devops",
        }:
            return ""
        return skill

    def _name_from_spacy(self, text: str) -> str | None:
        nlp = self._load_spacy()
        if nlp is None:
            return None
        try:
            doc = nlp(text[:1200])
        except Exception:
            logger.exception("spacy_ner_failed")
            return None
        for entity in doc.ents:
            if entity.label_ in {"PER", "PERSON"} and 2 <= len(entity.text.split()) <= 5:
                return " ".join(entity.text.split())[:120]
        return None

    def _load_spacy(self):
        if self._spacy_loaded:
            return self._nlp
        self._spacy_loaded = True
        try:
            spacy = self._spacy
            if spacy is None:
                import spacy as spacy  # noqa: PLC0415

            for model in ("pt_core_news_sm", "en_core_web_sm", "xx_ent_wiki_sm"):
                try:
                    self._nlp = spacy.load(model)
                    logger.info("spacy_model_loaded", extra={"model": model})
                    return self._nlp
                except OSError:
                    continue
        except ModuleNotFoundError:
            logger.info("spacy_not_installed")
        except Exception:
            logger.exception("spacy_load_failed")
        return None

    @staticmethod
    def _name_from_lines(lines: list[str]) -> str | None:
        for line in lines[:8]:
            words = line.split()
            if 2 <= len(words) <= 5 and not line.endswith(":"):
                return line[:120]
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        return "".join(char for char in normalized if not unicodedata.combining(char))
