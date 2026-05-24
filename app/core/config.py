from functools import lru_cache
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "resume-analyzer"
    # O Pydantic tenta ler a variável MONGO_URI do ambiente, se não achar, usa o valor padrão local
    mongo_uri: str = Field(default="mongodb://mongodb:27017")
    mongo_db: str = Field(default="resume-analyzer") # Mudamos o nome padrão aqui!
    hf_model: str = Field(default="google/flan-t5-small")
    use_local_llm: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024)
    max_files: int = Field(default=10)
    max_pages_per_document: int = Field(default=15)
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    top_k_citations: int = Field(default=3)
    summarization_max_new_tokens: int = Field(default=120)
    justification_max_new_tokens: int = Field(default=96)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()