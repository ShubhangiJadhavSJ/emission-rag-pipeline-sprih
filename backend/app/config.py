"""Central application configuration, loaded from environment variables.

Every tunable lives here so the rest of the code never reads os.environ
directly. Values come from docker-compose (which forwards them from .env).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database ---
    db_host: str = "mariadb"
    db_port: int = 3306
    db_name: str = "emissions"
    db_user: str = "emissions"
    db_password: str = "emissions"

    # --- Qdrant ---
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "emission_chunks"

    # --- LLM ---
    # "groq" (free hosted, default) | "ollama" (free local) | "anthropic" | "openai"
    llm_provider: str = "groq"
    # Groq — free hosted API, OpenAI-compatible, nothing to download.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Ollama (local, free — only if you prefer a fully offline model)
    ollama_host: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:3b"
    # Hosted paid providers (only used if selected)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-8"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # bge-small-en-v1.5 output dimensionality

    # --- Tracing (Langfuse optional) ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Storage / data ---
    blob_storage_dir: str = "/data/blobs"
    ground_truth_path: str = "/data/ground_truth/ground_truth.json"

    # --- Pipeline defaults (used by the live UI upload flow) ---
    default_prompt_version: str = "v3"
    default_chunk_size: int = 1200
    default_chunk_overlap: int = 200
    default_retrieval_k: int = 6

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
