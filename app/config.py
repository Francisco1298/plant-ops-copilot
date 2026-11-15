"""Centralised settings, read from environment variables or a .env file (12-factor style)."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
# Export .env to the process environment so provider SDKs (OpenAI, Anthropic, AWS) can read their API keys.
# Variables already set in the shell take precedence (override=False).
load_dotenv(ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # --- LLM -------------------------------------------------------------------------------
    # "offline" needs no API key: it uses a scripted model, useful for tests and CI.
    llm_provider: Literal["openai", "anthropic", "ollama", "bedrock", "offline"] = "ollama"
    llm_model: str = "qwen3:8b"            # e.g. gpt-4.1-mini | claude-haiku-5-5 | qwen3:8b
    llm_temperature: float = 0.0
    llm_timeout_s: int = 60
    llm_max_retries: int = 2

    # --- Embeddings ------------------------------------------------------------------------
    embeddings_provider: Literal["openai", "ollama", "bedrock", "offline"] = "ollama"
    embeddings_model: str = "nomic-embed-text"  # e.g. text-embedding-3-small

    # --- Vector database -------------------------------------------------------------------
    chroma_host: str | None = None        # set it to use a Chroma *server* (docker compose)
    chroma_port: int = 8000
    chroma_dir: Path = ROOT / "data" / "chroma"
    collection: str = "plant_manuals"

    # --- Data ------------------------------------------------------------------------------
    manuals_dir: Path = ROOT / "data" / "manuals"
    warehouse_db: Path = ROOT / "data" / "warehouse.db"

    # --- AWS -------------------------------------------------------------------------------
    aws_region: str = "us-east-1"


@lru_cache
def get_settings() -> Settings:
    return Settings()