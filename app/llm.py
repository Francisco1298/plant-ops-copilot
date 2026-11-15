"""
Model factory: the rest of the code never imports a provider directly.
Switching OpenAI -> Anthropic -> Ollama -> AWS Bedrock is a change in .env, not in code.
"""
import hashlib
import math
import re

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.config import Settings, get_settings

PROVIDER_PREFIX = {"openai": "openai", "anthropic": "anthropic", "ollama": "ollama", "bedrock": "bedrock_converse"}


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    if s.llm_provider == "offline":
        return OfflineChatModel(messages=iter([AIMessage(content="[offline mode] Configure LLM_PROVIDER in .env")] * 1000))
    kwargs = {"temperature": s.llm_temperature, "timeout": s.llm_timeout_s, "max_retries": s.llm_max_retries}
    if s.llm_provider == "bedrock":
        kwargs = {"temperature": s.llm_temperature, "region_name": s.aws_region}
    return init_chat_model(f"{PROVIDER_PREFIX[s.llm_provider]}:{s.llm_model}", **kwargs)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    match s.embeddings_provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=s.embeddings_model)
        case "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model=s.embeddings_model)
        case "bedrock":
            from langchain_aws import BedrockEmbeddings
            return BedrockEmbeddings(model_id=s.embeddings_model, region_name=s.aws_region)
        case _:
            return HashingEmbeddings()


class OfflineChatModel(GenericFakeChatModel):
    """Scripted model for tests: returns the messages you give it, in order. Supports tool calls."""

    def bind_tools(self, tools, **kwargs):  # the agent calls bind_tools; a scripted model ignores it
        return self


class HashingEmbeddings(Embeddings):
    """
    Tiny offline embedding (bag of words + hashing trick). NOT for production.
    Good enough to understand what an embedding is and to run tests without an API key:
    texts that share words end up with similar vectors (high cosine similarity).
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9_]{3,}", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
