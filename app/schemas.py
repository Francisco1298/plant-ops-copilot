"""Pydantic models = typed contracts shared by the LLM (structured output), REST and GraphQL."""
from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["How much biogas did we produce on 2026-10-19?"])
    session_id: str | None = Field(default=None, description="Reuse it to keep conversation memory")


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    session_id: str
    prompt_version: str
    latency_ms: int


class IncidentReport(BaseModel):
    text: str = Field(min_length=5, max_length=2000)


class IncidentClassification(BaseModel):
    """Schema the LLM must fill (structured output). Field descriptions guide the model."""
    severity: Literal["low", "medium", "high", "critical"] = Field(description="Impact level, see definitions")
    area: Literal["digester", "gas_handling", "chp_engine", "feeding", "instrumentation", "other"]
    requires_shutdown: bool = Field(description="True only if people or equipment are at immediate risk")
    summary: str = Field(description="One-sentence English summary of the incident")


class SearchHit(BaseModel):
    source: str
    content: str
    score: float
