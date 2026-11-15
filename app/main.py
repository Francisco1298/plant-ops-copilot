"""
API layer: REST (FastAPI) + GraphQL (Strawberry) over the same business logic.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs  (Swagger)   |   http://localhost:8000/graphql  (GraphiQL)
"""
import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated

import strawberry
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from strawberry.fastapi import GraphQLRouter

from app import rag
from app.agent import PlantCopilot
from app.config import get_settings
from app.schemas import AskRequest, AskResponse, IncidentClassification, IncidentReport, SearchHit
from app.tools import ALLOWED_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create heavy objects ONCE per process, not per request.
    app.state.copilot = PlantCopilot()
    # First start (e.g. a fresh container): index the manuals if the vector DB is empty.
    if not rag.get_vector_store().get(limit=1)["ids"]:
        log.info("Vector DB empty -> indexing manuals (%d chunks)", rag.build_index())
    log.info("Copilot ready (provider=%s, model=%s)", get_settings().llm_provider, get_settings().llm_model)
    yield


app = FastAPI(title="Plant Ops Copilot API", version="1.0.0", lifespan=lifespan,
              description="LLM agent for biogas plant operations: KPIs (SQL) + procedures (RAG).")


def get_copilot(request: Request) -> PlantCopilot:
    return request.app.state.copilot


Copilot = Annotated[PlantCopilot, Depends(get_copilot)]


def fetch_kpis(start: str, end: str) -> list[dict]:
    with sqlite3.connect(get_settings().warehouse_db) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM gold_kpi_diario WHERE fecha BETWEEN ? AND ? ORDER BY fecha", (start, end))
        return [dict(r) for r in rows]


# ------------------------------------------------------------------------------------------- REST
@app.get("/health", tags=["ops"])
def health():
    """Liveness probe for load balancers / container orchestrators."""
    return {"status": "ok", "provider": get_settings().llm_provider}


@app.post("/v1/ask", response_model=AskResponse, tags=["copilot"])
async def ask(body: AskRequest, copilot: Copilot):
    # LLM calls are slow and blocking -> run in a worker thread so the event loop keeps serving requests.
    try:
        return await run_in_threadpool(copilot.ask, body.question, body.session_id)
    except Exception as e:  # upstream LLM errors -> 502, never a raw stack trace to the client
        log.exception("ask failed")
        raise HTTPException(status_code=502, detail=f"LLM provider error: {type(e).__name__}") from e


@app.post("/v1/incidents/classify", response_model=IncidentClassification, tags=["copilot"])
async def classify(body: IncidentReport, copilot: Copilot):
    try:
        return await run_in_threadpool(copilot.classify_incident, body.text)
    except Exception as e:
        log.exception("classify failed")
        raise HTTPException(status_code=502, detail=f"LLM provider error: {type(e).__name__}") from e


@app.get("/v1/kpis", tags=["data"])
def kpis(start: str = Query("2026-10-01", pattern=r"^\d{4}-\d{2}-\d{2}$"),
         end: str = Query("2026-10-30", pattern=r"^\d{4}-\d{2}-\d{2}$")):
    """Plain data endpoint, no LLM involved."""
    return fetch_kpis(start, end)


@app.get("/v1/manuals/search", response_model=list[SearchHit], tags=["data"])
def search_manuals(q: str = Query(min_length=3), k: int = Query(3, ge=1, le=10)):
    return [SearchHit(source=d.metadata["source"], content=d.page_content, score=s) for d, s in rag.search(q, k=k)]


# ---------------------------------------------------------------------------------------- GraphQL
@strawberry.type
class DailyKpi:
    fecha: str
    biogas_nm3: float
    ch4_pct_prom: float
    electricidad_potencial_kwh: float
    temp_min_c: float
    estado_operacion: str


@strawberry.type
class CopilotAnswer:
    answer: str
    tools_used: list[str]
    session_id: str


@strawberry.type
class GqlQuery:
    @strawberry.field(description="Daily KPIs. The client chooses exactly which fields it needs.")
    def kpis(self, start: str = "2026-10-01", end: str = "2026-10-30") -> list[DailyKpi]:
        fields = DailyKpi.__annotations__.keys()
        return [DailyKpi(**{f: r[f] for f in fields}) for r in fetch_kpis(start, end)]

    @strawberry.field
    def available_metrics(self) -> list[str]:
        return list(ALLOWED_METRICS)


@strawberry.type
class GqlMutation:
    @strawberry.mutation(description="Ask the copilot (it may call tools).")
    async def ask(self, info: strawberry.Info, question: str, session_id: str | None = None) -> CopilotAnswer:
        r = await run_in_threadpool(info.context["request"].app.state.copilot.ask, question, session_id)
        return CopilotAnswer(answer=r.answer, tools_used=r.tools_used, session_id=r.session_id)


app.include_router(GraphQLRouter(strawberry.Schema(query=GqlQuery, mutation=GqlMutation)), prefix="/graphql")
