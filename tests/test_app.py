"""
Tests run fully offline (no API key, no network): scripted LLM + hashing embeddings.
Run: pytest -q
"""
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import rag
from app.agent import PlantCopilot
from app.config import get_settings
from app.llm import OfflineChatModel
from app.tools import query_kpis, search_manuals


@pytest.fixture(autouse=True)
def offline_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "offline")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    rag.build_index()
    yield
    get_settings.cache_clear()


# ----------------------------------------------------------------------------- tools
def test_query_kpis_rejects_dates_outside_available_range():
    out = query_kpis.invoke({"start_date": "2027-01-01", "end_date": "2027-01-31", "metrics": ["biogas_nm3"]})
    assert "outside the available range" in out

def test_query_kpis_returns_known_value():
    out = query_kpis.invoke({"start_date": "2026-10-19", "end_date": "2026-10-19", "metrics": ["biogas_nm3"]})
    assert "2463" in out


@pytest.mark.parametrize("args, error", [
    ({"start_date": "19/10/2026", "end_date": "2026-10-19", "metrics": ["biogas_nm3"]}, "YYYY-MM-DD"),
    ({"start_date": "2026-10-01", "end_date": "2026-10-02", "metrics": ["password; DROP TABLE x"]}, "unknown metrics"),
    ({"start_date": "2026-10-05", "end_date": "2026-10-01", "metrics": ["biogas_nm3"]}, "start_date must be"),
])
def test_query_kpis_rejects_bad_input(args, error):
    assert error in query_kpis.invoke(args)


def test_rag_finds_the_right_manual():
    hits = rag.search("boiler trip digester temperature low", k=1)
    assert hits[0][0].metadata["source"] == "sop_digester_temperature.md"
    assert "[sop_gas_safety_h2s.md]" in search_manuals.invoke({"query": "H2S detector alarm gas leak safety"})


def test_index_is_idempotent():
    n = rag.build_index()
    assert len(rag.get_vector_store().get()["ids"]) == n


# ----------------------------------------------------------------------------- agent
def scripted_copilot(*replies) -> PlantCopilot:
    return PlantCopilot(model=OfflineChatModel(messages=iter(replies)))


def test_agent_calls_tool_and_answers():
    copilot = scripted_copilot(
        AIMessage(content="", tool_calls=[{"name": "query_kpis", "id": "1",
                                           "args": {"start_date": "2026-10-19", "end_date": "2026-10-19",
                                                    "metrics": ["biogas_nm3"]}}]),
        AIMessage(content="On 2026-10-19 the plant produced 2,463 Nm3."),
    )
    r = copilot.ask("Biogas on 2026-10-19?")
    assert r.tools_used == ["query_kpis"]
    assert "2,463" in r.answer


def test_agent_memory_per_session():
    copilot = scripted_copilot(AIMessage(content="first"), AIMessage(content="second"))
    s = copilot.ask("hello").session_id
    state = copilot.agent.get_state({"configurable": {"thread_id": s}})
    copilot.ask("again", session_id=s)
    state2 = copilot.agent.get_state({"configurable": {"thread_id": s}})
    assert len(state2.values["messages"]) == len(state.values["messages"]) + 2


# ----------------------------------------------------------------------------- API
@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_rest_kpis_and_validation(client):
    rows = client.get("/v1/kpis", params={"start": "2026-10-18", "end": "2026-10-20"}).json()
    assert [r["fecha"] for r in rows] == ["2026-10-18", "2026-10-19", "2026-10-20"]
    assert client.get("/v1/kpis", params={"start": "yesterday"}).status_code == 422
    assert client.post("/v1/ask", json={"question": "?"}).status_code == 422


def test_rest_ask_with_scripted_model(client):
    client.app.state.copilot = scripted_copilot(AIMessage(content="All good."))
    body = client.post("/v1/ask", json={"question": "Status of the plant?"}).json()
    assert body["answer"] == "All good." and body["tools_used"] == []


def test_graphql_returns_only_requested_fields(client):
    q = '{ kpis(start: "2026-10-19", end: "2026-10-19") { fecha estadoOperacion } }'
    data = client.post("/graphql", json={"query": q}).json()["data"]["kpis"]
    assert data == [{"fecha": "2026-10-19", "estadoOperacion": "CRITICO"}]
