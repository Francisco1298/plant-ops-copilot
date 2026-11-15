"""
STAGE 7 - Consuming the API: REST vs GraphQL, and concurrent requests.
First start the server in another terminal:  uvicorn app.main:app --reload
Run: python -m stages.s07_api_client
"""
import asyncio
import time

import httpx

BASE = "http://localhost:8000"

# 1) REST: one URL per resource, the server decides the response shape
r = httpx.get(f"{BASE}/v1/kpis", params={"start": "2026-10-18", "end": "2026-10-20"})
print("REST status", r.status_code, "| fields per row:", len(r.json()[0]))

# 2) GraphQL: one endpoint, the CLIENT asks for exactly the fields it needs
query = """
query ($s: String!, $e: String!) {
  kpis(start: $s, end: $e) { fecha biogasNm3 estadoOperacion }
}"""
g = httpx.post(f"{BASE}/graphql", json={"query": query, "variables": {"s": "2026-10-18", "e": "2026-10-20"}})
print("GraphQL     :", g.json()["data"]["kpis"])

# 3) Validation errors come back as 422 with a clear message (FastAPI + Pydantic)
bad = httpx.post(f"{BASE}/v1/ask", json={"question": "?"})
print("Invalid body:", bad.status_code, bad.json()["detail"][0]["msg"])


# 4) Many clients at the same time: async requests
async def load_test(n: int = 20):
    async with httpx.AsyncClient(base_url=BASE, timeout=30) as client:
        t0 = time.perf_counter()
        responses = await asyncio.gather(*(client.get("/v1/manuals/search", params={"q": "H2S safety"}) for _ in range(n)))
        ok = sum(r.status_code == 200 for r in responses)
        print(f"{n} concurrent requests -> {ok} OK in {time.perf_counter() - t0:.2f} s")

asyncio.run(load_test())

# 5) The copilot through REST (needs an LLM configured on the server)
ask = httpx.post(f"{BASE}/v1/ask", json={"question": "What was the average CH4 % in October 2026?"}, timeout=120)
print("Copilot     :", ask.status_code, ask.json())
