"""
STAGE 1 - Modern Python for AI engineering (runs offline, no API key)

Concepts: type hints, Pydantic validation, comprehensions, error handling, env vars, async concurrency.
Run: python -m stages.s01_python_for_ai
"""
import asyncio
import os
import random
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


# 1) Type hints + Pydantic: data contracts that validate themselves --------------------------------
class SensorReading(BaseModel):
    tag: str = Field(pattern=r"^[A-Z]{2}_\d{3}$")   # e.g. TT_101
    value: float
    unit: str
    quality: Literal["GOOD", "BAD", "UNCERTAIN"] = "GOOD"


def demo_pydantic() -> None:
    ok = SensorReading(tag="TT_101", value="37.2", unit="C")   # "37.2" is coerced to float
    print("valid  :", ok.model_dump())
    try:
        SensorReading(tag="temp", value="hot", unit="C")
    except ValidationError as e:
        print("invalid:", e.error_count(), "errors ->", [err["loc"][0] for err in e.errors()])
    # JSON in / JSON out: exactly what APIs and LLM structured outputs use
    print("json   :", ok.model_dump_json())
    try:
        SensorReading(tag="TT_101", value=37, unit="C", quality="BUENO")
        print("ERROR: aceptó una calidad inválida")
    except ValidationError:
        print("OK: rechazó quality='BUENO'")


# 2) Comprehensions and small pure functions --------------------------------------------------------
def demo_comprehensions() -> None:
    readings = [SensorReading(tag="TT_101", value=36 + random.random() * 2, unit="C") for _ in range(5)]
    out_of_range = [r.value for r in readings if not 35 <= r.value <= 37.5]
    by_tag = {r.tag: round(r.value, 2) for r in readings}
    print("out of range:", out_of_range, "| last per tag:", by_tag)


# 3) Configuration from environment variables (never hard-code API keys) ---------------------------
def demo_env() -> None:
    key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY set?", bool(key), "(the value itself is never printed or committed)")


# 4) Async: many slow I/O calls at the same time (this is how you call LLMs/APIs at scale) ----------
async def fake_llm_call(i: int, sem: asyncio.Semaphore) -> str:
    async with sem:                 # wait here until one of the "cashiers" is free
        print(f"  call {i} started")
        await asyncio.sleep(1)      # simulates network latency of an LLM / REST call
        return f"answer {i}"


async def demo_async() -> None:
    sem = asyncio.Semaphore(5)      # max 3 calls at the same time
    t0 = time.perf_counter()
    results = await asyncio.gather(*(fake_llm_call(i, sem) for i in range(10)))
    print(f"10 calls with max 3 at a time took {time.perf_counter() - t0:.1f} s ->", results[:3], "...")


# 5) Retries with exponential backoff (LLM APIs fail and rate-limit: 429, 503...) ------------------
def with_retries(fn, attempts: int = 4, base_delay: float = 0.2):
    for n in range(1, attempts + 1):
        try:
            return fn()
        except ConnectionError as e:
            if n == attempts:
                raise
            delay = base_delay * 2 ** (n - 1)
            print(f"  attempt {n} failed ({e}); retrying in {delay:.1f} s")
            time.sleep(delay)


_calls = {"n": 0}


def flaky_service() -> str:
    """Fails the first 2 times, then works: like an API that is briefly overloaded."""
    _calls["n"] += 1
    if _calls["n"] <= 2:
        raise ConnectionError("503 Service Unavailable")
    return "200 OK"


if __name__ == "__main__":
    random.seed(7)
    print("\n== Pydantic ==");        demo_pydantic()
    print("\n== Comprehensions ==");  demo_comprehensions()
    print("\n== Env vars ==");        demo_env()
    print("\n== Async ==");           asyncio.run(demo_async())
    print("\n== Retries ==");         print("result:", with_retries(flaky_service))
