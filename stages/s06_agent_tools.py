"""
STAGE 6 - Agent with tools: the LLM decides which tool to call (SQL KPIs or manuals) and in what order.
Run: python -m stages.s06_agent_tools        (needs an LLM that supports tool calling)
"""
from app import rag
from app.agent import PlantCopilot

rag.build_index()
copilot = PlantCopilot()

questions = [
    "How much biogas did we produce on 2026-10-19 and what was the status?",           # -> query_kpis
    "What should I do if the digester temperature alarm goes off?",                    # -> search_manuals
    "Which day had the worst temperature problem, how much electricity did we lose "
    "compared with a normal day, and what does the procedure say to do?",               # -> both tools
]
session = None
for q in questions:
    r = copilot.ask(q, session_id=session)
    session = r.session_id
    print("=" * 100, f"\nQ: {q}\nTOOLS: {r.tools_used} | {r.latency_ms} ms\n{r.answer}")

# Memory: this follow-up only makes sense if the agent remembers the previous turn (same session_id)
r = copilot.ask("Summarise your last answer in one sentence for my manager.", session_id=session)
print("=" * 100, "\nFOLLOW-UP:", r.answer)
