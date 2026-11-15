"""
The agent = LLM + tools + system prompt + memory, orchestrated by LangGraph (via LangChain's create_agent).
Loop: the model decides -> calls a tool -> reads the result -> decides again -> final answer.
"""
import time
import uuid

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.llm import get_chat_model
from app.prompts import AGENT_SYSTEM_PROMPT, INCIDENT_CLASSIFIER_PROMPT, PROMPT_VERSION
from app.schemas import AskResponse, IncidentClassification
from app.tools import ALL_TOOLS


class PlantCopilot:
    def __init__(self, model: BaseChatModel | None = None):
        self.model = model or get_chat_model()
        # InMemorySaver keeps conversation memory per session_id inside THIS process.
        # In a distributed deployment (several API replicas) you would use a shared store
        # such as langgraph-checkpoint-postgres or Redis so any replica can continue a session.
        self.agent = create_agent(self.model, tools=ALL_TOOLS, system_prompt=AGENT_SYSTEM_PROMPT,
                                  checkpointer=InMemorySaver())

    def ask(self, question: str, session_id: str | None = None) -> AskResponse:
        session_id = session_id or uuid.uuid4().hex
        t0 = time.perf_counter()
        result = self.agent.invoke({"messages": [{"role": "user", "content": question}]},
                                   config={"configurable": {"thread_id": session_id}})
        msgs = result["messages"]
        # tools used in THIS turn = tool messages after the last human message
        last_human = max(i for i, m in enumerate(msgs) if m.type == "human")
        tools_used = [m.name for m in msgs[last_human:] if isinstance(m, ToolMessage)]
        answer = next(m.content for m in reversed(msgs) if isinstance(m, AIMessage) and m.content)
        if isinstance(answer, list):  # some providers return content blocks
            answer = "".join(b.get("text", "") for b in answer if isinstance(b, dict))
        return AskResponse(answer=answer, tools_used=tools_used, session_id=session_id,
                           prompt_version=PROMPT_VERSION, latency_ms=int((time.perf_counter() - t0) * 1000))

    def classify_incident(self, text: str) -> IncidentClassification:
        """Structured output: the model must return an object that validates against the Pydantic schema."""
        structured = self.model.with_structured_output(IncidentClassification)
        return structured.invoke([("system", INCIDENT_CLASSIFIER_PROMPT), ("human", f"Report: {text}")])
