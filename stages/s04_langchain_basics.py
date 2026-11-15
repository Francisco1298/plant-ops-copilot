"""
STAGE 4 - LangChain fundamentals: LCEL chains (prompt | model | parser), batch, retries and fallbacks.
Run: python -m stages.s04_langchain_basics
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model

model = get_chat_model()

# 1) A reusable prompt template with variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain industrial concepts to {audience}. Maximum {max_words} words."),
    ("human", "Explain: {concept}"),
])

# 2) LCEL: components are composed with | (each one is a "Runnable": invoke / batch / stream)
chain = prompt | model | StrOutputParser()
print(chain.invoke({"audience": "a new operator", "max_words": 40, "concept": "methane yield"}))

# 3) batch = several inputs processed concurrently
concepts = ["SCADA", "H2S desulfurisation", "CHP engine"]
for c, out in zip(concepts, chain.batch([{"audience": "a manager", "max_words": 25, "concept": c} for c in concepts])):
    print(f"\n- {c}: {out}")

# 4) Resilience: automatic retries + a fallback chain if the main one keeps failing
robust_chain = chain.with_retry(stop_after_attempt=3).with_fallbacks(
    [ChatPromptTemplate.from_template("Say that the service is busy and to retry later.") | model | StrOutputParser()]
)
print("\nRobust:", robust_chain.invoke({"audience": "an operator", "max_words": 20, "concept": "gas flare"}))

# 5) Inspect what a chain is made of (useful when debugging)
print("\nChain graph:")
chain.get_graph().print_ascii()
