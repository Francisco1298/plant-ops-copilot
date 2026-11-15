"""
STAGE 2 - Your first LLM call (needs a provider configured in .env)

Concepts: chat messages and roles (system / human / ai), temperature, tokens and cost, provider-agnostic code.
Run: python -m stages.s02_first_llm_call
"""
from app.config import get_settings
from app.llm import get_chat_model

s = get_settings()
model = get_chat_model()
print(f"Provider: {s.llm_provider} | model: {s.llm_model}\n")

# 1) A conversation is a list of messages with roles
messages = [
    ("system", "You are a process engineer. Answer in 2 sentences maximum."),
    ("human", "Why does a biogas digester run at 37 C?"),
]
reply = model.invoke(messages)
print("ANSWER:", reply.content)

# 2) Tokens = what you pay for. Always look at usage when you design a prompt.
print("USAGE :", reply.usage_metadata)   # input_tokens, output_tokens, total_tokens

# 3) The model has NO memory: to keep a conversation, YOU send the history back
messages += [("ai", reply.content), ("human", "And what happens if it drops to 32 C?")]
print("\nFOLLOW-UP:", model.invoke(messages).content)

# 4) Streaming: print tokens as they arrive (better UX in chat apps)
print("\nSTREAM:", end=" ")
for chunk in model.stream([("human", "List 3 gases found in biogas, comma separated.")]):
    print(chunk.content, end="", flush=True)
print()
