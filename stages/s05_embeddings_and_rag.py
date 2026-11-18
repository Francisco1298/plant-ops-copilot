"""
STAGE 5 - Embeddings, vector database (Chroma) and RAG.
Parts 1-3 run offline with EMBEDDINGS_PROVIDER=offline; part 4 needs an LLM.
Run: python -m stages.s05_embeddings_and_rag
"""
import math

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app import rag
from app.config import get_settings
from app.llm import get_chat_model, get_embeddings

s = get_settings()
emb = get_embeddings()


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# 1) An embedding turns text into a vector; similar meaning -> similar vectors
q = emb.embed_query("digester temperature is too low")
for text in ["the digester temperature dropped below 35 C", "H2S detector alarm near the gas holder"]:
    print(f"similarity {cosine(q, emb.embed_query(text)):.3f}  <- '{text}'")
print("vector length:", len(q))

# 2) Chunking: documents are split so each piece fits in the prompt and stays on one topic
chunks = rag.split_documents(rag.load_manuals())
print(f"\n{len(chunks)} chunks. Example metadata:", chunks[0].metadata)

# 3) Index in the vector DB and search
rag.build_index()
for doc, score in rag.search("what to do if the boiler trips", k=2):
    print(f"\n[{doc.metadata['source']} | {doc.metadata.get('section')}] score={score}\n{doc.page_content[:200]}...")

# 4) RAG chain: retrieve -> put the chunks in the prompt -> generate a grounded answer with sources
if s.llm_provider != "offline":
    retriever = rag.get_vector_store().as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(

        "Answer ONLY with the context. Cite the file name in brackets exactly as it appears "
        "in the context, e.g. [sop_digester_temperature.md]. If the answer is not in the context, "
        "reply only \"I don't know\" and do not cite any source.\n\n"        
        "Context:\n{context}\n\nQuestion: {question}")
    fmt = lambda docs: "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)
    rag_chain = {"context": retriever | fmt, "question": RunnablePassthrough()} | prompt | get_chat_model() | StrOutputParser()
    for question in ["The boiler tripped and the digester is at 34 C. What do I do?",
                     "What is the H2S exposure limit for 8 hours?",
                     "Who won the football match yesterday?"]:          # out of scope -> must say "I don't know"
        print(f"\nQ: {question}\nA: {rag_chain.invoke(question)}")
