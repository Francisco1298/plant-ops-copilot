"""
RAG (Retrieval-Augmented Generation) building blocks:
  load documents -> split into chunks -> embed -> store in a vector DB -> retrieve the most similar chunks.
"""
import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.llm import get_embeddings

log = logging.getLogger(__name__)


def load_manuals(settings: Settings | None = None) -> list[Document]:
    s = settings or get_settings()
    docs = []
    for path in sorted(s.manuals_dir.glob("*.md")):
        docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name}))
    return docs


def split_documents(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    """Split by Markdown headers first (keeps sections together), then by size."""
    header_splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")], strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        for part in header_splitter.split_text(doc.page_content):
            part.metadata = {**doc.metadata, **part.metadata}
            chunks.extend(size_splitter.split_documents([part]))
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata['source']}#{i}"
    return chunks


def get_vector_store(settings: Settings | None = None) -> Chroma:
    """Local persistent Chroma by default; a Chroma *server* if CHROMA_HOST is set (distributed setup)."""
    s = settings or get_settings()
    if s.chroma_host:
        client = chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)
    else:
        client = chromadb.PersistentClient(path=str(s.chroma_dir))
    return Chroma(client=client, collection_name=s.collection, embedding_function=get_embeddings(s),
                  collection_metadata={"hnsw:space": "cosine"})


def build_index(settings: Settings | None = None) -> int:
    """Idempotent ingestion: chunk ids are stable, so re-running replaces instead of duplicating."""
    store = get_vector_store(settings)
    chunks = split_documents(load_manuals(settings))
    ids = [c.metadata["chunk_id"] for c in chunks]
    existing = store.get(ids=ids)["ids"]
    if existing:
        store.delete(ids=existing)
    store.add_documents(chunks, ids=ids)
    log.info("Indexed %d chunks", len(chunks))
    return len(chunks)


def search(query: str, k: int = 3, settings: Settings | None = None) -> list[tuple[Document, float]]:
    """Returns (chunk, similarity) pairs; similarity = 1 - cosine distance (higher is better)."""
    store = get_vector_store(settings)
    return [(doc, round(1 - dist, 3)) for doc, dist in store.similarity_search_with_score(query, k=k)]
