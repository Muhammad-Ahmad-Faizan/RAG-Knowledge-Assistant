from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from collections import Counter

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from .config import get_settings
from .embeddings import get_embeddings
from .models import ChunkRecord, RetrievedChunk


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_path is not None:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.qdrant_path))
    return QdrantClient(location=":memory:")


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    settings = get_settings()
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )


def ensure_collection() -> None:
    settings = get_settings()
    client = get_qdrant_client()
    collections = {collection.name for collection in client.get_collections().collections}
    if settings.qdrant_collection not in collections:
        vector_size = len(get_embeddings().embed_query("dimension probe"))
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE),
        )


def upsert_chunks(chunks: Iterable[ChunkRecord]) -> int:
    ensure_collection()
    vector_store = get_vector_store()
    documents: list[Document] = []
    ids: list[str] = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "page_number": chunk.page_number,
                    "source_file": chunk.source_file,
                    "chunk_id": chunk.chunk_id,
                },
            )
        )
        ids.append(chunk.chunk_id)
    if not documents:
        return 0
    vector_store.add_documents(documents=documents, ids=ids)
    return len(documents)


def search_chunks(question: str, k: int, source_files: list[str] | None = None) -> list[RetrievedChunk]:
    ensure_collection()
    vector_store = get_vector_store()
    filter_conditions = None
    if source_files:
        filter_conditions = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.source_file",
                    match=qdrant_models.MatchAny(any=source_files),
                )
            ]
        )
    results = vector_store.similarity_search_with_score(question, k=k, filter=filter_conditions)
    retrieved: list[RetrievedChunk] = []
    for document, score in results:
        retrieved.append(
            RetrievedChunk(
                text=document.page_content,
                page_number=int(document.metadata["page_number"]),
                source_file=str(document.metadata["source_file"]),
                chunk_id=str(document.metadata["chunk_id"]),
                score=float(score),
            )
        )
    return retrieved


def list_source_files() -> list[str]:
    ensure_collection()
    client = get_qdrant_client()
    scroll_result = client.scroll(
        collection_name=get_settings().qdrant_collection,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    source_files = {
        str(point.payload.get("metadata", {}).get("source_file"))
        for point in scroll_result[0]
        if point.payload and point.payload.get("metadata", {}).get("source_file")
    }
    return sorted(source_files)


def list_document_counts() -> dict[str, int]:
    ensure_collection()
    client = get_qdrant_client()
    scroll_result = client.scroll(
        collection_name=get_settings().qdrant_collection,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    counts: Counter[str] = Counter()
    for point in scroll_result[0]:
        source_file = point.payload.get("metadata", {}).get("source_file") if point.payload else None
        if source_file:
            counts[str(source_file)] += 1
    return dict(counts)
