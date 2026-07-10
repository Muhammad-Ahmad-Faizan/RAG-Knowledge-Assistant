from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from .config import get_settings


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    settings = get_settings()
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
    return LocalSentenceTransformerEmbeddings(settings.local_embedding_model)
