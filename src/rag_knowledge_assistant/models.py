from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    text: str
    page_number: int
    source_file: str
    chunk_id: str


class CitationRecord(BaseModel):
    file: str
    page: int
    snippet: str


class UploadResponse(BaseModel):
    source_file: str
    chunks_indexed: int
    pages_extracted: int


class QueryRequest(BaseModel):
    question: str
    source_files: list[str] | None = None
    conversation_id: str | None = None


class RetrievedChunk(BaseModel):
    text: str
    page_number: int
    source_file: str
    chunk_id: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[CitationRecord]
    retrieved_chunks: list[RetrievedChunk]
    rewritten_question: str | None = None


class DocumentSummary(BaseModel):
    source_file: str
    chunks: int
    uploaded_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
