from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import get_settings
from ..ingestion import chunk_pdf_documents
from ..memory import conversation_memory
from ..models import DocumentSummary, QueryRequest, QueryResponse, UploadResponse
from ..rag import build_sources, generate_answer, rerank_chunks, rewrite_followup_question
from ..vectorstore import list_document_counts, search_chunks, upsert_chunks

app = FastAPI(title=get_settings().app_name)

FRONTEND_PATH = Path(__file__).resolve().parents[3] / "frontend" / "index.html"


@app.get("/")
def frontend() -> FileResponse:
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    settings = get_settings()
    temp_path = settings.temp_upload_dir / file.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    contents = await file.read()
    temp_path.write_bytes(contents)

    chunks = chunk_pdf_documents(temp_path)
    indexed_chunks = upsert_chunks(chunks)

    return UploadResponse(
        source_file=file.filename,
        chunks_indexed=indexed_chunks,
        pages_extracted=len({chunk.page_number for chunk in chunks}),
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    history = []
    if request.conversation_id:
        history = [(turn.question, turn.answer) for turn in conversation_memory.get_history(request.conversation_id)]

    rewritten_question = rewrite_followup_question(request.question, history)

    retrieved_chunks = search_chunks(
        question=rewritten_question,
        k=settings.retrieval_k,
        source_files=request.source_files,
    )
    ranked_chunks = rerank_chunks(rewritten_question, retrieved_chunks)
    answer = generate_answer(rewritten_question, ranked_chunks)
    sources = build_sources(answer, ranked_chunks)

    if request.conversation_id:
        conversation_memory.add_turn(request.conversation_id, request.question, answer.answer)

    return QueryResponse(
        answer=answer.answer,
        sources=sources,
        retrieved_chunks=ranked_chunks,
        rewritten_question=None if rewritten_question == request.question else rewritten_question,
    )


@app.get("/documents", response_model=list[DocumentSummary])
def documents() -> list[DocumentSummary]:
    counts = list_document_counts()
    return [DocumentSummary(source_file=source_file, chunks=counts.get(source_file, 0)) for source_file in sorted(counts)]
