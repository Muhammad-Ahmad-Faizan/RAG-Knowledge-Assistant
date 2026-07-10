from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from pydantic import BaseModel, Field

from .config import get_settings
from .models import CitationRecord, RetrievedChunk


class StructuredCitation(BaseModel):
    file: str
    page: int


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[StructuredCitation] = Field(default_factory=list)


class RewrittenQuestion(BaseModel):
    question: str


@lru_cache(maxsize=1)
def get_chat_model():
    settings = get_settings()
    if not settings.openai_api_key:
        return None

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.openai_chat_model, api_key=settings.openai_api_key, temperature=0)


def build_context_snippets(chunks: Iterable[RetrievedChunk]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {index} | {chunk.source_file} | page {chunk.page_number}]\n{chunk.text.strip()}"
        )
    return "\n\n".join(parts)


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context_snippets(chunks)
    return (
        "You are a document question-answering assistant. "
        "Answer only from the provided context. "
        "Cite page numbers inline using the format (source_file, p. page_number). "
        "If the context is insufficient, say you cannot find the answer in the provided documents.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def build_rewrite_prompt(question: str, history: list[tuple[str, str]]) -> str:
    history_text = "\n".join(f"Q: {turn_question}\nA: {turn_answer}" for turn_question, turn_answer in history)
    return (
        "Rewrite the user follow-up into a standalone question for document retrieval. "
        "Use the conversation context only when needed. Return JSON with a single key question.\n\n"
        f"Conversation:\n{history_text}\n\nFollow-up question: {question}"
    )


CITATION_PATTERN = re.compile(r"\((?P<file>[^,()]+),\s*p\.\s*(?P<page>\d+)\)")


def parse_citations(answer: str, chunks: list[RetrievedChunk]) -> list[CitationRecord]:
    citations: list[CitationRecord] = []
    seen: set[tuple[str, int]] = set()
    for match in CITATION_PATTERN.finditer(answer):
        file_name = match.group("file").strip()
        page = int(match.group("page"))
        key = (file_name, page)
        if key in seen:
            continue
        seen.add(key)
        snippet = next(
            (
                chunk.text[:240].strip()
                for chunk in chunks
                if chunk.source_file == file_name and chunk.page_number == page
            ),
            "",
        )
        citations.append(CitationRecord(file=file_name, page=page, snippet=snippet))
    return citations


def rewrite_followup_question(question: str, history: list[tuple[str, str]]) -> str:
    if not history:
        return question

    chat_model = get_chat_model()
    if chat_model is not None:
        try:
            structured_model = chat_model.with_structured_output(RewrittenQuestion)
            result = structured_model.invoke(build_rewrite_prompt(question, history[-3:]))
            rewritten = result.question.strip()
            if rewritten:
                return rewritten
        except Exception:
            pass

    return f"{question} about {history[-1][0]}"


def rerank_chunks(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    settings = get_settings()
    if not settings.rerank_enabled or len(chunks) < 2:
        return chunks

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(settings.reranker_model)
        scores = model.predict([(question, chunk.text) for chunk in chunks])
        ranked_pairs = sorted(zip(chunks, scores, strict=False), key=lambda item: float(item[1]), reverse=True)
        return [chunk for chunk, _, _ in ranked_pairs]
    except Exception:
        question_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]+", question) if len(term) > 2}

        def overlap_score(chunk: RetrievedChunk) -> int:
            chunk_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]+", chunk.text)}
            return len(question_terms & chunk_terms)

        ranked_chunks = sorted(chunks, key=overlap_score, reverse=True)

    if settings.rerank_top_n > 0:
        return ranked_chunks[: settings.rerank_top_n]
    return ranked_chunks


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
    if not chunks:
        return GeneratedAnswer(answer="I cannot find the answer in the provided documents.")

    settings = get_settings()
    context_chunks = chunks[: settings.max_context_chunks]
    chat_model = get_chat_model()
    prompt = build_rag_prompt(question, context_chunks)

    if chat_model is not None:
        try:
            structured_model = chat_model.with_structured_output(GeneratedAnswer)
            result = structured_model.invoke(prompt)
            if result.answer.strip():
                return result
        except Exception:
            pass

    cited_parts = []
    citations: list[StructuredCitation] = []
    for chunk in context_chunks[:3]:
        snippet = re.sub(r"\s+", " ", chunk.text).strip()
        snippet = snippet[:180].rstrip()
        cited_parts.append(f"- {snippet} ({chunk.source_file}, p. {chunk.page_number})")
        citations.append(StructuredCitation(file=chunk.source_file, page=chunk.page_number))

    answer_text = "Here are the most relevant points from your documents:\n" + "\n".join(cited_parts)
    return GeneratedAnswer(answer=answer_text, citations=citations)


def build_sources(answer: GeneratedAnswer, chunks: list[RetrievedChunk]) -> list[CitationRecord]:
    context_chunks = chunks[: get_settings().max_context_chunks]
    if answer.citations:
        records: list[CitationRecord] = []
        for citation in answer.citations:
            snippet = next(
                (
                    chunk.text[:240].strip()
                    for chunk in context_chunks
                    if chunk.source_file == citation.file and chunk.page_number == citation.page
                ),
                "",
            )
            records.append(CitationRecord(file=citation.file, page=citation.page, snippet=snippet))
        return records

    return parse_citations(answer.answer, context_chunks)
