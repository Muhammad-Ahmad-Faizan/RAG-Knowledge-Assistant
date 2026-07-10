from rag_knowledge_assistant.models import RetrievedChunk
from rag_knowledge_assistant.rag import GeneratedAnswer, StructuredCitation, build_sources, rewrite_followup_question


def test_rewrite_followup_question_uses_recent_history_when_model_missing(monkeypatch) -> None:
    monkeypatch.setattr("rag_knowledge_assistant.rag.get_chat_model", lambda: None)

    rewritten = rewrite_followup_question(
        "What about pricing?",
        [("What plans are available?", "The docs mention starter and enterprise plans.")],
    )

    assert rewritten == "What about pricing? about What plans are available?"


def test_build_sources_uses_structured_citations() -> None:
    chunks = [
        RetrievedChunk(
            text="Pricing is listed on page two.",
            page_number=2,
            source_file="pricing.pdf",
            chunk_id="chunk-1",
        )
    ]
    answer = GeneratedAnswer(
        answer="Pricing is on page two.",
        citations=[StructuredCitation(file="pricing.pdf", page=2)],
    )

    sources = build_sources(answer, chunks)

    assert len(sources) == 1
    assert sources[0].file == "pricing.pdf"
    assert sources[0].page == 2
    assert sources[0].snippet == "Pricing is listed on page two."