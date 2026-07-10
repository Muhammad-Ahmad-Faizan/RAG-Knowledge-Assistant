from pathlib import Path

from rag_knowledge_assistant.ingestion import chunk_pdf_documents


def test_chunk_pdf_documents_handles_empty_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

    chunks = chunk_pdf_documents(pdf_path)

    assert chunks == []
