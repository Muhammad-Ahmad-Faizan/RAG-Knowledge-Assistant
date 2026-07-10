from __future__ import annotations

import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import get_settings
from .models import ChunkRecord


def extract_pdf_pages(pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []
    source_file = pdf_path.name
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"page_number": page_index, "source_file": source_file},
            )
        )
    return documents


def chunk_pdf_documents(pdf_path: Path) -> list[ChunkRecord]:
    settings = get_settings()
    page_documents = extract_pdf_pages(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: list[ChunkRecord] = []
    for page_document in page_documents:
        split_docs = splitter.split_documents([page_document])
        for chunk_index, split_doc in enumerate(split_docs, start=1):
            source_file = split_doc.metadata["source_file"]
            page_number = int(split_doc.metadata["page_number"])
            # Qdrant local mode validates string point ids as UUID format.
            chunk_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_file}:{page_number}:{chunk_index}:{split_doc.page_content}",
                )
            )
            chunks.append(
                ChunkRecord(
                    text=split_doc.page_content,
                    page_number=page_number,
                    source_file=source_file,
                    chunk_id=chunk_id,
                )
            )
    return chunks
