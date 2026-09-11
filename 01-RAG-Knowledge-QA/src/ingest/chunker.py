from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.ingest.loader import LoadedDocument


def split_documents(
    documents: list[LoadedDocument],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", ".", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []
    for doc in documents:
        texts = splitter.split_text(doc.content)
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        **doc.metadata,
                        "chunk_index": i,
                    },
                }
            )
    return chunks
