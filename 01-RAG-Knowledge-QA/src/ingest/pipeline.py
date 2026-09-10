from __future__ import annotations

import uuid

from src.ingest.chunker import split_documents
from src.ingest.loader import load_directory, load_file
from src.vectorstore.embedder import get_dense_embeddings
from src.vectorstore.store import add_documents, ensure_collection, get_client


def ingest_path(path: str, recreate: bool = False) -> dict:
    import pathlib

    p = pathlib.Path(path)
    if p.is_dir():
        docs = load_directory(p)
    elif p.is_file():
        docs = load_file(p)
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    if not docs:
        return {"error": "No supported documents found", "chunks": 0}

    chunks = split_documents(docs)
    if not chunks:
        return {"error": "No chunks produced", "chunks": 0}

    dense_embeddings = get_dense_embeddings()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    dense_vecs = dense_embeddings.embed_documents(texts)

    client = get_client()
    ensure_collection(client, recreate=recreate)
    add_documents(client, texts, metadatas, ids, dense_vecs)

    return {"chunks": len(chunks), "documents": len(docs)}
