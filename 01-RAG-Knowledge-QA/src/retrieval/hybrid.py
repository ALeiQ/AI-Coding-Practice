from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.vectorstore.embedder import get_dense_embeddings
from src.vectorstore.store import get_client


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


def search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k

    dense_embeddings = get_dense_embeddings()
    query_dense = dense_embeddings.embed_query(query)

    client = get_client()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_dense,
        using="dense",
        with_payload=True,
        limit=top_k,
    )

    chunks: list[RetrievedChunk] = []
    for point in results.points:
        payload = point.payload or {}
        chunks.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                score=point.score,
                metadata=payload.get("metadata", {}),
            )
        )
    return chunks
