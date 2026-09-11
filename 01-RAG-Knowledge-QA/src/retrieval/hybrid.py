from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import settings
from src.vectorstore.embedder import get_dense_embeddings
from src.vectorstore.store import get_client


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


def _extract_keywords(query: str) -> list[str]:
    keywords = []
    for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", query):
        if len(token) >= 2:
            keywords.append(token)
    return keywords


def _build_metadata_filter(keywords: list[str]) -> dict | None:
    if not keywords:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchText

    conditions = []
    for kw in keywords[:5]:
        conditions.append(
            FieldCondition(
                key="text",
                match=MatchText(text=kw),
            )
        )
    return Filter(must=conditions)


def search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k
    client = get_client()

    dense_embeddings = get_dense_embeddings()
    query_dense = dense_embeddings.embed_query(query)

    dense_results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_dense,
        using="dense",
        with_payload=True,
        limit=top_k,
    )

    dense_items = []
    for point in dense_results.points:
        payload = point.payload or {}
        dense_items.append({
            "id": point.id,
            "text": payload.get("text", ""),
            "score": point.score,
            "metadata": payload.get("metadata", {}),
        })

    keywords = _extract_keywords(query)
    if keywords:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchText

            should_conditions = []
            for kw in keywords[:5]:
                should_conditions.append(
                    FieldCondition(
                        key="text",
                        match=MatchText(text=kw),
                    )
                )
            keyword_filter = Filter(should=should_conditions)

            keyword_results = client.query_points(
                collection_name=settings.qdrant_collection,
                query=query_dense,
                using="dense",
                query_filter=keyword_filter,
                with_payload=True,
                limit=top_k,
            )

            seen_ids = {item["id"] for item in dense_items}
            for point in keyword_results.points:
                if point.id not in seen_ids:
                    payload = point.payload or {}
                    dense_items.append({
                        "id": point.id,
                        "text": payload.get("text", ""),
                        "score": point.score * 1.1,
                        "metadata": payload.get("metadata", {}),
                    })
                    seen_ids.add(point.id)
        except Exception:
            pass

    dense_items.sort(key=lambda x: x["score"], reverse=True)

    chunks: list[RetrievedChunk] = []
    for item in dense_items[:top_k]:
        chunks.append(
            RetrievedChunk(
                text=item["text"],
                score=item["score"],
                metadata=item["metadata"],
            )
        )
    return chunks
