from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import settings

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = settings.qdrant_url
        if url == ":memory:" or url == "":
            _client = QdrantClient(":memory:")
        elif url.startswith("http"):
            _client = QdrantClient(url=url)
        else:
            _client = QdrantClient(path=url)
    return _client


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    name = settings.qdrant_collection
    if client.collection_exists(name):
        if recreate:
            client.delete_collection(name)
        else:
            return

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=768, distance=Distance.COSINE),
        },
    )


def add_documents(
    client: QdrantClient,
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
    dense_vectors: list[list[float]],
) -> None:
    from qdrant_client.models import PointStruct

    points = []
    for i in range(len(texts)):
        points.append(
            PointStruct(
                id=ids[i],
                vector={
                    "dense": dense_vectors[i],
                },
                payload={
                    "text": texts[i],
                    "metadata": metadatas[i],
                },
            )
        )

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )


def collection_info(client: QdrantClient) -> dict | None:
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        return None
    info = client.get_collection(name)
    return {
        "name": name,
        "points_count": info.points_count,
        "status": info.status,
    }
