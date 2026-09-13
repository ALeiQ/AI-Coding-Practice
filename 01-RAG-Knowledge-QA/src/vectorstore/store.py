from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from src.config import settings
from src.vectorstore.naming import register_alias, storage_name

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


def list_collections(client: QdrantClient) -> list[str]:
    try:
        return sorted(c.name for c in client.get_collections().collections)
    except Exception:
        return []


def delete_collection(client: QdrantClient, name: str) -> None:
    client.delete_collection(name)


def set_active_collection(client: QdrantClient, name: str) -> None:
    """Switch the active collection at runtime (in-memory, resets on restart).

    Non-ASCII display names are mapped to a Qdrant-legal storage name; the
    display name persists via the name alias file.
    """
    if not name or not name.strip():
        raise ValueError("Collection name must not be empty")
    storage = storage_name(name)
    register_alias(name, storage)
    settings.qdrant_collection = storage
    ensure_collection(client, recreate=False)


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
            "dense": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": {},
        },
    )


def add_documents(
    client: QdrantClient,
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict] | None = None,
) -> None:
    from qdrant_client.models import PointStruct, SparseVector

    points = []
    for i in range(len(texts)):
        vector = {"dense": dense_vectors[i]}
        if sparse_vectors and i < len(sparse_vectors):
            sv = sparse_vectors[i]
            vector["sparse"] = SparseVector(
                indices=sv["indices"],
                values=sv["values"],
            )

        points.append(
            PointStruct(
                id=ids[i],
                vector=vector,
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


def distinct_filenames(client: QdrantClient) -> list[dict]:
    """Return [{filename, chunks}] of distinct filenames actually indexed."""
    stats = source_stats(client)
    filename_ids: dict[str, int] = {}
    for item in stats:
        key = Path(item["source"]).name
        filename_ids[key] = filename_ids.get(key, 0) + item["chunks"]
    return [
        {"filename": f, "chunks": c}
        for f, c in sorted(filename_ids.items(), key=lambda x: -x[1])
    ]


def source_stats(client: QdrantClient) -> list[dict]:
    """Return [{source, filename, chunks, rel_path}] grouped by metadata.source."""
    if not client.collection_exists(settings.qdrant_collection):
        return []
    stats: dict[str, dict] = {}
    offset: dict | None = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            meta = (p.payload or {}).get("metadata", {})
            source = meta.get("source") or meta.get("rel_path") or meta.get("filename")
            if not source:
                continue
            entry = stats.setdefault(
                source,
                {
                    "source": source,
                    "filename": meta.get("filename", Path(source).name),
                    "chunks": 0,
                    "rel_path": meta.get("rel_path"),
                },
            )
            entry["chunks"] += 1
        if next_offset is None:
            break
        offset = next_offset
    return sorted(stats.values(), key=lambda x: x["source"])


def _source_filter(source: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value=source),
            )
        ]
    )


def delete_by_source(client: QdrantClient, source: str) -> int:
    """Delete all points whose metadata.source == source. Returns deleted count."""
    if not client.collection_exists(settings.qdrant_collection):
        return 0
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=_source_filter(source),
        limit=1000,
        with_vectors=True,
    )
    if not points:
        return 0
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=source),
                )
            ]
        ),
    )
    return len(points)
