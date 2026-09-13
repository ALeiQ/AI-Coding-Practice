from typing import Callable

import src.vectorstore.embedder as embedder_mod
from src.ingest import progress


def _make_matcher(batches: list[tuple[int, int]]) -> tuple[Callable[[int], None], list[int]]:
    captured: list[int] = []

    def matcher(n: int) -> None:
        captured.append(n)

    return matcher, captured


def test_embed_documents_progress_is_cumulative(monkeypatch):
    calls: list[int] = []

    def fake_post(endpoint, payload):
        calls.append(len(payload["input"]))
        return {"embeddings": [[0.0] * 4 for _ in payload["input"]]}

    monkeypatch.setattr(embedder_mod, "_ollama_post", fake_post)
    emb = embedder_mod.OllamaDenseEmbeddings("bge-m3")
    texts = [f"t{i} " * 60 for i in range(200)]  # batches of 96 + 96 + 8
    seen = []
    emb.embed_documents(texts, progress=seen.append)
    assert seen == [96, 192, 200]


def test_embed_documents_progress_advances_snapshot():
    embedder_mod._ollama_post = lambda endpoint, payload: {
        "embeddings": [[0.0] * 4 for _ in payload["input"]]
    }
    emb = embedder_mod.OllamaDenseEmbeddings("bge-m3")
    progress.begin()
    texts = [f"t{i} " * 60 for i in range(200)]

    def cb(n):
        progress.set_phase("embed", n, len(texts) * 2)

    emb.embed_documents(texts, progress=cb)
    snap = progress.snapshot()
    progress.finish("done", 0, 0)
    assert snap["phase"] == "embed"
    assert snap["done"] == len(texts)
    assert snap["total"] == len(texts) * 2
