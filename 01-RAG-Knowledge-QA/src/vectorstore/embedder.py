from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import requests as _requests

from src.config import settings
from src.ingest import progress as ingest_progress

OLLAMA_BASE = settings.ollama_base_url

OLLAMA_TIMEOUT = 300
EMBED_BATCH_SIZE = 96
EMBED_WORKERS = 1


def _ollama_post(endpoint: str, payload: dict) -> dict:
    import time as _time

    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = _requests.post(
                f"{OLLAMA_BASE}{endpoint}", json=payload, timeout=OLLAMA_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()
        except (_requests.exceptions.ConnectionError, _requests.exceptions.Timeout) as e:
            last = e
            if attempt < 2:
                _time.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]


@dataclass
class SparseEmbeddingResult:
    indices: list[int]
    values: list[float]


class OllamaDenseEmbeddings:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.dense_embedding_model

    def embed_documents(
        self, texts: list[str], progress: Callable[[int], None] | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        batches = [
            texts[i : i + EMBED_BATCH_SIZE]
            for i in range(0, len(texts), EMBED_BATCH_SIZE)
        ]

        def run(batch: list[str]) -> list[list[float]]:
            result = _ollama_post(
                "/api/embed",
                {"model": self.model_name, "input": batch, "keep_alive": "30m"},
            )
            return result["embeddings"]

        if len(batches) == 1:
            if ingest_progress.is_cancelled():
                raise RuntimeError("cancelled")
            result = run(batches[0])
            if progress:
                progress(len(texts))
            return result
        results: dict[int, list[list[float]]] = {}
        done_count = 0
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
            futures = {
                executor.submit(run, batch): (i, len(batch))
                for i, batch in enumerate(batches)
            }
            for fut in as_completed(futures):
                if ingest_progress.is_cancelled():
                    raise RuntimeError("cancelled")
                i, size = futures[fut]
                results[i] = fut.result()
                done_count += size
                if progress:
                    progress(done_count)
        return [vector for i in range(len(batches)) for vector in results[i]]

    def embed_query(self, text: str) -> list[float]:
        result = _ollama_post(
            "/api/embed",
            {"model": self.model_name, "input": [text], "keep_alive": "30m"},
        )
        return result["embeddings"][0]


class LocalSparseEmbeddings:
    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(
        self, texts: list[str], progress: Callable[[int], None] | None = None
    ) -> list[SparseEmbeddingResult]:
        model = self._get_model()
        results = []
        for s in model.embed(texts):
            if ingest_progress.is_cancelled():
                raise RuntimeError("cancelled")
            results.append(
                SparseEmbeddingResult(indices=s.indices.tolist(), values=s.values.tolist())
            )
            if progress:
                progress(1)
        return results

    def embed_query(self, text: str) -> SparseEmbeddingResult:
        model = self._get_model()
        results = list(model.embed([text]))
        result = results[0]
        return SparseEmbeddingResult(
            indices=result.indices.tolist(), values=result.values.tolist()
        )


def get_dense_embeddings() -> OllamaDenseEmbeddings:
    return OllamaDenseEmbeddings()


def get_sparse_embeddings() -> LocalSparseEmbeddings:
    return LocalSparseEmbeddings()
