from __future__ import annotations

from dataclasses import dataclass

import ollama

from src.config import settings


@dataclass
class SparseEmbeddingResult:
    indices: list[int]
    values: list[float]


class OllamaDenseEmbeddings:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.dense_embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = ollama.embed(model=self.model_name, input=texts)
        return result["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        result = ollama.embed(model=self.model_name, input=[text])
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

    def embed_documents(self, texts: list[str]) -> list[SparseEmbeddingResult]:
        model = self._get_model()
        return [
            SparseEmbeddingResult(indices=s.indices.tolist(), values=s.values.tolist())
            for s in model.embed(texts)
        ]

    def embed_query(self, text: str) -> SparseEmbeddingResult:
        model = self._get_model()
        result = model.embed([text])[0]
        return SparseEmbeddingResult(
            indices=result.indices.tolist(), values=result.values.tolist()
        )


def get_dense_embeddings() -> OllamaDenseEmbeddings:
    return OllamaDenseEmbeddings()


def get_sparse_embeddings() -> LocalSparseEmbeddings:
    return LocalSparseEmbeddings()
