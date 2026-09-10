from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 20


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


class IngestRequest(BaseModel):
    path: str = "data"
    recreate: bool = False


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int


class IngestErrorResponse(BaseModel):
    status: str
    error: str


class StatusResponse(BaseModel):
    collection: str
    points_count: Optional[int] = None
    status: str


class ConfigResponse(BaseModel):
    embedding_model: str
    llm_model: str
    llm_base_url: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    qdrant_url: str
