from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.config import settings


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default_factory=lambda: settings.top_k)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


class IngestRequest(BaseModel):
    path: str = "data"
    paths: Optional[list[str]] = None
    recreate: bool = False
    delete_missing: bool = True


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    error: Optional[str] = None


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


class ImportFile(BaseModel):
    filename: str
    rel_path: str
    status: str
    chunk_count: int
    file_size: int
    file_md5: Optional[str] = None


class ImportSession(BaseModel):
    id: int
    ts: str
    path: str
    recreate: bool
    documents: int
    chunks: int
    status: str
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    total_files: int = 0
    total_chunks: int = 0


class ImportSessionDetail(ImportSession):
    files: list[ImportFile]
    snapshot_files: list[ImportFile] = []


class ImportListResponse(BaseModel):
    count: int
    sessions: list[ImportSession]


class CollectionSwitchRequest(BaseModel):
    name: str


class CollectionsResponse(BaseModel):
    current: str
    collections: list[str]
