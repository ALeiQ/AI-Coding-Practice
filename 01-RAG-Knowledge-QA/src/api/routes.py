from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ConfigResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    StatusResponse,
)
from src.config import settings
from src.ingest.pipeline import ingest_path
from src.qa.chain import answer_question_stream
from src.vectorstore.store import collection_info, get_client

router = APIRouter(prefix="/api")


@router.post("/query")
async def query(req: QueryRequest):
    def event_stream():
        for event in answer_question_stream(req.question, top_k=req.top_k):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    result = ingest_path(req.path, recreate=req.recreate)
    if "error" in result:
        return IngestResponse(status="error", documents=0, chunks=0)
    return IngestResponse(
        status="ok",
        documents=result.get("documents", 0),
        chunks=result.get("chunks", 0),
    )


@router.get("/status", response_model=StatusResponse)
async def status():
    client = get_client()
    info = collection_info(client)
    if info is None:
        return StatusResponse(
            collection=settings.qdrant_collection,
            points_count=None,
            status="not_found",
        )
    return StatusResponse(
        collection=info["name"],
        points_count=info["points_count"],
        status=str(info["status"]),
    )


@router.get("/config", response_model=ConfigResponse)
async def config():
    return ConfigResponse(
        embedding_model=settings.dense_embedding_model,
        llm_model=settings.ollama_model,
        llm_base_url=settings.ollama_base_url,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        qdrant_url=settings.qdrant_url,
    )


@router.get("/files")
async def list_files():
    data_dir = Path("data")
    files = []
    for f in sorted(data_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            files.append(str(f.relative_to(data_dir)))
    return {"files": files, "count": len(files)}
