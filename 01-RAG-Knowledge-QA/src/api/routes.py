from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    CollectionsResponse,
    CollectionSwitchRequest,
    ConfigResponse,
    ImportFile,
    ImportListResponse,
    ImportSession,
    ImportSessionDetail,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    StatusResponse,
)
from src.config import settings
from src.imports.store import (
    collection_session_stats,
    get_session,
    list_sessions,
    session_snapshot,
)
from src.imports.store import list_files as list_session_files
from src.ingest.loader import load_file
from src.ingest.pipeline import ingest_paths
from src.qa.chain import answer_question_stream
from src.vectorstore.store import (
    collection_info,
    get_client,
    list_collections,
    set_active_collection,
    source_stats,
)

router = APIRouter(prefix="/api")


@router.post("/query")
async def query(req: QueryRequest):
    def event_stream():
        for event in answer_question_stream(req.question, top_k=req.top_k):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    if req.paths:
        for_ingest = list(dict.fromkeys(req.paths))
    else:
        for_ingest = [req.path]
    result = ingest_paths(
        for_ingest, recreate=req.recreate, delete_missing=req.delete_missing
    )
    if "error" in result:
        return IngestResponse(status="error", documents=0, chunks=0, error=result["error"])
    return IngestResponse(
        status="ok",
        documents=result.get("documents", 0),
        chunks=result.get("chunks", 0),
        added=result.get("added", 0),
        updated=result.get("updated", 0),
        unchanged=result.get("unchanged", 0),
        deleted=result.get("deleted", 0),
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
    cwd = Path.cwd().resolve()
    disk = {}
    for f in sorted(data_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            resolved = f.resolve()
            disk[str(resolved)] = {
                "rel": str(resolved.relative_to(cwd)),
                "name": f.name,
            }

    try:
        stats = source_stats(get_client())
    except Exception:
        stats = []
    kb = {str(Path(s["source"]).resolve()): s["chunks"] for s in stats}

    items = []
    imported_count = 0
    for abs_key, info in sorted(disk.items()):
        chunks = kb.get(abs_key)
        imported = chunks is not None
        if imported:
            imported_count += 1
        items.append(
            {
                "rel": info["rel"],
                "name": info["name"],
                "imported": imported,
                "chunks": chunks or 0,
            }
        )

    missing = [
        {"filename": Path(s["source"]).name, "chunks": s["chunks"]}
        for s in stats
        if str(Path(s["source"]).resolve()) not in disk
    ]
    return {
        "root": "data",
        "items": items,
        "missing": missing,
        "summary": {
            "total": len(items),
            "imported": imported_count,
            "unimported": len(items) - imported_count,
        },
    }


@router.get("/files/content")
async def file_content(rel: str):
    cwd = Path.cwd().resolve()
    base = (cwd / "data").resolve()
    target = (cwd / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="path is outside the data directory")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    try:
        docs = load_file(target)
        content = "\n\n".join(d.content for d in docs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot read file: {e}")
    return {"rel": rel, "name": target.name, "content": content}


@router.get("/collections", response_model=CollectionsResponse)
async def collections():
    return CollectionsResponse(
        current=settings.qdrant_collection,
        collections=list_collections(get_client()),
    )


@router.post("/collections/switch", response_model=CollectionsResponse)
async def switch_collection(req: CollectionSwitchRequest):
    try:
        set_active_collection(get_client(), req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CollectionsResponse(
        current=settings.qdrant_collection,
        collections=list_collections(get_client()),
    )


@router.get("/imports", response_model=ImportListResponse)
async def list_imports(q: Optional[str] = None, limit: int = 100):
    sessions = list_sessions(q=q, limit=limit, collection=settings.qdrant_collection)
    stats = collection_session_stats(settings.qdrant_collection)

    live = None
    newest_id = max(stats) if stats else None
    if sessions and newest_id is not None and sessions[0]["id"] == newest_id:
        try:
            src = source_stats(get_client())
            live = (len(src), sum(s["chunks"] for s in src))
        except Exception:
            live = None

    out = []
    for s in sessions:
        st = stats.get(s["id"], {})
        row = {
            **s,
            "recreate": bool(s["recreate"]),
            "added": st.get("added", 0),
            "updated": st.get("updated", 0),
            "unchanged": st.get("unchanged", 0),
            "deleted": st.get("deleted", 0),
            "total_files": st.get("total_files", 0),
            "total_chunks": st.get("total_chunks", 0),
        }
        if live and row["id"] == newest_id:
            row["total_files"], row["total_chunks"] = live
        out.append(ImportSession(**row))
    return ImportListResponse(count=len(out), sessions=out)


@router.get("/imports/{session_id}", response_model=ImportSessionDetail)
async def import_detail(session_id: int):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Import session not found")
    files = [ImportFile(**f) for f in list_session_files(session_id)]
    st = collection_session_stats(session["collection"]).get(session_id, {})
    snapshot = [
        ImportFile(
            filename=Path(f["rel_path"]).name,
            rel_path=f["rel_path"],
            status="present",
            chunk_count=f["chunk_count"],
            file_size=0,
        )
        for f in session_snapshot(session_id, session["collection"])
    ]
    return ImportSessionDetail(
        **{
            **session,
            "recreate": bool(session["recreate"]),
            "added": st.get("added", 0),
            "updated": st.get("updated", 0),
            "unchanged": st.get("unchanged", 0),
            "deleted": st.get("deleted", 0),
            "total_files": st.get("total_files", 0),
            "total_chunks": st.get("total_chunks", 0),
        },
        files=files,
        snapshot_files=snapshot,
    )
