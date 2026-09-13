from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    CollectionRenameRequest,
    CollectionsResponse,
    CollectionSwitchRequest,
    ConfigResponse,
    ImportFile,
    ImportListResponse,
    ImportSession,
    ImportSessionDetail,
    IngestCancelRequest,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    StatusResponse,
)
from src.config import settings
from src.imports.store import (
    collection_session_stats,
    delete_collection_sessions,
    get_session,
    list_sessions,
    session_snapshot,
)
from src.imports.store import list_files as list_session_files
from src.ingest import progress as ingest_progress
from src.ingest.loader import load_file
from src.ingest.pipeline import ingest_paths
from src.qa.chain import answer_question_stream
from src.vectorstore.naming import (
    delete_alias,
    display_name,
    rename_display,
    storage_for_display,
)
from src.vectorstore.store import (
    collection_info,
    delete_collection,
    get_client,
    list_collections,
    set_active_collection,
    source_stats,
)

router = APIRouter(prefix="/api")
logger = logging.getLogger("rag")


@router.post("/query")
async def query(req: QueryRequest):
    collection = (
        storage_for_display(req.collection) if req.collection else settings.qdrant_collection
    )

    def event_stream():
        for event in answer_question_stream(
            req.question, top_k=req.top_k, collection_name=collection
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    if ingest_progress.snapshot()["running"]:
        return IngestResponse(
            status="busy",
            documents=0,
            chunks=0,
            error="已有导入正在进行中，请等待完成或先取消",
        )
    if req.paths:
        for_ingest = list(dict.fromkeys(req.paths))
    else:
        for_ingest = [req.path]
    ingest_progress.begin()
    try:
        result = await run_in_threadpool(
            ingest_paths,
            for_ingest,
            recreate=req.recreate,
            delete_missing=req.delete_missing,
            progress=ingest_progress.set_phase,
        )
    except Exception as exc:
        cancelled = ingest_progress.is_cancelled()
        ingest_progress.finish(
            "cancelled" if cancelled else "error", 0, 0,
            summary=None if cancelled else {"error": str(exc)},
        )
        if cancelled:
            return IngestResponse(status="cancelled", documents=0, chunks=0, error="cancelled")
        raise
    if "error" in result:
        cancelled = ingest_progress.is_cancelled() or result.get("error") == "cancelled"
        ingest_progress.finish(
            "cancelled" if cancelled else "error", 0, 0,
            summary={"error": result["error"]},
        )
        return IngestResponse(
            status="cancelled" if cancelled else "error",
            documents=0,
            chunks=0,
            error=result["error"],
        )
    ingest_progress.finish(
        "done", 1, 1,
        summary={
            "documents": result.get("documents", 0),
            "chunks": result.get("chunks", 0),
            "added": result.get("added", 0),
            "updated": result.get("updated", 0),
            "unchanged": result.get("unchanged", 0),
            "deleted": result.get("deleted", 0),
        },
    )
    return IngestResponse(
        status="ok",
        documents=result.get("documents", 0),
        chunks=result.get("chunks", 0),
        added=result.get("added", 0),
        updated=result.get("updated", 0),
        unchanged=result.get("unchanged", 0),
        deleted=result.get("deleted", 0),
    )


@router.get("/ingest/progress")
async def ingest_progress_endpoint():
    return ingest_progress.snapshot()


@router.post("/ingest/cancel")
async def ingest_cancel(req: Optional[IngestCancelRequest] = None):
    snap = ingest_progress.snapshot()
    logger.info(
        "cancel requested: run_id=%s current_started_at=%s running=%s",
        None if req is None else req.run_id,
        snap.get("started_at"),
        snap.get("running"),
    )
    if snap.get("running") and not ingest_progress.is_current_run(
        None if req is None else req.run_id, snap
    ):
        logger.warning("cancel ignored for stale run_id=%s", None if req is None else req.run_id)
        return {**snap, "ignored_stale_cancel": True}
    ingest_progress.cancel()
    return snap


@router.get("/status", response_model=StatusResponse)
async def status():
    client = get_client()
    info = collection_info(client)
    if info is None:
        return StatusResponse(
            collection=display_name(settings.qdrant_collection),
            points_count=None,
            status="not_found",
        )
    return StatusResponse(
        collection=display_name(info["name"]),
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
        current=display_name(settings.qdrant_collection),
        collections=[display_name(c) for c in list_collections(get_client())],
    )


@router.post("/collections/switch", response_model=CollectionsResponse)
async def switch_collection(req: CollectionSwitchRequest):
    try:
        set_active_collection(get_client(), req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CollectionsResponse(
        current=display_name(settings.qdrant_collection),
        collections=[display_name(c) for c in list_collections(get_client())],
    )


@router.post("/collections/rename", response_model=CollectionsResponse)
async def rename_collection(req: CollectionRenameRequest):
    try:
        rename_display(req.name, req.new_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CollectionsResponse(
        current=display_name(settings.qdrant_collection),
        collections=[display_name(c) for c in list_collections(get_client())],
    )


@router.post("/collections/delete", response_model=CollectionsResponse)
async def remove_collection(req: CollectionSwitchRequest):
    client = get_client()
    storage = storage_for_display(req.name)
    existing = list_collections(client)
    if storage not in existing:
        raise HTTPException(status_code=404, detail="集合不存在")
    if len(existing) <= 1:
        raise HTTPException(status_code=400, detail="至少保留一个集合")
    delete_alias(storage)
    delete_collection_sessions(storage)
    delete_collection(client, storage)
    if settings.qdrant_collection == storage:
        settings.qdrant_collection = next(c for c in existing if c != storage)
    return CollectionsResponse(
        current=display_name(settings.qdrant_collection),
        collections=[display_name(c) for c in list_collections(client)],
    )


@router.get("/imports", response_model=ImportListResponse)
async def list_imports(
    q: Optional[str] = None, limit: int = 100, collection: Optional[str] = None
):
    collection = storage_for_display(collection) if collection else settings.qdrant_collection
    sessions = list_sessions(q=q, limit=limit, collection=collection)
    stats = collection_session_stats(collection)

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
