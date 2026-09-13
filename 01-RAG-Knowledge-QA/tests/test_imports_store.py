import threading

from src.imports import store


def test_store_threadsafe_across_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(store.settings, "import_db_path", str(tmp_path / "i.db"))
    store.init_db()
    results: list[tuple[int, str]] = []

    def worker(tag: str):
        sid = store.add_session(
            f"/tmp/{tag}", recreate=False, documents=1, chunks=1, status="ok"
        )
        store.add_files(sid, [{"filename": f"{tag}.md", "rel_path": f"/tmp/{tag}.md",
                               "status": "added", "chunk_count": 1, "file_size": 3,
                               "file_md5": "abc"}])
        results.append((sid, tag))

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 6
    rows = store.list_sessions(collection=store.settings.qdrant_collection)
    assert len(rows) == 6
    assert {r["path"] for r in rows} == {f"/tmp/t{i}" for i in range(6)}
