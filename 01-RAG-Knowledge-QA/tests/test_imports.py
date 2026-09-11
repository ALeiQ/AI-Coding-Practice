import pytest

from src.imports import store


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "imports.db"
    monkeypatch.setattr(store.settings, "import_db_path", str(path))
    return path


def test_add_and_list_sessions(db_path):
    store.init_db()
    id1 = store.add_session(
        "data",
        recreate=False,
        documents=3,
        chunks=12,
        status="ok",
        embedding_model="bge-m3",
        chunk_size=1000,
        chunk_overlap=150,
    )
    store.add_session(
        "data/notes",
        recreate=True,
        documents=1,
        chunks=5,
        status="ok",
    )
    sessions = store.list_sessions()
    assert len(sessions) == 2
    assert sessions[0]["status"] == "ok"
    assert sessions[0]["path"].endswith("notes")
    detail = store.get_session(id1)
    assert detail is not None
    assert detail["documents"] == 3
    assert detail["embedding_model"] == "bge-m3"


def test_filter_sessions_by_path(db_path):
    store.add_session("data", recreate=False, documents=1, chunks=1, status="ok")
    store.add_session("data/备忘录", recreate=False, documents=1, chunks=1, status="ok")
    store.add_session("docs/other", recreate=False, documents=1, chunks=1, status="ok")
    sessions = store.list_sessions(q="备忘")
    assert [s["path"] for s in sessions] == ["data/备忘录"]


def test_error_session_recorded(db_path):
    sid = store.add_session(
        "data",
        recreate=True,
        documents=5,
        chunks=0,
        status="error",
        error="boom",
    )
    detail = store.get_session(sid)
    assert detail["status"] == "error"
    assert detail["error"] == "boom"
    assert detail["recreate"] == 1


def test_add_files_and_list(db_path):
    sid = store.add_session("data", recreate=False, documents=2, chunks=3, status="ok")
    store.add_files(
        sid,
        [
            {"filename": "a.md", "rel_path": "data/a.md", "status": "added",
             "chunk_count": 2, "file_size": 100, "file_md5": "abc"},
            {"filename": "b.md", "rel_path": "data/b.md", "status": "unchanged",
             "chunk_count": 1, "file_size": 50, "file_md5": "def"},
        ],
    )
    files = store.list_files(sid)
    assert len(files) == 2
    assert files[0]["status"] == "added"
    assert files[1]["file_md5"] == "def"


def test_latest_md5_by_filename(db_path):
    s1 = store.add_session("data", recreate=False, documents=1, chunks=1, status="ok")
    store.add_files(s1, [{"filename": "a.md", "rel_path": "data/a.md", "status": "added",
                          "chunk_count": 1, "file_size": 10, "file_md5": "old"}])
    s2 = store.add_session("data", recreate=False, documents=1, chunks=1, status="ok")
    store.add_files(s2, [{"filename": "a.md", "rel_path": "data/a.md", "status": "updated",
                          "chunk_count": 2, "file_size": 20, "file_md5": "new"}])
    latest = store.latest_md5_by_filename()
    assert latest["a.md"]["file_md5"] == "new"
    assert latest["a.md"]["session_id"] == s2


def test_collection_session_stats_deltas_and_snapshot(db_path):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(store.settings, "qdrant_collection", "kb")
    try:
        s1 = store.add_session("kb", recreate=True, documents=2, chunks=3, status="ok")
        store.add_files(
            s1,
            [
                {"filename": "a.md", "rel_path": "kb/a.md", "status": "added",
                 "chunk_count": 2, "file_size": 10, "file_md5": "A"},
                {"filename": "b.md", "rel_path": "kb/b.md", "status": "added",
                 "chunk_count": 1, "file_size": 10, "file_md5": "B"},
            ],
        )
        s2 = store.add_session("kb", recreate=False, documents=2, chunks=1, status="ok")
        store.add_files(
            s2,
            [
                {"filename": "a.md", "rel_path": "kb/a.md", "status": "unchanged",
                 "chunk_count": 2, "file_size": 10, "file_md5": "A"},
                {"filename": "c.md", "rel_path": "kb/c.md", "status": "added",
                 "chunk_count": 1, "file_size": 10, "file_md5": "C"},
            ],
        )
        s3 = store.add_session("kb", recreate=False, documents=0, chunks=0, status="ok")
        store.add_files(
            s3,
            [
                {"filename": "b.md", "rel_path": "kb/b.md", "status": "deleted",
                 "chunk_count": 0, "file_size": 0, "file_md5": None},
            ],
        )

        stats = store.collection_session_stats("kb")
        assert stats[s1]["added"] == 2
        assert stats[s1]["total_files"] == 2
        assert stats[s1]["total_chunks"] == 3
        assert stats[s2]["added"] == 1 and stats[s2]["unchanged"] == 1
        assert stats[s2]["total_files"] == 3
        assert stats[s2]["total_chunks"] == 4
        assert stats[s3]["unchanged"] == 0 and stats[s3]["deleted"] == 1
        assert stats[s3]["total_files"] == 2
        assert stats[s3]["total_chunks"] == 3

        snap1 = store.session_snapshot(s1, "kb")
        assert [f["rel_path"] for f in snap1] == ["kb/a.md", "kb/b.md"]
        snap2 = store.session_snapshot(s2, "kb")
        assert [f["rel_path"] for f in snap2] == ["kb/a.md", "kb/b.md", "kb/c.md"]
        snap3 = store.session_snapshot(s3, "kb")
        assert [f["rel_path"] for f in snap3] == ["kb/a.md", "kb/c.md"]
    finally:
        monkeypatch.undo()
