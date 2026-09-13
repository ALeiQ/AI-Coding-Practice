from src.ingest import progress


def _reset():
    progress.finish("done", 0, 0)
    progress.set_phase("", 0, 0)


def test_progress_lifecycle():
    _reset()
    progress.begin()
    snap = progress.snapshot()
    assert snap["running"] is True
    assert snap["phase"] == "read"
    assert snap["started_at"] > 0
    assert snap["summary"] is None
    started_at = snap["started_at"]
    progress.set_phase("embed", 0, 8)
    progress.update(4, 8)
    progress.finish("done", 1, 1, summary={"added": 4, "chunks": 8})
    snap = progress.snapshot()
    assert snap["phase"] == "done"
    assert snap["running"] is False
    assert snap["started_at"] == started_at
    assert snap["summary"] == {"added": 4, "chunks": 8}
    progress.begin()
    assert progress.snapshot()["summary"] is None
    _reset()


def test_progress_snapshot_is_copy():
    _reset()
    progress.begin()
    snap = progress.snapshot()
    snap["running"] = False
    snap["done"] = 99
    assert progress.snapshot()["running"] is True
    assert progress.snapshot()["done"] == 0
    progress.finish("done", 0, 0)


def test_progress_cancel_flag_and_reset():
    progress.begin()
    assert progress.is_cancelled() is False
    progress.cancel()
    assert progress.is_cancelled() is True
    progress.begin()
    assert progress.is_cancelled() is False
    progress.finish("cancelled", 0, 0)


def test_progress_cancel_run_scoping():
    progress.begin()
    snap = progress.snapshot()
    current = snap["started_at"]
    assert progress.is_current_run(None, snap) is True
    assert progress.is_current_run(current, snap) is True
    assert progress.is_current_run(current + 1, snap) is False
    assert progress.is_current_run(current - 60, snap) is False
    progress.finish("done", 1, 1)
