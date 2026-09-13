from __future__ import annotations

import threading
import time
from typing import Any

_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "phase": "",
    "done": 0,
    "total": 0,
    "started_at": 0,
    "summary": None,
}
_cancelled = False


def begin() -> None:
    global _cancelled
    with _lock:
        _cancelled = False
        _state.update(
            running=True, phase="read", done=0, total=0,
            started_at=time.time(), summary=None,
        )


def is_cancelled() -> bool:
    with _lock:
        return _cancelled


def cancel() -> None:
    global _cancelled
    with _lock:
        _cancelled = True


def set_phase(phase: str, done: int = 0, total: int = 0) -> None:
    with _lock:
        _state["phase"] = phase
        _state["done"] = done
        _state["total"] = total


def update(done: int, total: int) -> None:
    with _lock:
        _state["done"] = done
        _state["total"] = total


def finish(phase: str = "done", done: int = 1, total: int = 1, summary: Any = None) -> None:
    with _lock:
        _state.update(running=False, phase=phase, done=done, total=total, summary=summary)


def snapshot() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def is_current_run(run_id: float | None, snapshot: dict[str, Any] | None = None) -> bool:
    """A cancel request only affects the run it was issued for.

    `run_id` is the import's `started_at` (epoch seconds) as seen by the client.
    A request without `run_id` or matching the current run is accepted; a stale
    run_id (e.g. from a leftover browser tab) is ignored.
    """
    if run_id is None:
        return True
    started = (snapshot or _state).get("started_at") or 0
    return abs(float(started) - float(run_id)) < 0.5
