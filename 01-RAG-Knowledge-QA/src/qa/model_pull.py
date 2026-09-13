from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Optional

from src.config import settings

PULL_STATE_PATH = Path("./qdrant_data/llm_pull.json")

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:-]+$")

_lock = threading.Lock()
_active_pull = None  # dict with model + thread state
_state: dict = {}


def _safe_state() -> dict:
    return {
        "model": _state.get("model"),
        "phase": _state.get("phase", "idle"),
        "completed": _state.get("completed", 0),
        "total": _state.get("total", 0),
        "pct": _state.get("pct", 0),
        "error": _state.get("error"),
        "active": _state.get("active", False),
        "done": _state.get("done", False),
    }


def _persist() -> None:
    try:
        PULL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PULL_STATE_PATH.write_text(
            json.dumps(_safe_state(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _update(**kw) -> None:
    with _lock:
        _state.update(kw)
    _persist()


def _load_saved() -> None:
    global _state
    try:
        _state = json.loads(PULL_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _state = {}


def get_pull_progress() -> dict:
    with _lock:
        return dict(_safe_state())


def is_pull_active() -> bool:
    return bool(_active_pull)


def validate_model_name(name: str) -> Optional[str]:
    name = (name or "").strip()
    if not name:
        return "模型名不能为空"
    if not _MODEL_NAME_RE.match(name):
        return "模型名格式非法（仅允许字母数字 . _ : -）"
    return None


def _pull_worker(model: str) -> None:
    url = f"{settings.ollama_base_url}/api/pull"
    proc = None
    try:
        proc = subprocess.Popen(
            [
                "curl", "-sN",
                "-X", "POST", url,
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"model": model, "stream": True}),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout or []:  # type: ignore[union-attr]
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            status = msg.get("status", "")
            if status.startswith("pulling manifest"):
                _update(phase="manifest", completed=0, total=0, pct=0)
            elif status.startswith("downloading "):
                completed = msg.get("completed", 0) or 0
                total = msg.get("total", 0) or 0
                pct = round(completed * 100.0 / total, 2) if total else 0
                _update(
                    model=model,
                    phase="downloading",
                    completed=completed,
                    total=total,
                    pct=pct,
                )
            elif status in ("verifying sha256 digest", "writing manifest", "creating model"):
                _update(phase=status)
            elif status == "success":
                _update(
                    model=model,
                    phase="done",
                    completed=0,
                    total=0,
                    pct=100,
                    done=True,
                    error=None,
                )
            if msg.get("error"):
                err = str(msg["error"])
                _update(model=model, phase="error", error=err, done=True)
                break
        proc.wait(timeout=5)
    except Exception as exc:  # noqa: BLE001
        _update(model=model, phase="error", error=str(exc), done=True)
    finally:
        if proc is not None:
            try:
                proc.stdout.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        _update(active=False)
        with _lock:
            global _active_pull
            _active_pull = None
        _persist()


def start_pull(model: str) -> str:
    global _active_pull
    with _lock:
        if _active_pull:
            return "busy"
    _update(
        model=model,
        phase="starting",
        completed=0,
        total=0,
        pct=0,
        error=None,
        active=True,
        done=False,
    )
    t = threading.Thread(target=_pull_worker, args=(model,), daemon=True)
    with _lock:
        _active_pull = t
    t.start()
    return "started"


_load_saved()
