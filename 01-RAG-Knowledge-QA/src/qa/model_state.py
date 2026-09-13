from __future__ import annotations

import json
import threading
from pathlib import Path

from src.config import settings

MODEL_STATE_PATH = Path("./qdrant_data/llm_model.json")

_lock = threading.Lock()
_current_model: str | None = None


def _load() -> str | None:
    try:
        data = json.loads(MODEL_STATE_PATH.read_text(encoding="utf-8"))
        model = data.get("model")
        return model if isinstance(model, str) and model else None
    except (OSError, ValueError):
        return None


def get_current_model() -> str:
    global _current_model
    if _current_model is None:
        _current_model = _load() or settings.ollama_model
    return _current_model


def set_current_model(model: str) -> str:
    global _current_model
    with _lock:
        _current_model = model
        try:
            MODEL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            MODEL_STATE_PATH.write_text(
                json.dumps({"model": model}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
        return get_current_model()
