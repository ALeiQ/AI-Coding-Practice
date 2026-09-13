from __future__ import annotations

from collections.abc import Generator

import requests as _requests

from src.config import settings
from src.retrieval.hybrid import RetrievedChunk, search

SYSTEM_PROMPT = """你是一个知识库问答助手。基于以下检索到的文档片段回答用户问题。

规则:
1. 只基于提供的文档回答，不要编造信息
2. 文档中的具体数字、数值、编号、名称、列表项必须原样引用，不得省略、不得改写、不得概括
3. 如果文档中确实没有相关信息，才回答"根据现有文档，我无法回答这个问题"
4. 引用来源: 在回答中标注 [来源: 文件名]
5. 回答简洁准确，先直接给出答案"""

OLLAMA_BASE = settings.ollama_base_url


def _build_context(chunks: list[RetrievedChunk]) -> tuple[list[str], str]:
    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk.metadata.get("filename", "unknown")
        page = chunk.metadata.get("page")
        loc = f"{filename}" + (f" (page {page})" if page else "")
        context_parts.append(f"[{i}] ({loc})\n{chunk.text}")
        sources.append(loc)
    return sources, "\n\n".join(context_parts)


def _make_messages(context: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"检索到的文档片段:\n\n{context}\n\n---\n用户问题: {question}",
        },
    ]


def _chunk_details(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "filename": c.metadata.get("filename", "unknown"),
            "text": c.text,
            "score": round(c.score, 4),
        }
        for c in chunks
    ]


def answer_question(
    question: str, top_k: int | None = None, collection_name: str | None = None
) -> dict:
    chunks = search(question, top_k=top_k or settings.top_k, collection_name=collection_name)
    if not chunks:
        return {
            "answer": "知识库中没有找到相关文档。",
            "sources": [],
            "chunks_used": 0,
            "retrieved": [],
        }

    sources, context = _build_context(chunks)

    resp = _requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": _make_messages(context, question),
            "options": {"temperature": 0},
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()

    return {
        "answer": data["message"]["content"],
        "sources": sources,
        "chunks_used": len(chunks),
        "retrieved": _chunk_details(chunks),
    }


def answer_question_stream(
    question: str, top_k: int | None = None, collection_name: str | None = None
) -> Generator[dict, None, None]:
    chunks = search(question, top_k=top_k or settings.top_k, collection_name=collection_name)
    if not chunks:
        yield {
            "type": "done",
            "answer": "知识库中没有找到相关文档。",
            "sources": [],
            "chunks_used": 0,
            "retrieved": [],
        }
        return

    sources, context = _build_context(chunks)

    with _requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": _make_messages(context, question),
            "options": {"temperature": 0},
            "stream": True,
        },
        stream=True,
        timeout=300,
    ) as resp:
        full_answer = ""
        for line in resp.iter_lines():
            if not line:
                continue
            data = __import__("json").loads(line)
            if "message" in data and "content" in data["message"]:
                text = data["message"]["content"]
                full_answer += text
                yield {"type": "chunk", "text": text}

    yield {
        "type": "done",
        "answer": full_answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "retrieved": _chunk_details(chunks),
    }
