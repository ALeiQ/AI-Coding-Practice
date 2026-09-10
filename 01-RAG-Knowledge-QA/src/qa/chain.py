from __future__ import annotations

import ollama

from src.config import settings
from src.retrieval.hybrid import search

SYSTEM_PROMPT = """你是一个知识库问答助手。基于以下检索到的文档片段回答用户问题。

规则:
1. 只基于提供的文档回答，不要编造信息
2. 如果文档中没有相关信息，回答"根据现有文档，我无法回答这个问题"
3. 引用来源: 在回答中标注 [来源: 文件名]
4. 保持回答简洁准确"""


def answer_question(question: str) -> dict:
    chunks = search(question, top_k=settings.top_k)
    if not chunks:
        return {
            "answer": "知识库中没有找到相关文档。",
            "sources": [],
        }

    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk.metadata.get("filename", "unknown")
        page = chunk.metadata.get("page")
        loc = f"{filename}" + (f" (page {page})" if page else "")
        context_parts.append(f"[{i}] ({loc})\n{chunk.text}")
        sources.append(loc)

    context = "\n\n".join(context_parts)

    client = ollama.Client(host=settings.ollama_base_url)
    response = client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"检索到的文档片段:\n\n{context}\n\n---\n用户问题: {question}",
            },
        ],
        options={"temperature": 0},
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources,
        "chunks_used": len(chunks),
    }
