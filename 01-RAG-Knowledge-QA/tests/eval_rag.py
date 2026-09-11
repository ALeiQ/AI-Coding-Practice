#!/usr/bin/env python3
"""Evaluate RAG system against test set."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

API_BASE = "http://localhost:8000/api"
TEST_FILE = Path(__file__).parent / "rag_eval.json"


def query_rag(question: str, top_k: int = 3) -> dict:
    resp = requests.post(
        f"{API_BASE}/query",
        json={"question": question, "top_k": top_k},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    answer = ""
    sources = []
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        if event["type"] == "done":
            answer = event.get("answer", "")
            sources = event.get("sources", [])
    return {"answer": answer, "sources": sources}


def check_answer(answer: str, expected_keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in expected_keywords)


def check_source(sources: list[str], expected_source: str) -> bool:
    return any(expected_source in s for s in sources)


def main():
    with open(TEST_FILE) as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases\n")

    total = len(test_cases)
    passed = 0
    failed_cases = []

    for tc in test_cases:
        tc_id = tc["id"]
        question = tc["question"]
        expected_kw = tc["expected_keywords"]
        expected_src = tc["expected_source"]
        category = tc["category"]

        print(f"[{tc_id:2d}] {category} | {question}")

        try:
            result = query_rag(question)
            answer = result["answer"]
            sources = result["sources"]

            kw_ok = check_answer(answer, expected_kw)
            src_ok = check_source(sources, expected_src)

            if kw_ok and src_ok:
                print("     PASS")
                passed += 1
            else:
                reasons = []
                if not kw_ok:
                    reasons.append(f"keywords missing: {expected_kw}")
                if not src_ok:
                    reasons.append(f"source missing: {expected_src}")
                print(f"     FAIL ({', '.join(reasons)})")
                print(f"     answer: {answer[:100]}...")
                print(f"     sources: {sources[:5]}")
                failed_cases.append(tc)
        except Exception as e:
            print(f"     ERROR: {e}")
            failed_cases.append(tc)

        time.sleep(0.1)

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed ({passed/total*100:.1f}%)")

    by_category: dict[str, dict] = {}
    for tc in test_cases:
        cat = tc["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0}
        by_category[cat]["total"] += 1
        if tc not in failed_cases:
            by_category[cat]["passed"] += 1

    print("\nBy category:")
    for cat, stats in by_category.items():
        pct = stats["passed"] / stats["total"] * 100
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({pct:.1f}%)")

    if failed_cases:
        print(f"\nFailed cases ({len(failed_cases)}):")
        for tc in failed_cases:
            print(f"  [{tc['id']}] {tc['question']}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
