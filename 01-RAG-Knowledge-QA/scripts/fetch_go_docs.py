#!/usr/bin/env python3
"""Fetch the Chinese translated Go documentation from golang.ac.cn.

Builds two trees under data/go:
  go-docs-text/golangacn-zh/   raw per-page text (mirrors python-docs-text)
  go-docs/NN-*.md              curated markdown used for ingestion
"""

from __future__ import annotations

import html as html_mod
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser

BASE = "https://golang.ac.cn"
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "go" / "go-docs-text" / "golangacn-zh"
DATA_KUR = ROOT / "data" / "go" / "go-docs"

HEADERS = {"User-Agent": "Mozilla/5.0 (knowledge-base builder; respect robots)"}

# (output file, 中文标题, [(path, 页面标题), ...])
GROUPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "01-教程.md",
        "Go 教程",
        [
            ("doc/tutorial/getting-started", "1. 教程：入门"),
            ("doc/tutorial/create-module", "2. 教程：创建模块"),
            ("doc/tutorial/workspaces", "3. 教程：多模块工作区入门"),
            ("doc/tutorial/web-service-gin", "4. 教程：使用 Gin 开发 RESTful API"),
            ("doc/tutorial/generics", "5. 教程：泛型入门"),
            ("doc/tutorial/fuzz", "6. 教程：模糊测试入门"),
            ("doc/tutorial/database-access", "7. 教程：访问关系数据库"),
            ("doc/code", "8. 如何编写 Go 代码"),
            ("doc/articles/wiki", "9. 编写 Web 应用程序"),
        ],
    ),
    (
        "02-语言规范.md",
        "Go 语言规范",
        [
            ("ref/spec", "1. Go 编程语言规范（go1.25）"),
            ("ref/mem", "2. Go 内存模型"),
        ],
    ),
    (
        "03-使用指南.md",
        "Go 使用指南",
        [
            ("doc/effective_go", "1. 高效 Go 编程"),
            ("doc/faq", "2. 常见问题 (FAQ)"),
            ("doc/gc-guide", "3. Go 垃圾收集器指南"),
            ("doc/diagnostics", "4. 诊断 Go 程序问题"),
            ("doc/build-cover", "5. Go 应用程序的覆盖率"),
            ("doc/pgo", "6. 配置文件引导优化 (PGO)"),
            ("doc/editors", "7. 编辑器插件和 IDE"),
            ("doc/asm", "8. Go 汇编器快速指南"),
            ("doc/gdb", "9. 使用 GDB 调试 Go 代码"),
            ("doc/comment", "10. Go Doc 注释"),
            ("doc/articles/go_command", "11. 关于 Go 命令"),
            ("doc/articles/race_detector", "12. 数据竞争检测器"),
            ("doc/database/index", "13. 数据库访问概述"),
            ("doc/database/open-handle", "14. 打开数据库句柄"),
            ("doc/database/change-data", "15. 执行不返回数据的 SQL 语句"),
            ("doc/database/querying", "16. 查询数据"),
            ("doc/database/prepared-statements", "17. 使用预处理语句"),
            ("doc/database/execute-transactions", "18. 执行事务"),
            ("doc/database/cancel-operations", "19. 取消数据库操作"),
            ("doc/database/manage-connections", "20. 管理连接"),
            ("doc/database/sql-injection", "21. 避免 SQL 注入风险"),
            ("doc/modules/managing-dependencies", "22. 管理依赖项"),
            ("doc/modules/developing", "23. 开发和发布模块"),
            ("doc/modules/release-workflow", "24. 模块发布和版本控制工作流"),
            ("doc/modules/managing-source", "25. 管理模块源"),
            ("doc/modules/layout", "26. 组织 Go 模块"),
            ("doc/modules/major-version", "27. 开发主要版本更新"),
            ("doc/modules/publishing", "28. 发布模块"),
            ("doc/modules/version-numbers", "29. 模块版本号"),
            ("doc/contribute", "30. 贡献 Go"),
        ],
    ),
    (
        "05-模块参考.md",
        "Go 模块参考",
        [
            ("ref/mod", "1. Go 模块参考"),
            ("doc/modules/gomod-ref", "2. go.mod 文件参考"),
        ],
    ),
    (
        "06-发行说明.md",
        "Go 发行说明",
        [
            ("doc/devel/release", "1. Go 发布历史"),
        ],
    ),
]

BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "caption", "col", "div",
    "dd", "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main",
    "nav", "ol", "p", "pre", "section", "td", "th", "tr", "ul",
}
SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe"}


class DocParser(HTMLParser):
    """Convert a go.dev style <article> page to clean plain text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._breadcrumb = 0
        self._toc = 0
        self._in_tail = 0
        self._in_pre = 0
        self._code_text: list[str] = []

    def _emit(self, text: str) -> None:
        if self._skip_depth or self._breadcrumb or self._toc or self._in_tail:
            return
        if self._in_pre:
            self._code_text.append(text)
            return
        self.parts.append(text)

    def _emit_block_sep(self) -> None:
        while self.parts and self.parts[-1].strip() == "":
            self.parts.pop()
        if not self.parts or self.parts[-1].endswith("\n\n"):
            return
        self.parts.append("\n\n")

    def _flush_code(self) -> None:
        if not self._code_text:
            return
        code = html_mod.unescape("".join(self._code_text)).rstrip()
        self._emit_block_sep()
        self.parts.append("\n".join("    " + ln for ln in code.split("\n")))
        self.parts.append("\n\n")
        self._code_text = []

    def handle_starttag(self, tag, attrs) -> None:  # noqa: ANN001
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "ol" and "SiteBreadcrumb" in classes:
            self._breadcrumb += 1
            return
        if tag == "div" and "TOC" in classes:
            self._toc += 1
            return
        if tag in ("aside", "div") and (classes & {"RelatedArticles", "Header", "Footer"}):
            self._in_tail += 1
            return
        if tag == "pre":
            self._in_pre += 1
            self._code_text = []
            return
        if self._breadcrumb or self._toc or self._in_tail:
            return
        if tag in BLOCK_TAGS:
            if tag != "pre":
                self._emit_block_sep()
        if tag == "li":
            self.parts.append("- ")
        elif tag in ("td", "th"):
            self.parts.append("  ")

    def handle_startendtag(self, tag, attrs) -> None:  # noqa: ANN001
        if tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag) -> None:  # noqa: ANN001
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "ol" and self._breadcrumb:
            self._breadcrumb = max(0, self._breadcrumb - 1)
            return
        if tag == "div" and self._toc:
            self._toc = max(0, self._toc - 1)
            return
        if tag in ("aside", "div") and self._in_tail:
            self._in_tail = max(0, self._in_tail - 1)
            return
        if tag == "pre":
            self._in_pre = max(0, self._in_pre - 1)
            self._flush_code()
            return
        if tag in BLOCK_TAGS and not self._in_pre:
            self._emit_block_sep()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._breadcrumb or self._toc or self._in_tail:
            return
        if self._in_pre:
            self._code_text.append(data)
            return
        self.parts.append(data)


def _extract_article(raw: str) -> str:
    start = raw.find("<main")
    if start == -1:
        return raw
    end = raw.find("</main>", start)
    if end == -1:
        end = len(raw)
    return raw[start : end + len("</main>")]


def html_to_text(raw: str) -> str:
    parser = DocParser()
    parser.feed(_extract_article(raw))
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]*\n", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch(url: str, timeout: int = 90, retries: int = 6) -> bytes:
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
                else:
                    wait = 10 + 5 * attempt
                print(f"    429, retry in {wait:.0f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as exc:
            wait = 10 + 5 * attempt
            print(f"    network error {exc}, retry in {wait:.0f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            continue
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts")


def slug_from_path(path: str) -> str:
    return path.replace("/", "_").replace(".html", "") or "index"


def main() -> None:
    pages: dict[str, dict] = {}
    for _, _, group_pages in GROUPS:
        for path, title in group_pages:
            pages.setdefault(path, {"path": path, "title": title})

    print(f"Fetching {len(pages)} pages from {BASE} ...")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_KUR.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for i, (path, info) in enumerate(pages.items(), 1):
        raw_file = DATA_RAW / f"{slug_from_path(path)}.html"
        txt_file = DATA_RAW / f"{slug_from_path(path)}.txt"
        if txt_file.exists():
            info["text"] = txt_file.read_text(encoding="utf-8")
            info["chars"] = len(info["text"])
            print(f"[{i}/{len(pages)}] cached {path}  ({info['chars']} chars)")
            continue
        try:
            raw = fetch(f"{BASE}/{path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(pages)}] ERROR {path}: {exc}")
            failures.append(path)
            continue
        text = html_to_text(raw.decode("utf-8"))
        if len(text) < 200:
            print(f"[{i}/{len(pages)}] TOO SHORT {path} ({len(text)} chars)")
            failures.append(path)
            continue
        txt_file.write_text(text, encoding="utf-8")
        if not raw_file.exists():
            raw_file.write_bytes(raw)
        info["text"] = text
        info["chars"] = len(text)
        print(f"[{i}/{len(pages)}] {path}  ({info['chars']} chars)")
        time.sleep(2)

    for outfile, title, group_pages in GROUPS:
        body: list[str] = []
        present = 0
        for path, page_title in group_pages:
            info = pages.get(path)
            if info is None or "text" not in info:
                continue
            present += 1
            body.append(f"\n\n---\n\n# {page_title}\n\n{info['text']}")
        header = [f"# {title}", "", f"共 {present} 个源文件。", "", "## 目录", ""]
        header += [
            f"- `{path}` — {pt}"
            for path, pt in group_pages
            if pages.get(path, {}).get("text")
        ]
        content = "\n".join(header) + "".join(body) + "\n"
        (DATA_KUR / outfile).write_text(content, encoding="utf-8")
        print(f"wrote {DATA_KUR / outfile}  ({len(content)} chars, {present}/{len(group_pages)} pages)")

    if failures:
        print("\nFailed pages:")
        for f in failures:
            print("  ", BASE + "/" + f)
        sys.exit(1)


if __name__ == "__main__":
    main()