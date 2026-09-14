#!/usr/bin/env python3
"""Build the Go standard-library knowledge base.

Populates data/go/go-docs/00-标准库/ with one .txt per package:
  1. Chinese pages from polaris1119/pkgdoc (studygolang.com/pkgdoc)
  2. English gap-fill via `go doc -all <pkg>` (Go 1.18+ and untranslated packages)

Also writes data/go/go-docs/04-命令与工具.md from `go doc -all cmd/go`.
"""

from __future__ import annotations

import html as html_mod
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parent.parent
STDLIB_DIR = ROOT / "data" / "go" / "go-docs" / "00-标准库"
GODOC_TEXT_DIR = ROOT / "data" / "go" / "go-docs-text" / "golangacn-zh"

PKGDOC_URL = "https://gh-proxy.com/https://github.com/polaris1119/pkgdoc/archive/refs/heads/master.tar.gz"
PKGDOC_RAW_URL = "https://raw.githubusercontent.com/polaris1119/pkgdoc/master/pkg/{}"

HEADERS = {"User-Agent": "Mozilla/5.0 (knowledge-base builder)"}

GO_BIN = shutil.which("go") or "/opt/homebrew/bin/go"


def _go_env() -> dict:
    env = dict(os.environ)
    env.pop("GOROOT", None)
    return env


def run_go(args: list[str]) -> str:
    proc = subprocess.run(
        [GO_BIN, *args], capture_output=True, text=True, env=_go_env(), timeout=180
    )
    if proc.returncode != 0:
        raise RuntimeError(f"go {' '.join(args)} failed: {proc.stderr[:500]}")
    return proc.stdout


class PkgdocParser(HTMLParser):
    """Convert a pkgdoc .htm page to plain text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._in_container = False
        self._in_pre = 0
        self._code = []
        self._in_para_link = False

    def _block_sep(self) -> None:
        while self.parts and self.parts[-1].strip() == "":
            self.parts.pop()
        if self.parts and not self.parts[-1].endswith("\n\n"):
            self.parts.append("\n\n")

    def _flush_code(self) -> None:
        if not self._code:
            return
        code = "".join(self._code).rstrip()
        self._block_sep()
        self.parts.append("\n".join("    " + ln for ln in code.split("\n")))
        self.parts.append("\n\n")
        self._code = []

    def handle_starttag(self, tag, attrs) -> None:  # noqa: ANN001
        if tag == "div" and dict(attrs).get("class") == "container":
            self._in_container = True
            return
        if not self._in_container:
            return
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        if tag == "pre":
            self._in_pre += 1
            self._code = []
            return
        if tag in {"h2", "h3", "h4", "h5"}:
            self._block_sep()
        elif tag in {"tr", "p", "ul", "ol", "dl"}:
            self._block_sep()
        elif tag == "br":
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")
        elif tag == "li":
            self.parts.append("- ")
        if tag == "h2":
            self.parts.append("# ")
        elif tag == "h3":
            self.parts.append("## ")
        elif tag in {"h4", "h5"}:
            self.parts.append("### ")

    def handle_endtag(self, tag) -> None:  # noqa: ANN001
        if not self._in_container:
            return
        if tag == "div" and self._in_container:
            self._in_container = False
            return
        if tag == "pre":
            self._in_pre = max(0, self._in_pre - 1)
            self._flush_code()
            return
        if tag in {"h2", "h3", "h4", "h5", "tr", "p", "ul", "ol", "dl"}:
            self._block_sep()

    def handle_data(self, data: str) -> None:
        if not self._in_container:
            return
        if self._in_pre:
            self._code.append(data)
            return
        if data.strip() == "\u00b6":  # pilcrow permalink
            return
        self.parts.append(data)


def pkgdoc_to_text(raw: str) -> str:
    parser = PkgdocParser()
    parser.feed(raw)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]*\n", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    for sep in ("Go语言中文网", "Go Language", "Back to top"):
        idx = text.rfind(sep)
        if idx > len(text) * 0.5:
            text = text[:idx].rstrip()
    return text.strip()


def fetch_raw(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def get_pkgdoc_dir() -> pathlib.Path:
    for cand in (
        pathlib.Path("/tmp/pkgdoc-master"),
        pathlib.Path(tempfile.gettempdir()) / "pkgdoc-master",
    ):
        if (cand / "pkg").is_dir():
            return cand
    print("Downloading pkgdoc archive (gh-proxy mirror)...")
    data = fetch_raw(PKGDOC_URL)
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="pkgdoc_"))
    tarpath = tmpdir / "pkgdoc.tar.gz"
    tarpath.write_bytes(data)
    with tarfile.open(tarpath) as tf:
        tf.extractall(tmpdir)  # noqa: S202
    for p in tmpdir.iterdir():
        if (p / "pkg").is_dir():
            cached = pathlib.Path(tempfile.gettempdir()) / "pkgdoc-master"
            shutil.rmtree(cached, ignore_errors=True)
            shutil.move(str(p), str(cached))
            return cached
    raise RuntimeError("pkgdoc archive layout unexpected")


def doc_import_path(raw: str) -> str | None:
    m = re.search(r'<p><code>import "([^"]+)"', raw)
    return m.group(1) if m else None


def is_chinese_page(raw: bytes) -> bool:
    body = re.sub(r"<[^>]+>", "", raw.decode("utf-8", errors="ignore"))
    return len(re.findall(r"[\u4e00-\u9fff]", body)) > 20


def std_packages() -> list[str]:
    out = run_go(["list", "std"])
    pkgs = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("internal/", "vendor/")):
            continue
        pkgs.append(line)
    return sorted(pkgs)


def build_stdlib() -> None:
    STDLIB_DIR.mkdir(parents=True, exist_ok=True)
    pkgdoc = get_pkgdoc_dir()
    pkg_dir = pkgdoc / "pkg"

    chinese_covered: dict[str, str] = {}
    for htm in sorted(pkg_dir.glob("*.htm")):
        raw = htm.read_bytes()
        imp = doc_import_path(raw.decode("utf-8", errors="ignore"))
        if not imp or not is_chinese_page(raw):
            continue
        text = pkgdoc_to_text(raw.decode("utf-8", errors="ignore"))
        if len(text) < 100:
            continue
        out = STDLIB_DIR / f"{imp.replace('/', '_')}.txt"
        header = (
            f"{imp} (包文档中文翻译)\n"
            f"来源：studygolang.com/pkgdoc (polaris1119/pkgdoc)\n"
            f"{'=' * 60}\n\n"
        )
        out.write_text(header + text, encoding="utf-8")
        chinese_covered[imp] = text

    print(f"pkgdoc 中文包：{len(chinese_covered)}")

    std = std_packages()
    missing = [p for p in std if p not in chinese_covered]
    print(f"标准库包：{len(std)}，go doc 补齐缺口：{len(missing)}")

    failed: list[str] = []
    go_version = run_go(["version"]).strip().split()[2]
    for p in missing:
        out = STDLIB_DIR / f"{p.replace('/', '_')}.txt"
        if out.exists():
            continue
        try:
            text = run_go(["doc", "-all", p])
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {p}: {exc}")
            failed.append(p)
            continue
        if not text.strip():
            failed.append(p)
            continue
        header = (
            f"{p} (包文档英文原文)\n"
            f"来源：go doc -all {p} (Go {go_version})\n"
            f"{'=' * 60}\n\n"
        )
        out.write_text(header + text, encoding="utf-8")

    print(f"go doc 失败：{len(failed)}  ({', '.join(failed) or 'none'})")


def build_command_docs() -> None:
    text = run_go(["doc", "-all", "cmd/go"])
    header = (
        "# Go 命令参考（英文原文）\n\n"
        "来源：go doc -all cmd/go (Go 官方工具链本地导出)；golang.ac.cn 无中文版。\n\n---\n\n"
    )
    out = ROOT / "data" / "go" / "go-docs" / "04-命令与工具.md"
    out.write_text(header + text, encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    build_stdlib()
    build_command_docs()
    total = len(list(STDLIB_DIR.glob("*.txt")))
    print(f"00-标准库 共 {total} 个包文档")


if __name__ == "__main__":
    main()