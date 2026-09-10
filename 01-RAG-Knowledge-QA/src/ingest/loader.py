from __future__ import annotations

import pathlib
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class LoadedDocument:
    content: str
    metadata: dict


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_directory(directory: str | pathlib.Path) -> list[LoadedDocument]:
    directory = pathlib.Path(directory)
    docs: list[LoadedDocument] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_file(file_path))
    return docs


def load_file(file_path: str | pathlib.Path) -> list[LoadedDocument]:
    file_path = pathlib.Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    elif suffix in (".md", ".txt"):
        return _load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _load_text(file_path: pathlib.Path) -> list[LoadedDocument]:
    text = file_path.read_text(encoding="utf-8")
    return [
        LoadedDocument(
            content=text,
            metadata={"source": str(file_path), "filename": file_path.name},
        )
    ]


def _load_pdf(file_path: pathlib.Path) -> list[LoadedDocument]:
    doc = fitz.open(str(file_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(
                LoadedDocument(
                    content=text,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "page": i + 1,
                    },
                )
            )
    doc.close()
    return pages
