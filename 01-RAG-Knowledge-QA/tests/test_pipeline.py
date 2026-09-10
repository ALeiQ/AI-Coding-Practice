from src.ingest.chunker import split_documents
from src.ingest.loader import LoadedDocument


def test_split_documents_produces_chunks():
    docs = [
        LoadedDocument(
            content="This is a test document. " * 50,
            metadata={"source": "test.txt", "filename": "test.txt"},
        )
    ]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 0
    assert all("text" in c and "metadata" in c for c in chunks)


def test_split_documents_respects_metadata():
    docs = [
        LoadedDocument(
            content="Hello world. " * 30,
            metadata={"source": "a.md", "filename": "a.md"},
        )
    ]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
    for chunk in chunks:
        assert chunk["metadata"]["source"] == "a.md"
        assert chunk["metadata"]["filename"] == "a.md"
        assert "chunk_index" in chunk["metadata"]
