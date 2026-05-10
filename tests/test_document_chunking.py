"""
Tests for sliding-window chunking in the document pipeline.
Written BEFORE implementation (TDD).

Covers three layers:
  1. chunk_text()               — pure function in core.document_loader
  2. DocumentLoader.chunk_documents() — produces DocumentSchema objects
  3. VectorStoreManager.add_documents_chunked() — end-to-end with mocked backends

VectorStoreManager tests patch chromadb.PersistentClient and
sentence_transformers.SentenceTransformer so no real backends are needed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

from core.document_loader import DocumentLoader, chunk_text
from core.schemas import DocumentSchema
from core.vector_store_manager import VectorStoreManager


# ── Helpers ───────────────────────────────────────────────────────────────────


def _doc(content: str, ticker="AAPL", filepath="test.csv") -> DocumentSchema:
    return DocumentSchema(
        content=content,
        source="bloomberg_export",
        ticker=ticker,
        date=datetime(2024, 1, 15, tzinfo=timezone.utc),
        filepath=filepath,
    )


def _long_text(n: int = 2000) -> str:
    """Return a deterministic string of length n."""
    return ("Bloomberg financial data report. " * 100)[:n]


def _vsm() -> VectorStoreManager:
    """Instantiate a VectorStoreManager with all heavy backends patched."""
    with patch("chromadb.PersistentClient") as mock_client, \
         patch("sentence_transformers.SentenceTransformer"):
        mock_client.return_value.get_or_create_collection.return_value = MagicMock()
        return VectorStoreManager(persist_dir="/tmp/test_chroma", collection_name="test")


# ── chunk_text (core.document_loader) ────────────────────────────────────────


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_short_text_returns_single_chunk():
    text = "Apple reported record revenue."
    chunks = chunk_text(text, size=800, overlap=100)
    assert chunks == [text]


def test_chunk_text_text_exactly_size_returns_single_chunk():
    text = "X" * 800
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long_text_produces_multiple_chunks():
    chunks = chunk_text(_long_text(2000), size=800, overlap=100)
    assert len(chunks) > 1


def test_chunk_text_every_chunk_at_most_800_chars():
    chunks = chunk_text(_long_text(3000), size=800, overlap=100)
    for chunk in chunks:
        assert len(chunk) <= 800


def test_chunk_text_adjacent_chunks_share_100_char_overlap():
    text = _long_text(2000)
    chunks = chunk_text(text, size=800, overlap=100)
    # tail of chunk[0] == head of chunk[1]
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_text_full_text_is_covered():
    """Reconstruction: first chunk + non-overlapping tails covers the original."""
    text = _long_text(1700)
    chunks = chunk_text(text, size=800, overlap=100)
    step = 800 - 100
    reconstructed = chunks[0]
    for c in chunks[1:]:
        reconstructed += c[100:]
    assert reconstructed == text


def test_chunk_text_custom_size_and_overlap():
    text = "A" * 500
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 200


# ── DocumentLoader.chunk_documents ───────────────────────────────────────────


def test_chunk_documents_empty_list_returns_empty_list():
    loader = DocumentLoader()
    assert loader.chunk_documents([]) == []


def test_chunk_documents_short_doc_returns_one_chunk():
    loader = DocumentLoader()
    doc = _doc("Short content under 800 chars.")
    result = loader.chunk_documents([doc])
    assert len(result) == 1


def test_chunk_documents_short_doc_content_unchanged():
    loader = DocumentLoader()
    doc = _doc("Brief text.")
    result = loader.chunk_documents([doc])
    assert result[0].content == "Brief text."


def test_chunk_documents_long_doc_produces_multiple_chunks():
    loader = DocumentLoader()
    doc = _doc(_long_text(2000))
    result = loader.chunk_documents([doc])
    assert len(result) > 1


def test_chunk_documents_all_chunks_at_most_800_chars():
    loader = DocumentLoader()
    doc = _doc(_long_text(3000))
    for chunk_doc in loader.chunk_documents([doc]):
        assert len(chunk_doc.content) <= 800


def test_chunk_documents_preserves_ticker_on_every_chunk():
    loader = DocumentLoader()
    doc = _doc(_long_text(2000), ticker="MSFT")
    for chunk_doc in loader.chunk_documents([doc]):
        assert chunk_doc.ticker == "MSFT"


def test_chunk_documents_preserves_none_ticker():
    loader = DocumentLoader()
    base = _doc(_long_text(2000))
    base = DocumentSchema(
        content=base.content, source=base.source, ticker=None,
        date=base.date, filepath=base.filepath
    )
    for chunk_doc in loader.chunk_documents([base]):
        assert chunk_doc.ticker is None


def test_chunk_documents_preserves_date_on_every_chunk():
    loader = DocumentLoader()
    expected_date = datetime(2024, 3, 10, tzinfo=timezone.utc)
    doc = DocumentSchema(
        content=_long_text(2000), source="bloomberg_export",
        ticker="AAPL", date=expected_date, filepath="test.csv"
    )
    for chunk_doc in loader.chunk_documents([doc]):
        assert chunk_doc.date == expected_date


def test_chunk_documents_preserves_none_date():
    loader = DocumentLoader()
    doc = DocumentSchema(
        content=_long_text(2000), source="bloomberg_export",
        ticker="AAPL", date=None, filepath="test.csv"
    )
    for chunk_doc in loader.chunk_documents([doc]):
        assert chunk_doc.date is None


def test_chunk_documents_preserves_source_on_every_chunk():
    loader = DocumentLoader()
    doc = _doc(_long_text(2000))
    for chunk_doc in loader.chunk_documents([doc]):
        assert chunk_doc.source == "bloomberg_export"


def test_chunk_documents_preserves_filepath_on_every_chunk():
    loader = DocumentLoader()
    doc = _doc(_long_text(2000), filepath="/data/AAPL.csv")
    for chunk_doc in loader.chunk_documents([doc]):
        assert chunk_doc.filepath == "/data/AAPL.csv"


def test_chunk_documents_multiple_docs_each_get_chunked():
    loader = DocumentLoader()
    docs = [_doc(_long_text(2000)), _doc(_long_text(2000), ticker="MSFT")]
    result = loader.chunk_documents(docs)
    aapl_chunks = [d for d in result if d.ticker == "AAPL"]
    msft_chunks = [d for d in result if d.ticker == "MSFT"]
    assert len(aapl_chunks) > 1
    assert len(msft_chunks) > 1


def test_chunk_documents_respects_custom_chunk_size():
    loader = DocumentLoader()
    doc = _doc("B" * 1000)
    result = loader.chunk_documents([doc], chunk_size=300, overlap=50)
    assert len(result) > 1
    for chunk_doc in result:
        assert len(chunk_doc.content) <= 300


def test_chunk_documents_adjacent_chunks_overlap():
    loader = DocumentLoader()
    doc = _doc(_long_text(2000))
    chunks = loader.chunk_documents([doc])
    # Adjacent chunks share an overlap tail/head
    assert chunks[0].content[-100:] == chunks[1].content[:100]


# ── VectorStoreManager.add_documents_chunked ─────────────────────────────────


def test_add_documents_chunked_short_doc_calls_add_documents_once():
    vsm = _vsm()
    short_doc = _doc("Short content.")
    with patch.object(vsm, "add_documents", return_value=1) as mock_add:
        vsm.add_documents_chunked([short_doc])
    mock_add.assert_called_once()


def test_add_documents_chunked_long_doc_passes_multiple_chunks_to_add_documents():
    vsm = _vsm()
    long_doc = _doc(_long_text(2000))
    with patch.object(vsm, "add_documents", return_value=5) as mock_add:
        vsm.add_documents_chunked([long_doc])
    passed_docs = mock_add.call_args[0][0]
    assert len(passed_docs) > 1


def test_add_documents_chunked_chunks_preserve_ticker():
    vsm = _vsm()
    doc = _doc(_long_text(2000), ticker="NVDA")
    with patch.object(vsm, "add_documents", return_value=4) as mock_add:
        vsm.add_documents_chunked([doc])
    for chunk_doc in mock_add.call_args[0][0]:
        assert chunk_doc.ticker == "NVDA"


def test_add_documents_chunked_returns_count_from_add_documents():
    vsm = _vsm()
    doc = _doc(_long_text(2000))
    with patch.object(vsm, "add_documents", return_value=7):
        result = vsm.add_documents_chunked([doc])
    assert result == 7


def test_add_documents_chunked_empty_list_returns_zero():
    vsm = _vsm()
    result = vsm.add_documents_chunked([])
    assert result == 0


def test_add_documents_chunked_respects_custom_chunk_size():
    vsm = _vsm()
    doc = _doc("C" * 1000)
    with patch.object(vsm, "add_documents", return_value=3) as mock_add:
        vsm.add_documents_chunked([doc], chunk_size=300, overlap=50)
    passed_docs = mock_add.call_args[0][0]
    for chunk_doc in passed_docs:
        assert len(chunk_doc.content) <= 300


def test_add_documents_chunked_does_not_drop_any_content():
    """Every character in the original must appear in at least one chunk."""
    vsm = _vsm()
    text = _long_text(2000)
    doc = _doc(text)
    captured: list[DocumentSchema] = []

    def capture(docs):
        captured.extend(docs)
        return len(docs)

    with patch.object(vsm, "add_documents", side_effect=capture):
        vsm.add_documents_chunked([doc])

    # Reconstruct: first chunk full + non-overlapping tails
    chunks = [d.content for d in captured]
    reconstructed = chunks[0]
    for c in chunks[1:]:
        reconstructed += c[100:]
    assert reconstructed == text
