"""
Tests for scripts/fetch_news.py — written BEFORE implementation (TDD).

Coverage:
  - chunk_text: basic chunking, overlap, edge cases
  - articles_to_documents: schema fields, date parsing
  - fetch_articles: missing key, rate-limit 429, happy path
  - ingest_news: missing key returns 0, success path
"""

import sys
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path so `from scripts.fetch_news import ...` works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.fetch_news import (
    articles_to_documents,
    chunk_text,
    fetch_articles,
    ingest_news,
)
from core.schemas import DocumentSchema


# ── chunk_text ────────────────────────────────────────────────────────────────


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_short_text_returns_single_chunk():
    text = "Hello world"
    chunks = chunk_text(text, size=800, overlap=100)
    assert chunks == [text]


def test_chunk_text_long_text_produces_multiple_chunks():
    text = "A" * 2000
    chunks = chunk_text(text, size=800, overlap=100)
    assert len(chunks) > 1


def test_chunk_text_each_chunk_at_most_size_chars():
    text = "B" * 3000
    chunks = chunk_text(text, size=800, overlap=100)
    for chunk in chunks:
        assert len(chunk) <= 800


def test_chunk_text_adjacent_chunks_share_overlap_chars():
    text = "C" * 2000
    chunks = chunk_text(text, size=800, overlap=100)
    # The tail of chunk[0] should equal the head of chunk[1]
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_text_covers_entire_text():
    text = "D" * 1700
    chunks = chunk_text(text, size=800, overlap=100)
    # Reconstruct: first chunk full, then each subsequent chunk adds (size-overlap) new chars
    step = 800 - 100
    reconstructed = chunks[0]
    for c in chunks[1:]:
        reconstructed += c[100:]
    assert reconstructed == text


# ── articles_to_documents ─────────────────────────────────────────────────────

SAMPLE_ARTICLE = {
    "title": "Apple hits new high",
    "description": "AAPL stock rose 3% on strong earnings.",
    "content": "Apple Inc. reported record profits. " * 10,
    "url": "https://example.com/aapl-news",
    "publishedAt": "2024-02-15T10:30:00Z",
    "source": {"name": "Reuters"},
}


def test_articles_to_documents_returns_document_schema_instances():
    docs = articles_to_documents([SAMPLE_ARTICLE], ticker="AAPL")
    assert len(docs) >= 1
    assert all(isinstance(d, DocumentSchema) for d in docs)


def test_articles_to_documents_sets_ticker():
    docs = articles_to_documents([SAMPLE_ARTICLE], ticker="AAPL")
    assert all(d.ticker == "AAPL" for d in docs)


def test_articles_to_documents_sets_source():
    docs = articles_to_documents([SAMPLE_ARTICLE], ticker="AAPL")
    assert all(d.source == "newsapi" for d in docs)


def test_articles_to_documents_sets_date():
    docs = articles_to_documents([SAMPLE_ARTICLE], ticker="AAPL")
    assert all(isinstance(d.date, datetime) for d in docs)
    assert docs[0].date.year == 2024
    assert docs[0].date.month == 2
    assert docs[0].date.day == 15


def test_articles_to_documents_missing_date_gives_none():
    article = {**SAMPLE_ARTICLE, "publishedAt": None}
    docs = articles_to_documents([article], ticker="AAPL")
    assert all(d.date is None for d in docs)


def test_articles_to_documents_content_not_empty():
    docs = articles_to_documents([SAMPLE_ARTICLE], ticker="AAPL")
    assert all(len(d.content) > 0 for d in docs)


def test_articles_to_documents_filepath_contains_ticker():
    docs = articles_to_documents([SAMPLE_ARTICLE], ticker="MSFT")
    assert all("MSFT" in d.filepath for d in docs)


# ── fetch_articles ────────────────────────────────────────────────────────────


def test_fetch_articles_missing_api_key_returns_empty_list():
    result = fetch_articles("AAPL", api_key="")
    assert result == []


def test_fetch_articles_none_api_key_returns_empty_list():
    result = fetch_articles("AAPL", api_key=None)
    assert result == []


def test_fetch_articles_rate_limit_returns_empty_list():
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {"status": "error", "code": "rateLimited"}

    with patch("scripts.fetch_news.requests.get", return_value=mock_resp):
        result = fetch_articles("AAPL", api_key="fake-key")

    assert result == []


def test_fetch_articles_success_returns_articles():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "totalResults": 2,
        "articles": [SAMPLE_ARTICLE, SAMPLE_ARTICLE],
    }

    with patch("scripts.fetch_news.requests.get", return_value=mock_resp):
        result = fetch_articles("AAPL", api_key="fake-key")

    assert len(result) == 2
    assert result[0]["title"] == "Apple hits new high"


def test_fetch_articles_api_error_status_returns_empty_list():
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"status": "error", "code": "apiKeyInvalid"}

    with patch("scripts.fetch_news.requests.get", return_value=mock_resp):
        result = fetch_articles("AAPL", api_key="bad-key")

    assert result == []


# ── ingest_news ───────────────────────────────────────────────────────────────


def test_ingest_news_missing_api_key_returns_zero_without_crash():
    store = MagicMock()
    count = ingest_news("AAPL", api_key="", store=store)
    assert count == 0
    store.add_documents.assert_not_called()


def test_ingest_news_happy_path_returns_ingested_count():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "totalResults": 1,
        "articles": [SAMPLE_ARTICLE],
    }

    store = MagicMock()
    store.add_documents.return_value = 3  # suppose 1 article → 3 chunks

    with patch("scripts.fetch_news.requests.get", return_value=mock_resp):
        count = ingest_news("AAPL", api_key="fake-key", store=store)

    assert count == 3
    store.add_documents.assert_called_once()


def test_ingest_news_passes_document_schemas_to_store():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "totalResults": 1,
        "articles": [SAMPLE_ARTICLE],
    }

    store = MagicMock()
    store.add_documents.return_value = 1

    with patch("scripts.fetch_news.requests.get", return_value=mock_resp):
        ingest_news("AAPL", api_key="fake-key", store=store)

    call_args = store.add_documents.call_args[0][0]
    assert all(isinstance(d, DocumentSchema) for d in call_args)
