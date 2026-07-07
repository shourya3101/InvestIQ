"""Ingestion: company-name query, article-level aboutness floor, metadata plumbing."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

from core.company_registry import CompanyInfo
from core.document_loader import DocumentLoader
from core.schemas import DocumentSchema
from scripts.fetch_news import articles_to_documents, fetch_articles

TSLA = CompanyInfo(ticker="TSLA", name="Tesla", aliases=["Tesla"], source="config")


def _article(title, description="", content="", url="https://x.com/a"):
    return {"title": title, "description": description, "content": content,
            "publishedAt": "2026-06-30T12:00:00Z", "url": url}


# ── query construction ────────────────────────────────────────────────────────

def test_query_uses_company_name_or_ticker():
    with patch("scripts.fetch_news.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "ok", "articles": []}
        fetch_articles("TSLA", "key", company=TSLA)
    params = mock_get.call_args.kwargs["params"]
    assert params["q"] == '"Tesla" OR "TSLA"'


# ── article-level floor ───────────────────────────────────────────────────────

def test_passing_mention_article_is_skipped():
    arts = [_article("Bybit lists new contracts",
                     content="Contracts for TSLA, AMZN, META and other symbols.")]
    docs = articles_to_documents(arts, "TSLA", company=TSLA)
    assert docs == []

def test_on_topic_article_is_kept_with_about_score():
    arts = [_article("Tesla beats delivery estimates",
                     description="Tesla shares rose.",
                     content="Tesla reported record quarterly deliveries.")]
    docs = articles_to_documents(arts, "TSLA", company=TSLA)
    assert len(docs) >= 1
    assert all(d.about_score is not None and d.about_score >= 0.3 for d in docs)

def test_title_mention_weighted_double():
    # Name only in the title: title counted twice → 2 name hits → score 1.0
    arts = [_article("Tesla expands Berlin plant", content="The factory will grow.")]
    docs = articles_to_documents(arts, "TSLA", company=TSLA)
    assert docs and docs[0].about_score == 1.0


# ── chunk propagation ─────────────────────────────────────────────────────────

def test_chunk_documents_propagates_about_score():
    doc = DocumentSchema(content="x" * 2000, source="newsapi", ticker="TSLA",
                         filepath="f", about_score=0.8)
    chunks = DocumentLoader().chunk_documents([doc])
    assert len(chunks) > 1
    assert all(c.about_score == 0.8 for c in chunks)


# ── vector store metadata ─────────────────────────────────────────────────────

def _vsm():
    # NOTE: patch the consuming module, not sentence_transformers itself —
    # VectorStoreManager binds the name at import time (from X import Y).
    from core.vector_store_manager import VectorStoreManager
    with patch("chromadb.PersistentClient") as mock_client, \
         patch("core.vector_store_manager.SentenceTransformer") as mock_st:
        mock_client.return_value.get_or_create_collection.return_value = MagicMock()
        mock_st.return_value.encode.return_value.tolist.return_value = [0.0]
        store = VectorStoreManager(persist_dir="/tmp/t", collection_name="t")
    return store

def test_add_documents_writes_about_score_when_present():
    store = _vsm()
    doc = DocumentSchema(content="c", source="s", ticker="TSLA", filepath="f",
                         about_score=0.75)
    store.add_documents([doc])
    metadata = store.collection.upsert.call_args.kwargs["metadatas"][0]
    assert metadata["about_score"] == 0.75

def test_add_documents_omits_about_score_when_none():
    store = _vsm()
    doc = DocumentSchema(content="c", source="s", ticker="TSLA", filepath="f")
    store.add_documents([doc])
    metadata = store.collection.upsert.call_args.kwargs["metadatas"][0]
    assert "about_score" not in metadata
