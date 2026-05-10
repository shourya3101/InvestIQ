"""
Fetch financial news from NewsAPI, chunk it, and ingest into ChromaDB.

Usage (CLI):
    python scripts/fetch_news.py AAPL
    python scripts/fetch_news.py MSFT --page-size 50

Requires NEWSAPI_KEY in .env (or environment).
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# Allow running from repo root or from scripts/ directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHROMA_DIR, CHROMA_COLLECTION
from core.schemas import DocumentSchema
from core.vector_store_manager import VectorStoreManager

_NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
_DEFAULT_CHUNK_SIZE = 800
_DEFAULT_OVERLAP = 100


# ── chunking ──────────────────────────────────────────────────────────────────


def chunk_text(text: str, size: int = _DEFAULT_CHUNK_SIZE, overlap: int = _DEFAULT_OVERLAP) -> list[str]:
    """Split *text* into windows of *size* chars with *overlap* chars shared
    between adjacent windows. Returns [] for empty input."""
    if not text:
        return []
    if len(text) <= size:
        return [text]

    step = size - overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


# ── API fetch ─────────────────────────────────────────────────────────────────


def fetch_articles(
    ticker: str,
    api_key: Optional[str],
    page_size: int = 20,
) -> list[dict]:
    """Fetch up to *page_size* articles about *ticker* from NewsAPI.

    Returns an empty list when:
    - *api_key* is absent/empty
    - the API returns a non-200 status (rate limit, bad key, …)
    """
    if not api_key:
        return []

    params = {
        "q": ticker,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }

    try:
        response = requests.get(_NEWSAPI_ENDPOINT, params=params, timeout=10)
    except requests.RequestException:
        return []

    if response.status_code != 200:
        return []

    data = response.json()
    if data.get("status") != "ok":
        return []

    return data.get("articles", [])


# ── schema conversion ─────────────────────────────────────────────────────────


def articles_to_documents(articles: list[dict], ticker: str) -> list[DocumentSchema]:
    """Convert NewsAPI article dicts into chunked DocumentSchema objects."""
    docs: list[DocumentSchema] = []

    for article in articles:
        raw_content = " ".join(filter(None, [
            article.get("title", ""),
            article.get("description", ""),
            article.get("content", ""),
        ]))

        published_at = article.get("publishedAt")
        date: Optional[datetime] = None
        if published_at:
            try:
                date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                date = date.replace(tzinfo=None)  # store as naive UTC for ChromaDB
            except (ValueError, AttributeError):
                date = None

        url = article.get("url", "")
        filepath = f"newsapi/{ticker}/{url.split('/')[-1] or 'article'}"

        for chunk in chunk_text(raw_content):
            docs.append(
                DocumentSchema(
                    content=chunk,
                    source="newsapi",
                    ticker=ticker,
                    date=date,
                    filepath=filepath,
                )
            )

    return docs


# ── orchestration ─────────────────────────────────────────────────────────────


def ingest_news(
    ticker: str,
    api_key: Optional[str],
    store: VectorStoreManager,
    page_size: int = 20,
) -> int:
    """Fetch, chunk, and ingest news for *ticker*. Returns count of upserted docs."""
    if not api_key:
        return 0

    articles = fetch_articles(ticker, api_key, page_size=page_size)
    if not articles:
        return 0

    docs = articles_to_documents(articles, ticker)
    if not docs:
        return 0

    return store.add_documents(docs)


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch and ingest news into ChromaDB.")
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument("--page-size", type=int, default=20, help="Articles to fetch (max 100)")
    args = parser.parse_args()

    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        print("ERROR: NEWSAPI_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    store = VectorStoreManager(
        persist_dir=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION,
    )

    count = ingest_news(args.ticker, api_key, store, page_size=args.page_size)
    print(f"Ingested {count} chunk(s) for {args.ticker}.")


if __name__ == "__main__":
    main()
