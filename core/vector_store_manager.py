"""
Vector Store Manager v2 using ChromaDB and Sentence Transformers.

Schema-native: accepts DocumentSchema objects directly.
Metadata is always JSON-serializable (no datetime objects in Chroma).
Retrieval supports time-awareness with a fallback when no dated docs match.
"""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL_NAME
from core.schemas import DocumentSchema


class VectorStoreManager:
    """
    Local vector store using ChromaDB and sentence-transformers embeddings.

    Features:
    - Persistent storage via ChromaDB
    - Automatic embedding using the configured model
    - Stable deduplication via SHA-1 hashing
    - Ticker filtering at the Chroma level
    - Time-aware retrieval with optional fallback
    """

    def __init__(
        self,
        persist_dir: str = str(CHROMA_DIR),
        collection_name: str = CHROMA_COLLECTION,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        settings = Settings(allow_reset=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=settings,
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _create_document_id(filepath: str, content: str) -> str:
        """SHA-1 hash of ``filepath|content`` for stable dedup."""
        combined = f"{filepath}|{content}"
        return hashlib.sha1(combined.encode()).hexdigest()

    @staticmethod
    def _parse_meta_date(meta_date: str) -> Optional[datetime]:
        """Parse an ISO-format date string stored in Chroma metadata.

        Returns ``None`` for empty strings or unparsable values.
        """
        if not meta_date:
            return None
        try:
            return datetime.fromisoformat(meta_date)
        except (ValueError, TypeError):
            return None

    # ── ingestion ─────────────────────────────────────────────────────────

    def add_documents(self, docs: list[DocumentSchema]) -> int:
        """Upsert *docs* into the vector store.

        Metadata values are normalised to non-``None`` strings so that
        ChromaDB never receives unserialisable objects.

        Returns the number of documents upserted.
        """
        if not docs:
            return 0

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for doc in docs:
            doc_id = self._create_document_id(
                doc.filepath or "", doc.content
            )
            ids.append(doc_id)

            embedding = self.embedder.encode(doc.content).tolist()
            embeddings.append(embedding)

            documents.append(doc.content)

            # Normalise metadata – every value must be a plain string.
            if doc.date is not None:
                if isinstance(doc.date, datetime):
                    date_str = doc.date.isoformat()
                else:
                    date_str = str(doc.date)
            else:
                date_str = ""

            metadata = {
                "ticker": doc.ticker or "",
                "source": doc.source or "",
                "filepath": doc.filepath or "",
                "date": date_str,
            }
            # Chroma rejects None values; only write the key when present so
            # legacy documents remain distinguishable from score-0 documents.
            if doc.about_score is not None:
                metadata["about_score"] = float(doc.about_score)
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(ids)

    def add_documents_chunked(
        self,
        docs: list[DocumentSchema],
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> int:
        """Chunk *docs* then upsert all windows into the vector store.

        Each chunk inherits the ticker, date, source, and filepath of its
        parent document.  Short documents (content ≤ chunk_size) are stored
        as a single chunk — identical to calling ``add_documents`` directly.

        Args:
            docs: Source documents to chunk and ingest.
            chunk_size: Maximum characters per chunk (default 800).
            overlap: Characters shared between adjacent chunks (default 100).

        Returns:
            Number of chunk documents upserted.
        """
        if not docs:
            return 0
        # Lazy import avoids circular dependency at module load time.
        from core.document_loader import DocumentLoader
        chunks = DocumentLoader().chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
        return self.add_documents(chunks)

    # ── retrieval ─────────────────────────────────────────────────────────

    def query(
        self,
        ticker: str,
        query_text: str,
        top_k: int = 5,
        days_back: Optional[int] = None,
        allow_fallback: bool = True,
    ) -> list[dict]:
        """Query the vector store with optional time filtering.

        Args:
            ticker: Ticker symbol to filter on (Chroma-level).
            query_text: Natural-language query to embed.
            top_k: Desired number of results.
            days_back: If set, only keep docs whose date >= now - days_back.
            allow_fallback: When ``True`` and strict time filtering yields
                0 results, return the best unfiltered results instead (each
                result will carry ``time_filtered: False``).

        Returns:
            List of dicts with keys ``text``, ``metadata``, ``distance``,
            and ``time_filtered`` (bool).
        """
        query_embedding = self.embedder.encode(query_text).tolist()

        # Retrieve extra candidates so time-filtering still has enough.
        fetch_n = top_k * 4 if days_back is not None else top_k

        where_filter = None
        if ticker:
            where_filter = {"ticker": ticker}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # ── build candidate list ──────────────────────────────────────────
        candidates: list[dict] = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                candidates.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )

        # ── time filtering ────────────────────────────────────────────────
        if days_back is not None:
            cutoff = datetime.now() - timedelta(days=days_back)
            time_filtered: list[dict] = []
            for c in candidates:
                parsed = self._parse_meta_date(c["metadata"].get("date", ""))
                if parsed is not None and parsed >= cutoff:
                    c["time_filtered"] = True
                    time_filtered.append(c)

            if time_filtered:
                return time_filtered[:top_k]

            # strict filtering returned nothing
            if allow_fallback:
                for c in candidates:
                    c["time_filtered"] = False
                return candidates[:top_k]

            return []  # strict, no fallback

        # No time filter requested
        for c in candidates:
            c["time_filtered"] = True
        return candidates[:top_k]


# ── test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from core.document_loader import DocumentLoader

    print("=" * 70)
    print("Vector Store Manager v2 Test")
    print("=" * 70)

    loader = DocumentLoader()
    store = VectorStoreManager()

    # ── 1. Load documents ─────────────────────────────────────────────────
    print("\n[1] Loading sample documents...")
    docs: list[DocumentSchema] = []

    sample_txt_path = Path("sample_document.txt")
    if sample_txt_path.exists():
        docs.extend(loader.load_txt(sample_txt_path, ticker="AAPL"))
        print("  ✓ Loaded sample_document.txt")
    else:
        print("  ⚠ sample_document.txt not found")

    sample_csv_path = Path("sample_bloomberg_export.csv")
    if sample_csv_path.exists():
        docs.extend(loader.load_csv(sample_csv_path, ticker="AAPL"))
        print("  ✓ Loaded sample_bloomberg_export.csv")
    else:
        print("  ⚠ sample_bloomberg_export.csv not found")

    if not docs:
        print("  ℹ Creating sample_document.txt for demo...")
        sample_txt_path.write_text(
            "Apple Inc. (AAPL) Financial Report\n"
            "Date: 2025-01-25\n\n"
            "AAPL recently announced strong quarterly earnings with revenue "
            "of $120 billion.\n"
            "The stock price increased 5% following the announcement.\n"
            "Market analysts are bullish on the company's AI initiatives.\n"
        )
        docs.extend(loader.load_txt(sample_txt_path, ticker="AAPL"))
        print("  ✓ Created and loaded sample_document.txt")

    # ── 2. Ingest ─────────────────────────────────────────────────────────
    print(f"\n[2] Upserting {len(docs)} document(s) into vector store...")
    added = store.add_documents(docs)
    print(f"  ✓ Upserted {added} document(s)")

    # ── 3A. Query with days_back=30 (likely fallback) ─────────────────────
    QUERY = "What happened to AAPL recently?"

    print(f"\n[3A] Query (days_back=30): '{QUERY}'")
    results_30 = store.query(ticker="AAPL", query_text=QUERY, top_k=3, days_back=30)
    print(f"  Found {len(results_30)} result(s)")
    for rank, r in enumerate(results_30, 1):
        meta = r["metadata"]
        print(f"  [{rank}] dist={r['distance']:.4f}  time_filtered={r['time_filtered']}")
        print(f"       date={meta.get('date', '')}  ticker={meta.get('ticker', '')}")
        snippet = r["text"][:80] + ("..." if len(r["text"]) > 80 else "")
        print(f"       snippet: {snippet}")

    # ── 3B. Query with days_back=365 ──────────────────────────────────────
    print(f"\n[3B] Query (days_back=365): '{QUERY}'")
    results_365 = store.query(ticker="AAPL", query_text=QUERY, top_k=3, days_back=365)
    print(f"  Found {len(results_365)} result(s)")
    for rank, r in enumerate(results_365, 1):
        meta = r["metadata"]
        print(f"  [{rank}] dist={r['distance']:.4f}  time_filtered={r['time_filtered']}")
        print(f"       date={meta.get('date', '')}  ticker={meta.get('ticker', '')}")
        snippet = r["text"][:80] + ("..." if len(r["text"]) > 80 else "")
        print(f"       snippet: {snippet}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
