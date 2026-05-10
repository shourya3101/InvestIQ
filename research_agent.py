"""
Research Agent v2 using VectorStoreManager.

Retrieves evidence from the vector store and packages it into strict
Pydantic schemas (EvidenceSchema / ResearchOutputSchema).
No LLM calls – this agent only retrieves and structures evidence.
"""

from datetime import datetime
from typing import Optional

from config import CHROMA_DIR, CHROMA_COLLECTION, DEFAULT_TICKER
from schemas import EvidenceSchema, ResearchOutputSchema
from vector_store_manager import VectorStoreManager


def run_research(
    ticker: str,
    question: str,
    days_back: Optional[int] = 30,
    top_k: int = 5,
) -> ResearchOutputSchema:
    """
    Query the vector store and return a validated ResearchOutputSchema.

    Args:
        ticker: Stock ticker to search for (e.g. "AAPL").
        question: Natural-language research question.
        days_back: How many days back to search (``None`` = all time).
        top_k: Number of top results to retrieve.

    Returns:
        A ``ResearchOutputSchema`` with evidence items and a summary.
    """

    # ── initialise vector store ───────────────────────────────────────────
    persist_dir = str(CHROMA_DIR)
    collection_name = CHROMA_COLLECTION

    print(f"[INFO] Initializing vector store:")
    print(f"       persist_dir: {persist_dir}")
    print(f"       collection_name: {collection_name}\n")

    store = VectorStoreManager(
        persist_dir=persist_dir, collection_name=collection_name
    )

    # ── query ─────────────────────────────────────────────────────────────
    results = store.query(
        ticker=ticker,
        query_text=question,
        top_k=top_k,
        days_back=days_back,
        allow_fallback=True,
    )

    # ── build evidence list ───────────────────────────────────────────────
    evidence: list[EvidenceSchema] = []

    for idx, r in enumerate(results, 1):
        metadata = r["metadata"]
        text = r["text"]
        distance = r["distance"]
        
        # similarity = 1 - distance, clamped to [0, 1]
        similarity = max(0.0, min(1.0, 1.0 - distance))

        # snippet: first 280 chars, stripped
        snippet = text.strip()[:280]

        # date: parse ISO string from metadata
        raw_date = metadata.get("date", "")
        parsed_date: Optional[datetime] = None
        if raw_date:
            try:
                parsed_date = datetime.fromisoformat(raw_date)
            except (ValueError, TypeError):
                parsed_date = None

        evidence.append(
            EvidenceSchema(
                citation_id=f"E{idx}",
                snippet=snippet,
                filepath=metadata.get("filepath", ""),
                source=metadata.get("source", ""),
                ticker=metadata.get("ticker", "") or None,
                date=parsed_date,
                similarity_score=round(similarity, 4),
            )
        )

    # ── summary ───────────────────────────────────────────────────────────
    if not evidence:
        summary = f"No evidence found for {ticker}."
    else:
        all_time_filtered = all(r.get("time_filtered", False) for r in results)
        if all_time_filtered:
            summary = (
                f"Using time-filtered evidence from the last {days_back} days."
            )
        else:
            summary = (
                f"No recent evidence found in last {days_back} days; "
                f"using best available evidence (fallback)."
            )

    return ResearchOutputSchema(
        ticker=ticker,
        question=question,
        days_back=days_back,
        evidence=evidence,
        summary=summary,
    )


# ── test block ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("Research Agent v2 Test")
    print("=" * 80)

    QUESTION = "What are the key catalysts and risks?"
    DAYS_BACK = 30
    TOP_K = 5

    print(f"\n[1] Running research query...")
    print(f"    Ticker   : {DEFAULT_TICKER}")
    print(f"    Question : '{QUESTION}'")
    print(f"    Days back: {DAYS_BACK}")
    print(f"    Top K    : {TOP_K}\n")

    output = run_research(
        ticker=DEFAULT_TICKER,
        question=QUESTION,
        days_back=DAYS_BACK,
        top_k=TOP_K,
    )

    # ── print summary ────────────────────────────────────────────────────
    print("[2] Summary:\n")
    print(f"  Ticker  : {output.ticker}")
    print(f"  Question: {output.question}")
    print(f"  Window  : {output.days_back} days")
    print(f"  Summary : {output.summary}\n")

    # ── print evidence ───────────────────────────────────────────────────
    print(f"[3] Evidence Pack ({len(output.evidence)} item(s)):\n")
    if output.evidence:
        for e in output.evidence:
            print(f"  {e.citation_id} | similarity={e.similarity_score:.4f}")
            print(f"     date    : {e.date}")
            print(f"     filepath: {e.filepath}")
            print(f"     snippet : {e.snippet[:100]}{'...' if len(e.snippet) > 100 else ''}")
            print()
    else:
        print("  No evidence found.\n")

    print("=" * 80)
    print("Test completed!")
    print("=" * 80)
