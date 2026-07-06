"""
Research Agent v3 — delegates retrieval policy to core.retrieval.

Packages gated evidence into ResearchOutputSchema, carrying the typed
evidence_status so downstream agents can refuse to fabricate.
No LLM calls — this agent only retrieves and structures evidence.
"""

from typing import Optional

from config import DEFAULT_TICKER
from core.retrieval import retrieve_evidence
from core.schemas import ResearchOutputSchema
from core.vector_store_manager import VectorStoreManager


def run_research(
    ticker: str,
    question: str,
    days_back: Optional[int] = 30,
    top_k: int = 5,
    store: Optional[VectorStoreManager] = None,
) -> ResearchOutputSchema:
    """Run gated retrieval and return a validated ResearchOutputSchema."""
    result = retrieve_evidence(
        ticker=ticker,
        question=question,
        days_back=days_back,
        top_k=top_k,
        store=store,
    )

    if result.evidence_status == "insufficient":
        summary = f"No trustworthy evidence found for {result.ticker}. {result.status_reason}"
    elif result.evidence_status == "partial":
        summary = f"Partial evidence for {result.ticker}. {result.status_reason}"
    else:
        summary = f"Sufficient evidence for {result.ticker}. {result.status_reason}"

    return ResearchOutputSchema(
        ticker=result.ticker,
        question=question,
        days_back=days_back,
        evidence=result.evidence,
        summary=summary,
        evidence_status=result.evidence_status,
        status_reason=result.status_reason,
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
