"""
Sentiment Agent – evaluates sentiment over the Research Agent's evidence pack.

Primary scorer : FinBERT (yiyanghkust/finbert-tone), loaded once as a singleton.
Fallback scorer: VADER (rule-based), used when transformers is unavailable or
                 when _scorer=None is injected explicitly (e.g. in tests).
"""

from datetime import datetime, timezone
from statistics import mean
from typing import Optional

from config import DEFAULT_TICKER
from agents.research_agent import run_research
from core.schemas import SentimentItemSchema, SentimentOutputSchema
from core.singletons import get_finbert_scorer
from core.vector_store_manager import VectorStoreManager

# Sentinel: distinguishes "not injected" (use singleton) from None (force VADER)
_UNSET = object()

_FINBERT_LABELS = {"positive", "negative", "neutral"}


# ── FinBERT helpers ───────────────────────────────────────────────────────────


def _finbert_polarity(label: str, score: float) -> float:
    """Map a FinBERT (label, score) pair to a signed polarity in [-1, +1].

    - "Positive" → +score
    - "Negative" → −score
    - "Neutral"  →  0.0
    - Unknown    →  0.0
    """
    normalised = label.strip().lower()
    if normalised == "positive":
        return max(-1.0, min(1.0, score))
    if normalised == "negative":
        return max(-1.0, min(1.0, -score))
    return 0.0


def _score_snippet(text: str, scorer) -> tuple[float, str]:
    """Score *text* with *scorer* (FinBERT) or VADER when scorer is None.

    Returns ``(polarity, label)`` where polarity ∈ [-1, +1] and
    label ∈ {"positive", "negative", "neutral"}.
    """
    if scorer is not None:
        result = scorer(text[:512])
        label_raw = result[0]["label"]
        score = result[0]["score"]
        polarity = _finbert_polarity(label_raw, score)
        label = label_raw.strip().lower()
        if label not in _FINBERT_LABELS:
            label = "neutral"
        return polarity, label

    # VADER fallback
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: PLC0415
    compound = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label


# ── Main entry-point ──────────────────────────────────────────────────────────


def run_sentiment(
    ticker: str,
    question: str = "What are the key catalysts and risks?",
    window_days: int = 365,
    top_k: int = 5,
    store: Optional[VectorStoreManager] = None,
    _scorer=_UNSET,
) -> SentimentOutputSchema:
    """Score sentiment for *ticker* using the Research Agent's evidence.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    question : str
        Research question forwarded to ``run_research``.
    window_days : int
        How many days of evidence to consider.
    top_k : int
        Maximum number of evidence items to retrieve.
    store : VectorStoreManager | None
        Shared vector store singleton (injected by coordinator).
    _scorer : callable | None | _UNSET
        Test-only. Pass a FakeScorer to bypass the real model.
        Pass None to force VADER. Omit to use the FinBERT singleton.
    """
    # Resolve scorer: use production singleton unless explicitly overridden
    if _scorer is _UNSET:
        scorer = get_finbert_scorer()
    else:
        scorer = _scorer

    scorer_name = "FinBERT" if scorer is not None else "VADER"

    # ── Retrieve evidence ─────────────────────────────────────────────
    research = run_research(
        ticker=ticker,
        question=question,
        days_back=window_days,
        top_k=top_k,
        store=store,
    )

    # ── Score each evidence item ──────────────────────────────────────
    items: list[SentimentItemSchema] = []

    for ev in research.evidence:
        polarity, label = _score_snippet(ev.snippet, scorer)

        items.append(
            SentimentItemSchema(
                citation_id=ev.citation_id,
                polarity=round(polarity, 4),
                label=label,
                rationale=f"{scorer_name} label={label}, score={polarity:.4f}",
                date=ev.date,
                filepath=ev.filepath,
            )
        )

    # ── Aggregate ─────────────────────────────────────────────────────
    if items:
        overall_score = round(mean(item.polarity for item in items), 4)
        if overall_score >= 0.05:
            overall_label = "positive"
        elif overall_score <= -0.05:
            overall_label = "negative"
        else:
            overall_label = "neutral"
        summary = (
            f"Overall sentiment is {overall_label} "
            f"(score {overall_score:.2f}) "
            f"based on {len(items)} evidence items ({scorer_name})."
        )
    else:
        overall_score = 0.0
        overall_label = "neutral"
        summary = "No evidence available for sentiment analysis."

    return SentimentOutputSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        window_days=window_days,
        overall_score=overall_score,
        overall_label=overall_label,
        items=items,
        summary=summary,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running sentiment analysis for {DEFAULT_TICKER} …\n")
    result = run_sentiment(DEFAULT_TICKER)

    print(f"Summary : {result.summary}")
    print(f"As-of   : {result.as_of.isoformat()}")
    print(f"Window  : {result.window_days}d\n")

    for item in result.items[:3]:
        print(
            f"  {item.citation_id:>8s}  |  {item.label:<9s}  "
            f"|  polarity {item.polarity:+.4f}  "
            f"|  {item.rationale}"
        )
