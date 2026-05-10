"""
Sentiment Agent – evaluates sentiment over the Research Agent's evidence pack.

Uses VADER (rule-based) for polarity scoring.  No LLM calls.
"""

from datetime import datetime, timezone
from statistics import mean

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import DEFAULT_TICKER
from research_agent import run_research
from schemas import SentimentItemSchema, SentimentOutputSchema


# ── Helpers ─────────────────────────────────────────────────────────


def _label_from_compound(compound: float) -> str:
    """Map a VADER compound score to a human-readable label."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


# ── Main entry-point ────────────────────────────────────────────────


def run_sentiment(
    ticker: str,
    question: str = "What are the key catalysts and risks?",
    window_days: int = 365,
    top_k: int = 5,
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

    Returns
    -------
    SentimentOutputSchema
    """
    # ── Retrieve evidence ───────────────────────────────────────────
    research = run_research(
        ticker=ticker,
        question=question,
        days_back=window_days,
        top_k=top_k,
    )

    # ── Score each evidence item ────────────────────────────────────
    analyzer = SentimentIntensityAnalyzer()
    items: list[SentimentItemSchema] = []

    for ev in research.evidence:
        compound = analyzer.polarity_scores(ev.snippet)["compound"]
        label = _label_from_compound(compound)

        items.append(
            SentimentItemSchema(
                citation_id=ev.citation_id,
                polarity=round(compound, 4),
                label=label,
                rationale=f"VADER compound={compound:.2f}",
                date=ev.date,
                filepath=ev.filepath,
            )
        )

    # ── Aggregate ───────────────────────────────────────────────────
    if items:
        overall_score = round(mean(item.polarity for item in items), 4)
        overall_label = _label_from_compound(overall_score)
        summary = (
            f"Overall sentiment is {overall_label} "
            f"(score {overall_score:.2f}) "
            f"based on {len(items)} evidence items."
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


# ── Smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running sentiment analysis for {DEFAULT_TICKER} …\n")
    result = run_sentiment(DEFAULT_TICKER)

    print(f"Summary : {result.summary}")
    print(f"As-of   : {result.as_of.isoformat()}")
    print(f"Window  : {result.window_days}d\n")

    for item in result.items[:3]:
        print(
            f"  {item.citation_id:>8s}  |  {item.label:<9s}  "
            f"|  polarity {item.polarity:+.2f}  "
            f"|  {item.filepath}"
        )
