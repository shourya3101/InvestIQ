"""
Risk Agent – combines Trend + Sentiment outputs into structured risk flags.

Deterministic, rule-based scoring.  No LLM calls.
"""

from __future__ import annotations

from datetime import datetime, timezone

from typing import Optional

from config import DEFAULT_TICKER
from agents.trend_agent import run_trend
from agents.sentiment_agent import run_sentiment
from core.schemas import RiskFlagSchema, RiskOutputSchema
from core.vector_store_manager import VectorStoreManager


# ── Helpers ─────────────────────────────────────────────────────────


def _clip(x: float, lo: float, hi: float) -> float:
    """Clamp *x* to the interval [lo, hi]."""
    return max(lo, min(hi, x))


# severity → points mapping used by the scoring engine
_SEVERITY_POINTS: dict[str, int] = {"low": 0, "moderate": 0, "high": 0}

_DRAWDOWN_POINTS = {"low": 10, "moderate": 20, "high": 35}
_VOLATILITY_POINTS = {"low": 8, "moderate": 15, "high": 25}
_RETURN_POINTS = {"low": 8, "moderate": 15, "high": 25}
_SENTIMENT_POINTS = {"moderate": 10, "high": 20}


# ── Main entry-point ────────────────────────────────────────────────


def run_risk(
    ticker: str,
    mode: str = "live",
    price_filepath: str | None = None,
    question: str = "What are the key catalysts and risks?",
    window_days: int = 365,
    top_k: int = 5,
    store: Optional[VectorStoreManager] = None,
) -> RiskOutputSchema:
    """Assess risk for *ticker* by combining trend and sentiment signals.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    mode : str
        ``"live"`` or ``"offline"`` – forwarded to the Trend agent.
    price_filepath : str | None
        CSV/Parquet path when *mode* is ``"offline"``.
    question : str
        Research question forwarded to the Sentiment agent.
    window_days : int
        Evidence look-back window for sentiment.
    top_k : int
        Max evidence items for sentiment scoring.

    Returns
    -------
    RiskOutputSchema
    """
    # ── Gather sub-agent outputs ────────────────────────────────────
    trend = run_trend(
        ticker,
        mode=mode,
        filepath=price_filepath if mode == "offline" else None,
    )
    sent = run_sentiment(
        ticker,
        question=question,
        window_days=window_days,
        top_k=top_k,
        store=store,
    )

    # ── Pick the reference trend signal (prefer 30d) ────────────────
    sig = None
    for preferred in ("30d", "7d"):
        sig = next((s for s in trend.signals if s.horizon == preferred), None)
        if sig is not None:
            break
    if sig is None and trend.signals:
        sig = trend.signals[0]

    # Safely extract metrics (default to zero if no signal at all)
    max_dd = sig.max_drawdown_pct if sig else 0.0
    vol = sig.volatility_pct if sig else 0.0
    ret = sig.return_pct if sig else 0.0

    # ── Build risk flags ────────────────────────────────────────────
    flags: list[RiskFlagSchema] = []
    risk_score = 10.0  # baseline

    # 1) Price drawdown
    if max_dd <= -15:
        sev = "high"
    elif max_dd <= -8:
        sev = "moderate"
    elif max_dd <= -4:
        sev = "low"
    else:
        sev = None

    if sev is not None:
        flags.append(
            RiskFlagSchema(
                category="price",
                severity=sev,
                message=f"Max drawdown {max_dd:.1f}% over {sig.horizon if sig else 'N/A'}.",
            )
        )
        risk_score += _DRAWDOWN_POINTS[sev]

    # 2) Volatility
    if vol >= 45:
        sev = "high"
    elif vol >= 30:
        sev = "moderate"
    elif vol >= 20:
        sev = "low"
    else:
        sev = None

    if sev is not None:
        flags.append(
            RiskFlagSchema(
                category="volatility",
                severity=sev,
                message=f"Annualised volatility {vol:.1f}%.",
            )
        )
        risk_score += _VOLATILITY_POINTS[sev]

    # 3) Trend return
    if ret <= -8:
        sev = "high"
    elif ret <= -4:
        sev = "moderate"
    elif ret <= -2:
        sev = "low"
    else:
        sev = None

    if sev is not None:
        flags.append(
            RiskFlagSchema(
                category="price",
                severity=sev,
                message=f"Return {ret:+.1f}% over {sig.horizon if sig else 'N/A'}.",
            )
        )
        risk_score += _RETURN_POINTS[sev]

    # 4) Sentiment
    if sent.overall_label == "negative":
        sev = "high" if sent.overall_score <= -0.3 else "moderate"
        flags.append(
            RiskFlagSchema(
                category="sentiment",
                severity=sev,
                message=f"Overall sentiment is negative (score {sent.overall_score:.2f}).",
            )
        )
        risk_score += _SENTIMENT_POINTS[sev]

    # ── Clip and classify ───────────────────────────────────────────
    risk_score = _clip(risk_score, 0, 100)

    if risk_score <= 33:
        risk_level = "low"
    elif risk_score <= 66:
        risk_level = "moderate"
    else:
        risk_level = "high"

    # ── Summary ─────────────────────────────────────────────────────
    if flags:
        flag_msgs = "; ".join(f.message for f in flags)
        summary = (
            f"Risk level: {risk_level} (score {risk_score:.0f}/100). "
            f"Key flags: {flag_msgs}"
        )
    else:
        summary = (
            f"Risk level: {risk_level} (score {risk_score:.0f}/100). "
            "No material risk flags triggered."
        )

    return RiskOutputSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        risk_score=risk_score,
        risk_level=risk_level,
        flags=flags,
        summary=summary,
    )


# ── Smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running risk assessment for {DEFAULT_TICKER} (live) …\n")
    result = run_risk(DEFAULT_TICKER, mode="live")

    print(f"Summary : {result.summary}")
    print(f"Score   : {result.risk_score:.0f}/100")
    print(f"Level   : {result.risk_level}")
    print(f"As-of   : {result.as_of.isoformat()}\n")

    if result.flags:
        for f in result.flags:
            print(f"  [{f.severity:>8s}]  {f.category:<12s}  {f.message}")
    else:
        print("  (no flags)")
