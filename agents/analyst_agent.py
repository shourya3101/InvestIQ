"""
Analyst Agent – generates the final investment memo.

LLM mode splits the work into three focused calls so no single prompt
exhausts the token budget before completing:
  1. _write_thesis          – 5-7 sentence investment thesis
  2. _write_catalysts_risks – catalysts + risks via JSON
  3. _write_recommendation  – action signal via JSON

Each call uses max_tokens=1500 and falls back to deterministic output on
any failure, so the pipeline never crashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config import DEFAULT_TICKER, LLM_MODE, GROQ_MODEL, CLAUDE_MODEL, OPENAI_MODEL
from core.schemas import (
    ActionSignalSchema,
    InvestmentMemoSchema,
    ResearchOutputSchema,
    RiskOutputSchema,
    SentimentOutputSchema,
    TrendOutputSchema,
)
from core.utils import parse_llm_json
from llm.providers import GroqProvider, ClaudeProvider, LLMRouter, OpenAIProvider

_MAX_TOKENS = 1500
_VALID_SIGNALS = ("buy", "hold", "sell")
_PARTIAL_CONFIDENCE_CAP = 0.6


# ── Shared context dataclass ──────────────────────────────────────────────────


@dataclass
class _MemoContext:
    """Formatted data bundle shared across all three LLM calls."""
    ticker: str
    trend_summary: str
    sentiment_label: str
    sentiment_score: float
    risk_level: str
    risk_score: float
    risk_flags: str
    evidence_lines: str


def _build_memo_context(
    ticker: str,
    research: ResearchOutputSchema,
    trend: TrendOutputSchema,
    sentiment: SentimentOutputSchema,
    risk: RiskOutputSchema,
) -> _MemoContext:
    sig = next((s for s in trend.signals if s.horizon == "30d"), None)
    sig = sig or next((s for s in trend.signals if s.horizon == "7d"), None)
    sig = sig or (trend.signals[0] if trend.signals else None)

    trend_summary = (
        f"{sig.horizon} return {sig.return_pct:+.1f}%, "
        f"trend {sig.trend_label}, vol {sig.volatility_pct:.1f}%"
        if sig else "No trend data available"
    )
    risk_flags = "; ".join(f.message for f in risk.flags) or "None"
    evidence_lines = "\n".join(
        f"{e.citation_id}: {e.snippet[:200]}" for e in research.evidence[:5]
    ) or "No evidence retrieved."

    return _MemoContext(
        ticker=ticker,
        trend_summary=trend_summary,
        sentiment_label=sentiment.overall_label,
        sentiment_score=sentiment.overall_score,
        risk_level=risk.risk_level,
        risk_score=risk.risk_score,
        risk_flags=risk_flags,
        evidence_lines=evidence_lines,
    )


# ── Call 1: thesis ────────────────────────────────────────────────────────────


def _write_thesis(provider, ctx: _MemoContext, evidence_note: str = "") -> str:
    """One focused LLM call: returns a 5-7 sentence investment thesis string."""
    system = (
        "You are a senior equity analyst writing a concise investment memo. "
        "Write only the investment thesis section. "
        "Use complete sentences. Do not use bullet points or headers."
    )
    user = (
        f"Write a 5-7 sentence investment thesis for {ctx.ticker}.\n\n"
        f"Data:\n"
        f"- Trend: {ctx.trend_summary}\n"
        f"- Sentiment: {ctx.sentiment_label} (score {ctx.sentiment_score:.2f})\n"
        f"- Risk: {ctx.risk_level} (score {ctx.risk_score:.0f}/100)\n"
        f"- Risk flags: {ctx.risk_flags}\n\n"
        f"Evidence:\n{ctx.evidence_lines}\n\n"
        f"{evidence_note}"
        f"Write ONLY the thesis paragraph. Use 5-7 complete sentences. "
        f"Do not truncate mid-sentence."
    )
    try:
        result = provider.generate(system, user).strip()
        if result:
            return result
    except Exception:
        pass

    return (
        f"{ctx.ticker} shows a {ctx.risk_level}-risk profile "
        f"(score {ctx.risk_score:.0f}/100) with "
        f"{ctx.sentiment_label} sentiment (score {ctx.sentiment_score:.2f}). "
        f"Trend: {ctx.trend_summary}."
    )


def _write_market_data_thesis(provider, ctx: _MemoContext, status_reason: str) -> str:
    """Thesis for the insufficient-evidence path: market data only, no fabrication."""
    system = (
        "You are a senior equity analyst. Write only from the market data provided. "
        "You have NO news or document evidence — do not invent company events, "
        "products, or fundamentals. Use complete sentences, no bullets or headers."
    )
    user = (
        f"Retrieval found no trustworthy evidence for {ctx.ticker} ({status_reason}).\n"
        f"Write a 3-4 sentence market-data-only summary for {ctx.ticker}. "
        f"State first that no reliable evidence was found and no investment view is taken.\n\n"
        f"Market data:\n"
        f"- Trend: {ctx.trend_summary}\n"
        f"- Risk: {ctx.risk_level} (score {ctx.risk_score:.0f}/100)\n"
        f"- Risk flags: {ctx.risk_flags}\n"
    )
    return provider.generate(system, user).strip()


def _insufficient_memo(
    ticker: str,
    research: ResearchOutputSchema,
    trend: TrendOutputSchema,
    sentiment: SentimentOutputSchema,
    risk: RiskOutputSchema,
    question: str,
    mode: str,
) -> InvestmentMemoSchema:
    """Degraded memo: genuine no-view, market-data claims only, no fabrication."""
    ctx = _build_memo_context(ticker, research, trend, sentiment, risk)

    thesis = ""
    if mode in ("groq", "claude", "auto", "openai"):
        try:
            provider = _build_provider(mode)
            thesis = _write_market_data_thesis(provider, ctx, research.status_reason)
        except Exception:
            thesis = ""
    if not thesis:
        thesis = (
            f"No trustworthy evidence was retrieved for {ticker}; this report is "
            f"based on market data only and takes no investment view. "
            f"Trend: {ctx.trend_summary}. "
            f"Risk: {ctx.risk_level} (score {ctx.risk_score:.0f}/100)."
        )

    return InvestmentMemoSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        question=question,
        thesis=thesis,
        catalysts=[],
        risks=[f.message for f in risk.flags] or ["No material risk flags triggered."],
        action=ActionSignalSchema(
            signal="no_view",
            confidence=0.0,
            rationale=f"No trustworthy evidence retrieved for {ticker}; declining to take a view.",
        ),
        citations=[],
        risk_level=risk.risk_level,
        risk_score=risk.risk_score,
        writer_mode=mode,
        evidence_status="insufficient",
    )


# ── Call 2: catalysts + risks ─────────────────────────────────────────────────


def _write_catalysts_risks(
    provider, ctx: _MemoContext
) -> tuple[list[str], list[str]]:
    """One focused LLM call: returns (catalysts, risks) parsed from JSON."""
    system = (
        "You are a senior equity analyst. "
        "Output ONLY valid JSON — no markdown fences, no prose."
    )
    user = (
        f"List the key investment catalysts and risks for {ctx.ticker}.\n\n"
        f"Evidence:\n{ctx.evidence_lines}\n\n"
        f"Data:\n"
        f"- Trend: {ctx.trend_summary}\n"
        f"- Sentiment: {ctx.sentiment_label}\n"
        f"- Risk flags: {ctx.risk_flags}\n\n"
        f'Output exactly this JSON:\n'
        f'{{\n'
        f'  "catalysts": ["catalyst 1", "catalyst 2", "catalyst 3"],\n'
        f'  "risks": ["risk 1", "risk 2", "risk 3"]\n'
        f'}}'
    )

    _fallback_catalysts = [ctx.evidence_lines[:120]] if ctx.evidence_lines.strip() != "No evidence retrieved." else ["Insufficient evidence available."]
    _fallback_risks = [ctx.risk_flags] if ctx.risk_flags != "None" else ["See trend and sentiment data."]

    try:
        raw = provider.generate(system, user)
        parsed = parse_llm_json(raw, ["catalysts", "risks"])
        if parsed:
            catalysts = [str(c) for c in parsed["catalysts"]] or _fallback_catalysts
            risks = [str(r) for r in parsed["risks"]] or _fallback_risks
            return catalysts, risks
    except Exception:
        pass

    return _fallback_catalysts, _fallback_risks


# ── Call 3: recommendation ────────────────────────────────────────────────────


def _write_recommendation(
    provider, ctx: _MemoContext
) -> tuple[str, float, str]:
    """One focused LLM call: returns (signal, confidence, rationale) from JSON."""
    system = (
        "You are a senior equity analyst making an investment recommendation. "
        "Output ONLY valid JSON — no markdown fences, no prose."
    )
    user = (
        f"Make an investment recommendation for {ctx.ticker}.\n\n"
        f"Data:\n"
        f"- Trend: {ctx.trend_summary}\n"
        f"- Sentiment: {ctx.sentiment_label} (score {ctx.sentiment_score:.2f})\n"
        f"- Risk: {ctx.risk_level} (score {ctx.risk_score:.0f}/100)\n\n"
        f'Output exactly this JSON:\n'
        f'{{\n'
        f'  "signal": "buy",\n'
        f'  "confidence": 0.70,\n'
        f'  "rationale": "One sentence explaining the recommendation."\n'
        f'}}\n'
        f'signal must be exactly one of: buy, hold, sell'
    )

    try:
        raw = provider.generate(system, user)
        parsed = parse_llm_json(raw, ["signal", "confidence", "rationale"])
        if parsed:
            signal = str(parsed["signal"]).lower()
            if signal not in _VALID_SIGNALS:
                signal = "hold"
            confidence = max(0.0, min(1.0, float(parsed["confidence"])))
            rationale = str(parsed["rationale"])
            return signal, confidence, rationale
    except Exception:
        pass

    return _deterministic_recommendation(ctx)


def _deterministic_recommendation(ctx: _MemoContext) -> tuple[str, float, str]:
    if ctx.risk_level == "low" and ctx.sentiment_label == "positive":
        return "buy", 0.75, "Low risk and positive sentiment support accumulation."
    if ctx.risk_level == "high" or ctx.sentiment_label == "negative":
        return "sell", 0.65, "Elevated risk or negative sentiment warrants caution."
    return "hold", 0.60, "Mixed signals; maintain current position and monitor."


# ── Provider factory ──────────────────────────────────────────────────────────


def _build_provider(mode: str):
    """Construct the configured LLM provider with max_tokens=1500."""
    if mode == "groq":
        return GroqProvider(model=GROQ_MODEL, max_tokens=_MAX_TOKENS)
    if mode == "claude":
        return ClaudeProvider(model=CLAUDE_MODEL, max_tokens=_MAX_TOKENS)
    if mode == "openai":
        return OpenAIProvider(model=OPENAI_MODEL, max_tokens=_MAX_TOKENS)
    # auto
    return LLMRouter(
        primary=GroqProvider(model=GROQ_MODEL, max_tokens=_MAX_TOKENS),
        fallback=ClaudeProvider(model=CLAUDE_MODEL, max_tokens=_MAX_TOKENS),
    )


# ── Main entry-point ──────────────────────────────────────────────────────────


def run_analyst_memo(
    ticker: str,
    research: ResearchOutputSchema,
    trend: TrendOutputSchema,
    sentiment: SentimentOutputSchema,
    risk: RiskOutputSchema,
    question: str = "What are the key catalysts and risks?",
    writer_mode: Optional[str] = None,
) -> InvestmentMemoSchema:
    """Generate an investment memo using 3 focused LLM calls (or deterministic fallback)."""
    mode = writer_mode or LLM_MODE
    ctx = _build_memo_context(ticker, research, trend, sentiment, risk)

    status = research.evidence_status
    if status == "insufficient":
        return _insufficient_memo(ticker, research, trend, sentiment, risk, question, mode)

    evidence_note = (
        f"NOTE: the evidence base is limited ({research.status_reason}) — "
        f"explicitly acknowledge the limited evidence in the thesis.\n\n"
        if status == "partial" else ""
    )

    citations = [e.citation_id for e in research.evidence]

    # ── LLM path ──────────────────────────────────────────────────────
    if mode in ("groq", "claude", "auto", "openai"):
        try:
            provider = _build_provider(mode)

            thesis = _write_thesis(provider, ctx, evidence_note)
            catalysts, risks = _write_catalysts_risks(provider, ctx)
            signal, confidence, rationale = _write_recommendation(provider, ctx)
            if status == "partial":
                confidence = min(confidence, _PARTIAL_CONFIDENCE_CAP)

            return InvestmentMemoSchema(
                ticker=ticker,
                as_of=datetime.now(timezone.utc),
                question=question,
                thesis=thesis,
                catalysts=catalysts,
                risks=risks,
                action=ActionSignalSchema(
                    signal=signal,
                    confidence=confidence,
                    rationale=rationale,
                ),
                citations=citations,
                risk_level=risk.risk_level,
                risk_score=risk.risk_score,
                writer_mode=mode,
                evidence_status=status,
            )

        except Exception:
            pass  # fall through to deterministic

    # ── Deterministic path ────────────────────────────────────────────
    thesis = (
        f"{ticker} shows a {risk.risk_level}-risk profile "
        f"(score {risk.risk_score:.0f}/100) with "
        f"{sentiment.overall_label} sentiment "
        f"(score {sentiment.overall_score:.2f})."
    )

    catalysts = [
        e.snippet[:120]
        for e in research.evidence
        if e.similarity_score >= 0.5
    ][:3] or ["Insufficient high-confidence evidence available."]

    risks = [f.message for f in risk.flags] or ["No material risk flags triggered."]

    signal, confidence, rationale = _deterministic_recommendation(ctx)
    if status == "partial":
        confidence = min(confidence, _PARTIAL_CONFIDENCE_CAP)

    return InvestmentMemoSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        question=question,
        thesis=thesis,
        catalysts=catalysts,
        risks=risks,
        action=ActionSignalSchema(
            signal=signal,
            confidence=confidence,
            rationale=rationale,
        ),
        citations=citations,
        risk_level=risk.risk_level,
        risk_score=risk.risk_score,
        writer_mode=mode,
        evidence_status=status,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agents.research_agent import run_research
    from agents.trend_agent import run_trend
    from agents.sentiment_agent import run_sentiment
    from agents.risk_agent import run_risk

    print(f"Running analyst memo for {DEFAULT_TICKER} …\n")

    research = run_research(DEFAULT_TICKER, question="What are the key catalysts and risks?")
    trend = run_trend(DEFAULT_TICKER, mode="live")
    sentiment = run_sentiment(DEFAULT_TICKER)
    risk = run_risk(DEFAULT_TICKER, mode="live")

    memo = run_analyst_memo(
        ticker=DEFAULT_TICKER,
        research=research,
        trend=trend,
        sentiment=sentiment,
        risk=risk,
    )

    print(f"Ticker     : {memo.ticker}")
    print(f"Thesis     : {memo.thesis}")
    print(f"Action     : {memo.action.signal} (confidence {memo.action.confidence:.0%})")
    print(f"Rationale  : {memo.action.rationale}")
    print(f"Risk level : {memo.risk_level} ({memo.risk_score:.0f}/100)")
    print(f"Citations  : {memo.citations}")
    print(f"Writer mode: {memo.writer_mode}")
