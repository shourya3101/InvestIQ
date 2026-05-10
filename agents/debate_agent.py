"""
Debate Agent – Bull vs Bear LLM debate with deterministic fallback.

Two LLM personas (Bull analyst, Bear analyst) independently argue their
case using the same retrieved evidence.  A Coordinator then reads both
arguments and delivers a verdict.  When LLM_MODE == "off" or any LLM
call fails, the agent falls back to a fully deterministic rule-based
debate so the pipeline never crashes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from config import LLM_MODE, GROQ_MODEL, CLAUDE_MODEL, OPENAI_MODEL
from core.utils import parse_llm_json
from core.schemas import (
    DebateOutputSchema,
    DebateArgumentSchema,
    ResearchOutputSchema,
    TrendOutputSchema,
    SentimentOutputSchema,
    RiskOutputSchema,
)
from llm.providers import GroqProvider, ClaudeProvider, LLMRouter, OpenAIProvider


# ── Helpers ─────────────────────────────────────────────────────────


def _build_evidence_block(
    research: ResearchOutputSchema, max_per_item: int = 400
) -> str:
    """Format the evidence pack into a readable text block for LLM prompts.

    Returns a multi-line string like:
        E1 (2025-01-25, bloomberg_export): <snippet>
        E2 (unknown, AAPL_news.csv): <snippet>

    Returns "No evidence available." when the evidence list is empty.
    """
    if not research.evidence:
        return "No evidence available."

    lines: list[str] = []
    for ev in research.evidence:
        date_str = ev.date.strftime("%Y-%m-%d") if ev.date else "unknown"
        source_str = ev.source if ev.source else "unknown"
        snippet = ev.snippet[:max_per_item]
        lines.append(f"{ev.citation_id} ({date_str}, {source_str}): {snippet}")
    return "\n".join(lines)


def _get_30d_signal(trend: TrendOutputSchema):
    """Return the 30d trend signal, falling back to 7d then first available.

    Returns None if trend.signals is empty.
    """
    if not trend.signals:
        return None
    for horizon in ("30d", "7d"):
        sig = next((s for s in trend.signals if s.horizon == horizon), None)
        if sig is not None:
            return sig
    return trend.signals[0]


# ── Deterministic fallback ──────────────────────────────────────────


def _deterministic_debate(
    ticker: str,
    signal,
    sentiment: SentimentOutputSchema,
    risk: RiskOutputSchema,
    research: ResearchOutputSchema,
) -> DebateOutputSchema:
    """Build a bull/bear debate using only deterministic rules (no LLM).

    Always returns a valid DebateOutputSchema, even with zero evidence.
    """
    # ── Bull arguments ──────────────────────────────────────────────
    bull_args: list[str] = []
    if signal and signal.return_pct > 2:
        bull_args.append(
            f"30d return of {signal.return_pct:.1f}% shows positive price momentum"
        )
    if sentiment.overall_label == "positive":
        bull_args.append(
            f"News sentiment is positive (score {sentiment.overall_score:.2f})"
        )
    if risk.risk_score < 40:
        bull_args.append(
            f"Low risk score of {risk.risk_score:.0f}/100 supports position entry"
        )
    if signal and signal.trend_label == "bullish":
        bull_args.append(
            "Current trend is classified as bullish across the analysis window"
        )
    bull_args.append("Monitor upcoming catalysts for confirmation")

    if len(bull_args) < 2:
        bull_args.insert(0, "Insufficient bullish signals at this time")

    # ── Bear arguments ──────────────────────────────────────────────
    bear_args: list[str] = []
    if signal and signal.max_drawdown_pct < -8:
        bear_args.append(
            f"Max drawdown of {signal.max_drawdown_pct:.1f}% indicates significant price weakness"
        )
    if sentiment.overall_label == "negative":
        bear_args.append(
            f"Negative news sentiment detected (score {sentiment.overall_score:.2f})"
        )
    if risk.risk_score > 60:
        bear_args.append(
            f"Elevated risk score of {risk.risk_score:.0f}/100 warrants caution"
        )
    if signal and signal.volatility_pct > 30:
        bear_args.append(
            f"Annualized volatility of {signal.volatility_pct:.1f}% indicates uncertain price action"
        )
    bear_args.append("Macro environment uncertainty requires careful position sizing")

    if len(bear_args) < 2:
        bear_args.insert(0, "Insufficient bearish signals at this time")

    # ── Confidence scoring ──────────────────────────────────────────
    bull_confidence = 0.5
    if signal and signal.return_pct > 3:
        bull_confidence += 0.1
    if sentiment.overall_label == "positive":
        bull_confidence += 0.1
    if risk.risk_score < 40:
        bull_confidence += 0.1
    if risk.risk_score > 60:
        bull_confidence -= 0.1
    bull_confidence = max(0.2, min(0.85, bull_confidence))

    bear_confidence = 1.0 - bull_confidence
    bear_confidence = max(0.2, min(0.85, bear_confidence))

    # ── Bias / verdict ──────────────────────────────────────────────
    if bull_confidence > 0.6:
        final_bias = "bullish"
    elif bear_confidence > 0.6:
        final_bias = "bearish"
    else:
        final_bias = "neutral"

    coordinator_verdict = (
        f"Deterministic analysis: {final_bias} bias based on "
        f"{signal.trend_label if signal else 'unknown'} trend, "
        f"{sentiment.overall_label} sentiment, and risk score "
        f"{risk.risk_score:.0f}/100."
    )

    memo_update = f"Debate verdict: {final_bias}. {coordinator_verdict[:100]}"

    # ── Evidence citations ──────────────────────────────────────────
    key_evidence = [e.citation_id for e in research.evidence[:2]]

    return DebateOutputSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        bull=DebateArgumentSchema(
            side="bull",
            arguments=bull_args,
            confidence=bull_confidence,
            key_evidence=key_evidence,
        ),
        bear=DebateArgumentSchema(
            side="bear",
            arguments=bear_args,
            confidence=bear_confidence,
            key_evidence=key_evidence,
        ),
        coordinator_verdict=coordinator_verdict,
        final_bias=final_bias,
        memo_update=memo_update,
    )


# ── Main entry-point ────────────────────────────────────────────────


def run_debate(
    ticker: str,
    research: ResearchOutputSchema,
    trend: TrendOutputSchema,
    sentiment: SentimentOutputSchema,
    risk: RiskOutputSchema,
) -> DebateOutputSchema:
    """Run a Bull-vs-Bear debate and return the coordinator's verdict.

    Uses LLM providers when configured; falls back to deterministic
    rules on any failure or when LLM_MODE == "off".
    """
    signal = _get_30d_signal(trend)
    evidence_block = _build_evidence_block(research)

    # ── Fast path: deterministic ────────────────────────────────────
    if LLM_MODE == "off" or evidence_block == "No evidence available.":
        return _deterministic_debate(ticker, signal, sentiment, risk, research)

    # ── LLM path ────────────────────────────────────────────────────
    try:
        # Set up provider
        if LLM_MODE == "groq":
            provider = GroqProvider(model=GROQ_MODEL)
        elif LLM_MODE == "claude":
            provider = ClaudeProvider(model=CLAUDE_MODEL)
        elif LLM_MODE == "openai":
            provider = OpenAIProvider(model=OPENAI_MODEL)
        elif LLM_MODE == "auto":
            provider = LLMRouter(
                primary=GroqProvider(model=GROQ_MODEL),
                fallback=ClaudeProvider(model=CLAUDE_MODEL),
            )
        else:
            return _deterministic_debate(ticker, signal, sentiment, risk, research)

        # Shared context for prompts
        trend_label = signal.trend_label if signal else "unknown"
        return_pct = f"{signal.return_pct:.1f}" if signal else "N/A"
        vol_pct = f"{signal.volatility_pct:.1f}" if signal else "N/A"

        data_user_block = (
            f"Ticker: {ticker}\n"
            f"Trend: {trend_label}, {return_pct}% 30d return\n"
            f"Sentiment: {sentiment.overall_label} ({sentiment.overall_score:.2f})\n"
            f"Risk score: {risk.risk_score:.0f}/100 ({risk.risk_level})\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            f'Output this exact JSON:\n'
            f'{{\n'
            f'  "arguments": ["point 1", "point 2", "point 3"],\n'
            f'  "confidence": 0.65,\n'
            f'  "key_evidence": ["E1", "E2"]\n'
            f'}}'
        )

        # ── LLM CALL 1: Bull Agent ─────────────────────────────────
        bull_system = (
            "You are an aggressive buy-side equity analyst. "
            "Your job is to make the strongest possible BULL case "
            "for the stock. Use ONLY the evidence provided. "
            "Be specific and cite evidence IDs. "
            "Output ONLY valid JSON with no markdown fences."
        )
        bull_raw = provider.generate(bull_system, data_user_block)
        bull_parsed = parse_llm_json(bull_raw, ["arguments", "confidence", "key_evidence"])

        if bull_parsed:
            bull_args = bull_parsed["arguments"]
            bull_conf = max(0.2, min(0.85, float(bull_parsed["confidence"])))
            bull_evidence = bull_parsed["key_evidence"]
        else:
            # Deterministic bull fallback
            det = _deterministic_debate(ticker, signal, sentiment, risk, research)
            bull_args = det.bull.arguments
            bull_conf = det.bull.confidence
            bull_evidence = det.bull.key_evidence

        # ── LLM CALL 2: Bear Agent ─────────────────────────────────
        bear_system = (
            "You are a skeptical risk-focused analyst. "
            "Your job is to make the strongest possible BEAR case "
            "for the stock. Use ONLY the evidence provided. "
            "Be specific and cite evidence IDs. "
            "Output ONLY valid JSON with no markdown fences."
        )
        bear_raw = provider.generate(bear_system, data_user_block)
        bear_parsed = parse_llm_json(bear_raw, ["arguments", "confidence", "key_evidence"])

        if bear_parsed:
            bear_args = bear_parsed["arguments"]
            bear_conf = max(0.2, min(0.85, float(bear_parsed["confidence"])))
            bear_evidence = bear_parsed["key_evidence"]
        else:
            det = _deterministic_debate(ticker, signal, sentiment, risk, research)
            bear_args = det.bear.arguments
            bear_conf = det.bear.confidence
            bear_evidence = det.bear.key_evidence

        # ── LLM CALL 3: Coordinator ────────────────────────────────
        coord_system = (
            "You are a senior portfolio manager. "
            "Read the bull and bear arguments and decide "
            "which case is stronger given the data. "
            "Output ONLY valid JSON with no markdown fences."
        )
        coord_user = (
            f"Ticker: {ticker}\n\n"
            f"BULL CASE (confidence {bull_conf:.2f}):\n"
            + "\n".join(f"- {a}" for a in bull_args)
            + f"\n\nBEAR CASE (confidence {bear_conf:.2f}):\n"
            + "\n".join(f"- {a}" for a in bear_args)
            + f"\n\nSupporting data:\n"
            f"- 30d return: {return_pct}% ({trend_label})\n"
            f"- Volatility: {vol_pct}%\n"
            f"- Risk score: {risk.risk_score:.0f}/100 ({risk.risk_level})\n"
            f"- Sentiment: {sentiment.overall_label}\n\n"
            f'Output this exact JSON:\n'
            f'{{\n'
            f'  "coordinator_verdict": "2-3 sentence verdict explaining '
            f'which side is stronger and why",\n'
            f'  "final_bias": "bullish",\n'
            f'  "memo_update": "one sentence summary for investment memo"\n'
            f'}}\n'
            f'final_bias must be exactly one of: bullish, bearish, neutral'
        )

        coord_raw = provider.generate(coord_system, coord_user)
        coord_parsed = parse_llm_json(
            coord_raw, ["coordinator_verdict", "final_bias", "memo_update"]
        )

        if coord_parsed:
            coordinator_verdict = coord_parsed["coordinator_verdict"]
            final_bias = coord_parsed["final_bias"]
            memo_update = coord_parsed["memo_update"]
        else:
            # Deterministic coordinator fallback
            if bull_conf > 0.6:
                final_bias = "bullish"
            elif bear_conf > 0.6:
                final_bias = "bearish"
            else:
                final_bias = "neutral"
            coordinator_verdict = (
                f"LLM coordinator parse failed; deterministic fallback: "
                f"{final_bias} bias based on {trend_label} trend, "
                f"{sentiment.overall_label} sentiment, "
                f"risk {risk.risk_score:.0f}/100."
            )
            memo_update = f"Debate verdict: {final_bias}. {coordinator_verdict[:100]}"

        # Validate final_bias
        if final_bias not in ("bullish", "bearish", "neutral"):
            final_bias = "neutral"

        return DebateOutputSchema(
            ticker=ticker,
            as_of=datetime.now(timezone.utc),
            bull=DebateArgumentSchema(
                side="bull",
                arguments=bull_args,
                confidence=bull_conf,
                key_evidence=bull_evidence,
            ),
            bear=DebateArgumentSchema(
                side="bear",
                arguments=bear_args,
                confidence=bear_conf,
                key_evidence=bear_evidence,
            ),
            coordinator_verdict=coordinator_verdict,
            final_bias=final_bias,
            memo_update=memo_update,
        )

    except Exception:
        return _deterministic_debate(ticker, signal, sentiment, risk, research)


# ── Smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from agents.research_agent import run_research
    from agents.trend_agent import run_trend
    from agents.sentiment_agent import run_sentiment
    from agents.risk_agent import run_risk
    from config import DEFAULT_TICKER

    print(f"Running debate for {DEFAULT_TICKER}...")

    research = run_research(DEFAULT_TICKER, "What are the key catalysts and risks?")
    trend = run_trend(DEFAULT_TICKER, mode="live")
    sentiment = run_sentiment(DEFAULT_TICKER)
    risk = run_risk(DEFAULT_TICKER, mode="live")

    result = run_debate(DEFAULT_TICKER, research, trend, sentiment, risk)

    print(f"\nFinal bias: {result.final_bias}")
    print(f"Verdict: {result.coordinator_verdict}")
    print(f"\nBull (confidence {result.bull.confidence:.2f}):")
    for arg in result.bull.arguments:
        print(f"  + {arg}")
    print(f"\nBear (confidence {result.bear.confidence:.2f}):")
    for arg in result.bear.arguments:
        print(f"  - {arg}")
    print(f"\nMemo update: {result.memo_update}")
