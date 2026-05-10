"""
Coordinator Agent – orchestrates the full analysis pipeline.

Execution DAG (parallel branches shown with →):
    gather(research → trend)
        ↓ both done
    sentiment
        ↓ (trend already complete from gather)
    risk
        ↓
    gather(analyst → debate)          [debate optional]
        ↓ both done
    memory

Each step is isolated: failures fall back to a stub schema so one bad
agent never crashes the whole pipeline (except analyst, which is critical).
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from config import DEFAULT_TICKER
from core.singletons import get_store
from core.schemas import (
    FullAnalysisSchema,
    InvestmentMemoSchema,
    MemoryComparisonSchema,
    ResearchOutputSchema,
    RiskOutputSchema,
    SentimentOutputSchema,
    TrendOutputSchema,
)
from agents.research_agent import run_research
from agents.trend_agent import run_trend
from agents.sentiment_agent import run_sentiment
from agents.risk_agent import run_risk
from agents.debate_agent import run_debate
from agents.analyst_agent import run_analyst_memo
from agents.memory_agent import save_memo, compare_to_last


# ── Helpers ──────────────────────────────────────────────────────────────────


def _trace(step: str, detail: str, elapsed: float) -> str:
    return f"[{step}] {detail} -- {elapsed:.2f}s"


async def _guarded(coro, fallback):
    """Await *coro*; on exception return (*fallback*, elapsed, exception)."""
    t = time.time()
    try:
        result = await coro
        return result, time.time() - t, None
    except Exception as exc:
        return fallback, time.time() - t, exc


# ── Async pipeline core (injectable for testing) ─────────────────────────────


async def _run_pipeline(
    ticker: str,
    question: str,
    mode: str,
    days_back: int,
    price_filepath: Optional[str],
    run_debate_flag: bool,
    *,
    _research_fn: Optional[Callable] = None,
    _trend_fn: Optional[Callable] = None,
    _sentiment_fn: Optional[Callable] = None,
    _risk_fn: Optional[Callable] = None,
    _analyst_fn: Optional[Callable] = None,
    _debate_fn: Optional[Callable] = None,
    _compare_fn: Optional[Callable] = None,
    _save_fn: Optional[Callable] = None,
) -> FullAnalysisSchema:
    """Async pipeline core. Inject async callables for testing.

    The shared VectorStoreManager singleton is retrieved once here and
    closed over by all default agent wrappers, eliminating redundant
    SentenceTransformer loads.
    """

    # ── Resolve shared store (one construction per pipeline run) ─────────────
    store = get_store()

    # ── Build default async wrappers (close over shared store) ──────────────
    async def _default_research(t, q, db, tk):
        return await asyncio.to_thread(run_research, t, q, days_back=db, top_k=tk, store=store)

    async def _default_trend(t, m, fp):
        return await asyncio.to_thread(run_trend, t, mode=m, filepath=fp)

    async def _default_sentiment(t, q, wd, tk):
        return await asyncio.to_thread(run_sentiment, t, question=q, window_days=wd, top_k=tk, store=store)

    async def _default_risk(t, m, fp, q, wd):
        return await asyncio.to_thread(run_risk, t, mode=m, price_filepath=fp, question=q, window_days=wd, store=store)

    async def _default_analyst(t, res, tr, se, ri, q):
        return await asyncio.to_thread(run_analyst_memo, ticker=t, research=res, trend=tr, sentiment=se, risk=ri, question=q)

    async def _default_debate(t, res, tr, se, ri):
        return await asyncio.to_thread(run_debate, t, res, tr, se, ri)

    async def _default_compare(memo):
        return await asyncio.to_thread(compare_to_last, memo)

    async def _default_save(memo):
        await asyncio.to_thread(save_memo, memo)

    # ── Resolve injected vs default callables ────────────────────────────────
    rf = _research_fn or _default_research
    tf = _trend_fn or _default_trend
    sf = _sentiment_fn or _default_sentiment
    rkf = _risk_fn or _default_risk
    af = _analyst_fn or _default_analyst
    df = _debate_fn or _default_debate
    cf = _compare_fn or _default_compare
    svf = _save_fn or _default_save

    pipeline_trace: list[str] = []
    total_start = time.time()
    ticker = ticker.upper().strip()

    # ── Stage 1: Research + Trend in parallel ────────────────────────────────
    _fallback_research = ResearchOutputSchema(
        ticker=ticker, question=question, days_back=days_back,
        evidence=[], summary="Research agent failed.",
    )
    _fallback_trend = TrendOutputSchema(
        ticker=ticker, mode=mode, as_of=datetime.now(timezone.utc),
        signals=[], summary="Trend agent failed.",
    )

    (research, r_t, r_err), (trend, t_t, t_err) = await asyncio.gather(
        _guarded(rf(ticker, question, days_back, 5), _fallback_research),
        _guarded(tf(ticker, mode, price_filepath), _fallback_trend),
    )

    if r_err:
        pipeline_trace.append(_trace("research", f"FAILED: {r_err}", r_t))
    else:
        ev_count = len(research.evidence)
        fb = " (fallback)" if "fallback" in research.summary.lower() else ""
        pipeline_trace.append(_trace("research", f"{ev_count} doc(s) retrieved{fb}", r_t))

    if t_err:
        pipeline_trace.append(_trace("trend", f"FAILED: {t_err}", t_t))
    else:
        sig_30d = next((s for s in trend.signals if s.horizon == "30d"), None)
        td = (
            f"{sig_30d.trend_label} {sig_30d.return_pct:.1f}% 30d"
            if sig_30d else "no 30d signal"
        )
        pipeline_trace.append(_trace("trend", td, t_t))

    # ── Stage 2: Sentiment (needs research) ──────────────────────────────────
    _fallback_sentiment = SentimentOutputSchema(
        ticker=ticker, as_of=datetime.now(timezone.utc), window_days=days_back,
        overall_score=0.0, overall_label="neutral",
        items=[], summary="Sentiment agent failed.",
    )

    t = time.time()
    sentiment, s_t, s_err = await _guarded(
        sf(ticker, question, days_back, 5), _fallback_sentiment
    )

    if s_err:
        pipeline_trace.append(_trace("sentiment", f"FAILED: {s_err}", s_t))
    else:
        pipeline_trace.append(
            _trace(
                "sentiment",
                f"{sentiment.overall_label} (score {sentiment.overall_score:.2f})"
                f" from {len(sentiment.items)} item(s)",
                s_t,
            )
        )

    # ── Stage 3: Risk (needs sentiment; trend already complete) ──────────────
    _fallback_risk = RiskOutputSchema(
        ticker=ticker, as_of=datetime.now(timezone.utc),
        risk_score=50.0, risk_level="moderate",
        flags=[], summary="Risk agent failed.",
    )

    risk, rk_t, rk_err = await _guarded(
        rkf(ticker, mode, price_filepath, question, days_back), _fallback_risk
    )

    if rk_err:
        pipeline_trace.append(_trace("risk", f"FAILED: {rk_err}", rk_t))
    else:
        pipeline_trace.append(
            _trace(
                "risk",
                f"{risk.risk_level} ({risk.risk_score:.0f}/100) {len(risk.flags)} flag(s)",
                rk_t,
            )
        )

    # ── Stage 4: Analyst + Debate in parallel ────────────────────────────────
    analyst_coro = af(ticker, research, trend, sentiment, risk, question)

    if run_debate_flag:
        debate_coro = df(ticker, research, trend, sentiment, risk)

        (memo, an_t, an_err), (debate, db_t, db_err) = await asyncio.gather(
            _guarded(analyst_coro, None),
            _guarded(debate_coro, None),
        )
    else:
        memo, an_t, an_err = await _guarded(analyst_coro, None)
        debate, db_t, db_err = None, 0.0, None

    if an_err or memo is None:
        raise RuntimeError(f"Analyst agent failed critically: {an_err}")

    if run_debate_flag:
        if db_err:
            pipeline_trace.append(_trace("debate", f"FAILED: {db_err}", db_t))
        else:
            pipeline_trace.append(
                _trace(
                    "debate",
                    f"bias={debate.final_bias} | "
                    f"bull={debate.bull.confidence:.2f} "
                    f"bear={debate.bear.confidence:.2f}",
                    db_t,
                )
            )
        if debate is not None:
            memo.debate = debate

    pipeline_trace.append(
        _trace(
            "analyst",
            f"memo generated (writer={memo.writer_mode}) | "
            f"action={memo.action.signal} conf={memo.action.confidence:.2f}",
            an_t,
        )
    )

    # ── Stage 5: Memory (runs last) ──────────────────────────────────────────
    t = time.time()
    memory: Optional[MemoryComparisonSchema] = None
    try:
        memory = await cf(memo)
        memo.memory = memory
        await svf(memo)
        pipeline_trace.append(_trace("memory", memory.summary[:80], time.time() - t))
    except Exception as e:
        pipeline_trace.append(_trace("memory", f"FAILED: {e}", time.time() - t))

    total_runtime = time.time() - total_start
    pipeline_trace.append(f"[total] {total_runtime:.2f}s")

    return FullAnalysisSchema(
        ticker=ticker,
        mode=mode,
        as_of=datetime.now(timezone.utc),
        memo=memo,
        research=research,
        trend=trend,
        sentiment=sentiment,
        risk=risk,
        debate=debate if run_debate_flag else None,
        memory=memory,
        pipeline_trace=pipeline_trace,
        total_runtime_seconds=total_runtime,
    )


# ── Public sync entry-point (preserves original API) ────────────────────────


def run_full_analysis(
    ticker: str,
    question: str = "What are the key catalysts and risks?",
    mode: str = "live",
    days_back: int = 365,
    price_filepath: Optional[str] = None,
    run_debate_flag: bool = True,
) -> FullAnalysisSchema:
    """Run the full agent pipeline and return a FullAnalysisSchema.

    Internally async; safe to call from synchronous contexts (FastAPI sync
    routes, CLI, etc.).  Creates a fresh event loop via asyncio.run().
    """
    return asyncio.run(
        _run_pipeline(
            ticker=ticker,
            question=question,
            mode=mode,
            days_back=days_back,
            price_filepath=price_filepath,
            run_debate_flag=run_debate_flag,
        )
    )


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import DEFAULT_TICKER

    print(f"Running full analysis for {DEFAULT_TICKER}...\n")

    result = run_full_analysis(
        ticker=DEFAULT_TICKER,
        mode="live",
        run_debate_flag=True,
    )

    print("=== PIPELINE TRACE ===")
    for step in result.pipeline_trace:
        print(f"  {step}")

    print(f"\n=== MEMO ===")
    print(f"Ticker:     {result.memo.ticker}")
    print(f"Action:     {result.memo.action.signal.upper()}")
    print(f"Confidence: {result.memo.action.confidence:.2f}")
    print(f"Risk:       {result.risk.risk_level} ({result.risk.risk_score:.0f}/100)")
    print(f"Writer:     {result.memo.writer_mode}")
    print(f"\nThesis (first 300 chars):")
    print(f"  {result.memo.thesis[:300]}")

    if result.debate:
        print(f"\n=== DEBATE ===")
        print(f"Bias:    {result.debate.final_bias}")
        print(f"Verdict: {result.debate.coordinator_verdict[:120]}")

    if result.memory:
        print(f"\n=== MEMORY ===")
        print(f"  {result.memory.summary}")

    print(f"\nTotal runtime: {result.total_runtime_seconds:.2f}s")
