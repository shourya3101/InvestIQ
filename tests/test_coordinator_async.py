"""
Tests for the async coordinator refactor — written BEFORE implementation (TDD).

Ordering proofs use two techniques:
  - asyncio.Event deadlock: if two coroutines mutually wait for each other's
    start event, asyncio.gather() is the ONLY way they can both complete.
    Sequential execution deadlocks and asyncio.wait_for raises TimeoutError.
  - call_log ordering: append markers at entry/exit of each fake agent, then
    assert relative positions in the log.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.coordinator_agent import _run_pipeline
from core.schemas import (
    ActionSignalSchema,
    DebateArgumentSchema,
    DebateOutputSchema,
    FullAnalysisSchema,
    InvestmentMemoSchema,
    MemoryComparisonSchema,
    ResearchOutputSchema,
    RiskOutputSchema,
    SentimentOutputSchema,
    TrendOutputSchema,
)


# ── Fixture factories ─────────────────────────────────────────────────────────


def _research(ticker="TEST"):
    return ResearchOutputSchema(
        ticker=ticker, question="q", days_back=30, evidence=[], summary="ok"
    )


def _trend(ticker="TEST"):
    return TrendOutputSchema(
        ticker=ticker,
        mode="live",
        as_of=datetime.now(timezone.utc),
        signals=[],
        summary="ok",
    )


def _sentiment(ticker="TEST"):
    return SentimentOutputSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        window_days=30,
        overall_score=0.0,
        overall_label="neutral",
        items=[],
        summary="ok",
    )


def _risk(ticker="TEST"):
    return RiskOutputSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        risk_score=50.0,
        risk_level="moderate",
        flags=[],
        summary="ok",
    )


def _memo(ticker="TEST"):
    return InvestmentMemoSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        question="q",
        thesis="t",
        catalysts=[],
        risks=[],
        action=ActionSignalSchema(signal="hold", confidence=0.5, rationale="r"),
        citations=[],
        risk_level="moderate",
        risk_score=50.0,
        writer_mode="deterministic",
    )


def _debate(ticker="TEST"):
    return DebateOutputSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        bull=DebateArgumentSchema(
            side="bull", arguments=[], confidence=0.6, key_evidence=[]
        ),
        bear=DebateArgumentSchema(
            side="bear", arguments=[], confidence=0.4, key_evidence=[]
        ),
        coordinator_verdict="balanced",
        final_bias="neutral",
        memo_update="",
    )


def _memory(ticker="TEST"):
    return MemoryComparisonSchema(
        ticker=ticker,
        current_as_of=datetime.now(timezone.utc),
        signal_changed=False,
        thesis_changed=False,
        summary="no prior",
    )


def _minimal_fakes(call_log=None, *, include_debate=True):
    """Return a dict of minimal async fake agents that record to call_log."""

    log = call_log if call_log is not None else []

    async def fake_research(ticker, question, days_back, top_k):
        log.append("research_end")
        return _research(ticker)

    async def fake_trend(ticker, mode, filepath):
        log.append("trend_end")
        return _trend(ticker)

    async def fake_sentiment(ticker, question, window_days, top_k):
        log.append("sentiment_start")
        log.append("sentiment_end")
        return _sentiment(ticker)

    async def fake_risk(ticker, mode, price_filepath, question, window_days):
        log.append("risk_start")
        log.append("risk_end")
        return _risk(ticker)

    async def fake_analyst(ticker, research, trend, sentiment, risk, question):
        log.append("analyst_start")
        log.append("analyst_end")
        return _memo(ticker)

    async def fake_debate(ticker, research, trend, sentiment, risk):
        log.append("debate_start")
        log.append("debate_end")
        return _debate(ticker)

    async def fake_compare(memo):
        log.append("memory_start")
        return _memory(memo.ticker)

    async def fake_save(memo):
        pass

    return dict(
        _research_fn=fake_research,
        _trend_fn=fake_trend,
        _sentiment_fn=fake_sentiment,
        _risk_fn=fake_risk,
        _analyst_fn=fake_analyst,
        _debate_fn=fake_debate if include_debate else None,
        _compare_fn=fake_compare,
        _save_fn=fake_save,
    )


def _run(coro, timeout=5.0):
    """Run an async coroutine from a sync test, with a hard timeout.

    Patches get_store so the coordinator tests never trigger a real
    SentenceTransformer load (which would exceed the deadline).
    The coroutine body runs when awaited, so the patch is active in time.
    """
    async def _wrapper():
        return await asyncio.wait_for(coro, timeout=timeout)

    with patch("agents.coordinator_agent.get_store", return_value=MagicMock()):
        return asyncio.run(_wrapper())


# ── Concurrency proof ─────────────────────────────────────────────────────────


def test_research_and_trend_run_concurrently():
    """asyncio.Event mutual-wait: completes only if gather runs them in parallel."""

    async def _inner():
        research_started = asyncio.Event()
        trend_started = asyncio.Event()

        async def fake_research(ticker, question, days_back, top_k):
            research_started.set()
            await trend_started.wait()  # deadlocks if sequential
            return _research(ticker)

        async def fake_trend(ticker, mode, filepath):
            trend_started.set()
            await research_started.wait()  # deadlocks if sequential
            return _trend(ticker)

        fakes = _minimal_fakes()
        fakes["_research_fn"] = fake_research
        fakes["_trend_fn"] = fake_trend

        await asyncio.wait_for(
            _run_pipeline("TEST", "q", "live", 30, None, False, **fakes),
            timeout=5.0,
        )

    with patch("agents.coordinator_agent.get_store", return_value=MagicMock()):
        asyncio.run(_inner())


# ── Ordering constraints ──────────────────────────────────────────────────────


def test_sentiment_starts_only_after_research_completes():
    call_log = []

    async def fake_research(ticker, question, days_back, top_k):
        await asyncio.sleep(0)  # yield so other tasks can interleave
        call_log.append("research_end")
        return _research(ticker)

    async def fake_sentiment(ticker, question, window_days, top_k):
        call_log.append("sentiment_start")
        return _sentiment(ticker)

    fakes = _minimal_fakes(call_log)
    fakes["_research_fn"] = fake_research
    fakes["_sentiment_fn"] = fake_sentiment

    _run(_run_pipeline("TEST", "q", "live", 30, None, False, **fakes))

    assert call_log.index("research_end") < call_log.index("sentiment_start"), (
        f"Expected research_end before sentiment_start in {call_log}"
    )


def test_risk_starts_only_after_both_trend_and_sentiment_complete():
    call_log = []

    async def fake_trend(ticker, mode, filepath):
        await asyncio.sleep(0)
        call_log.append("trend_end")
        return _trend(ticker)

    async def fake_sentiment(ticker, question, window_days, top_k):
        await asyncio.sleep(0)
        call_log.append("sentiment_end")
        return _sentiment(ticker)

    async def fake_risk(ticker, mode, price_filepath, question, window_days):
        call_log.append("risk_start")
        return _risk(ticker)

    fakes = _minimal_fakes(call_log)
    fakes["_trend_fn"] = fake_trend
    fakes["_sentiment_fn"] = fake_sentiment
    fakes["_risk_fn"] = fake_risk

    _run(_run_pipeline("TEST", "q", "live", 30, None, False, **fakes))

    risk_pos = call_log.index("risk_start")
    assert call_log.index("trend_end") < risk_pos, (
        f"trend_end must precede risk_start: {call_log}"
    )
    assert call_log.index("sentiment_end") < risk_pos, (
        f"sentiment_end must precede risk_start: {call_log}"
    )


def test_analyst_starts_only_after_risk_completes():
    call_log = []

    async def fake_risk(ticker, mode, price_filepath, question, window_days):
        await asyncio.sleep(0)
        call_log.append("risk_end")
        return _risk(ticker)

    async def fake_analyst(ticker, research, trend, sentiment, risk, question):
        call_log.append("analyst_start")
        return _memo(ticker)

    fakes = _minimal_fakes(call_log)
    fakes["_risk_fn"] = fake_risk
    fakes["_analyst_fn"] = fake_analyst

    _run(_run_pipeline("TEST", "q", "live", 30, None, False, **fakes))

    assert call_log.index("risk_end") < call_log.index("analyst_start"), (
        f"Expected risk_end before analyst_start in {call_log}"
    )


def test_debate_starts_only_after_risk_completes():
    call_log = []

    async def fake_risk(ticker, mode, price_filepath, question, window_days):
        await asyncio.sleep(0)
        call_log.append("risk_end")
        return _risk(ticker)

    async def fake_debate(ticker, research, trend, sentiment, risk):
        call_log.append("debate_start")
        return _debate(ticker)

    fakes = _minimal_fakes(call_log)
    fakes["_risk_fn"] = fake_risk
    fakes["_debate_fn"] = fake_debate

    _run(_run_pipeline("TEST", "q", "live", 30, None, True, **fakes))

    assert call_log.index("risk_end") < call_log.index("debate_start"), (
        f"Expected risk_end before debate_start in {call_log}"
    )


def test_analyst_and_debate_run_concurrently():
    """Mutual-wait proof that analyst and debate run in parallel after risk."""

    async def _inner():
        analyst_started = asyncio.Event()
        debate_started = asyncio.Event()

        async def fake_analyst(ticker, research, trend, sentiment, risk, question):
            analyst_started.set()
            await debate_started.wait()
            return _memo(ticker)

        async def fake_debate(ticker, research, trend, sentiment, risk):
            debate_started.set()
            await analyst_started.wait()
            return _debate(ticker)

        fakes = _minimal_fakes()
        fakes["_analyst_fn"] = fake_analyst
        fakes["_debate_fn"] = fake_debate

        await asyncio.wait_for(
            _run_pipeline("TEST", "q", "live", 30, None, True, **fakes),
            timeout=5.0,
        )

    with patch("agents.coordinator_agent.get_store", return_value=MagicMock()):
        asyncio.run(_inner())


def test_memory_starts_only_after_analyst_completes():
    call_log = []

    async def fake_analyst(ticker, research, trend, sentiment, risk, question):
        await asyncio.sleep(0)
        call_log.append("analyst_end")
        return _memo(ticker)

    async def fake_compare(memo):
        call_log.append("memory_start")
        return _memory(memo.ticker)

    fakes = _minimal_fakes(call_log)
    fakes["_analyst_fn"] = fake_analyst
    fakes["_compare_fn"] = fake_compare

    _run(_run_pipeline("TEST", "q", "live", 30, None, False, **fakes))

    assert call_log.index("analyst_end") < call_log.index("memory_start"), (
        f"Expected analyst_end before memory_start in {call_log}"
    )


# ── Output schema ─────────────────────────────────────────────────────────────


def test_returns_full_analysis_schema():
    result = _run(
        _run_pipeline("AAPL", "q", "live", 30, None, False, **_minimal_fakes())
    )
    assert isinstance(result, FullAnalysisSchema)


def test_ticker_is_normalised_to_uppercase():
    result = _run(
        _run_pipeline("aapl", "q", "live", 30, None, False, **_minimal_fakes())
    )
    assert result.ticker == "AAPL"


def test_pipeline_trace_contains_all_agent_names():
    result = _run(
        _run_pipeline("TEST", "q", "live", 30, None, True, **_minimal_fakes())
    )
    trace_text = " ".join(result.pipeline_trace)
    for step in ("research", "trend", "sentiment", "risk", "analyst", "debate", "memory"):
        assert step in trace_text, f"'{step}' missing from pipeline trace: {result.pipeline_trace}"


def test_debate_absent_from_result_when_flag_is_false():
    result = _run(
        _run_pipeline("TEST", "q", "live", 30, None, False, **_minimal_fakes())
    )
    assert result.debate is None


def test_debate_present_in_result_when_flag_is_true():
    result = _run(
        _run_pipeline("TEST", "q", "live", 30, None, True, **_minimal_fakes())
    )
    assert result.debate is not None
    assert isinstance(result.debate, DebateOutputSchema)


def test_debate_attached_to_memo_when_flag_is_true():
    result = _run(
        _run_pipeline("TEST", "q", "live", 30, None, True, **_minimal_fakes())
    )
    assert result.memo.debate is not None


# ── Resilience ────────────────────────────────────────────────────────────────


def test_research_failure_produces_fallback_and_pipeline_continues():
    async def exploding_research(ticker, question, days_back, top_k):
        raise RuntimeError("research kaboom")

    fakes = _minimal_fakes()
    fakes["_research_fn"] = exploding_research

    result = _run(_run_pipeline("TEST", "q", "live", 30, None, False, **fakes))

    assert isinstance(result, FullAnalysisSchema)
    trace_text = " ".join(result.pipeline_trace)
    assert "FAILED" in trace_text


def test_trend_failure_produces_fallback_and_pipeline_continues():
    async def exploding_trend(ticker, mode, filepath):
        raise RuntimeError("trend kaboom")

    fakes = _minimal_fakes()
    fakes["_trend_fn"] = exploding_trend

    result = _run(_run_pipeline("TEST", "q", "live", 30, None, False, **fakes))

    assert isinstance(result, FullAnalysisSchema)
    trace_text = " ".join(result.pipeline_trace)
    assert "FAILED" in trace_text


def test_memory_failure_does_not_crash_pipeline():
    async def exploding_compare(memo):
        raise RuntimeError("memory kaboom")

    fakes = _minimal_fakes()
    fakes["_compare_fn"] = exploding_compare

    result = _run(_run_pipeline("TEST", "q", "live", 30, None, False, **fakes))

    assert isinstance(result, FullAnalysisSchema)
    assert result.memory is None


def test_total_runtime_seconds_is_positive():
    result = _run(
        _run_pipeline("TEST", "q", "live", 30, None, False, **_minimal_fakes())
    )
    assert result.total_runtime_seconds > 0
