"""End-to-end status propagation through _run_pipeline and stream_pipeline_events."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from agents.coordinator_agent import _run_pipeline, stream_pipeline_events
from core.schemas import (
    ActionSignalSchema,
    InvestmentMemoSchema,
    MemoryComparisonSchema,
    ResearchOutputSchema,
    RiskOutputSchema,
    SentimentOutputSchema,
    TrendOutputSchema,
    TrendSignalSchema,
)

NOW = datetime.now(timezone.utc)


def _research(status):
    return ResearchOutputSchema(ticker="TSLA", question="q", evidence=[],
                                summary="s", evidence_status=status,
                                status_reason="reason")

def _trend():
    return TrendOutputSchema(ticker="TSLA", mode="live", as_of=NOW,
                             signals=[TrendSignalSchema(horizon="30d", return_pct=0.0,
                                                        volatility_pct=1.0,
                                                        max_drawdown_pct=0.0,
                                                        trend_label="neutral")],
                             summary="s")

def _sentiment():
    return SentimentOutputSchema(ticker="TSLA", as_of=NOW, window_days=30,
                                 overall_score=0.0, overall_label="neutral",
                                 items=[], summary="s", data_status="no_data")

def _risk():
    return RiskOutputSchema(ticker="TSLA", as_of=NOW, risk_score=50.0,
                            risk_level="moderate", flags=[], summary="s")

def _memo(status="insufficient"):
    return InvestmentMemoSchema(ticker="TSLA", as_of=NOW, question="q", thesis="t",
                                catalysts=[], risks=[],
                                action=ActionSignalSchema(signal="no_view",
                                                          confidence=0.0, rationale="r"),
                                citations=[], risk_level="moderate", risk_score=50.0,
                                writer_mode="off", evidence_status=status)

def _memory():
    return MemoryComparisonSchema(ticker="TSLA", current_as_of=NOW,
                                  signal_changed=False, thesis_changed=False,
                                  summary="first analysis")


def _fns(status, debate_mock, sentiment_spy=None, risk_spy=None):
    async def rf(t, q, db, tk):
        return _research(status)
    async def tf(t, m, fp):
        return _trend()
    async def sf(t, q, wd, tk, research):
        if sentiment_spy is not None:
            sentiment_spy.append(research)
        return _sentiment()
    async def rkf(t, m, fp, q, wd, trend, sentiment):
        if risk_spy is not None:
            risk_spy.append((trend, sentiment))
        return _risk()
    async def af(t, res, tr, se, ri, q):
        return _memo(status)
    async def df(t, res, tr, se, ri):
        debate_mock(t)
        raise AssertionError("debate must not run on insufficient evidence")
    async def cf(memo):
        return _memory()
    async def svf(memo):
        pass
    return dict(_research_fn=rf, _trend_fn=tf, _sentiment_fn=sf, _risk_fn=rkf,
                _analyst_fn=af, _debate_fn=df, _compare_fn=cf, _save_fn=svf)


def test_research_output_reaches_sentiment_and_risk():
    sentiment_spy, risk_spy = [], []
    fns = _fns("insufficient", MagicMock(), sentiment_spy, risk_spy)
    result = asyncio.run(_run_pipeline("TSLA", "q", "live", 30, None, False, **fns))
    assert sentiment_spy[0].evidence_status == "insufficient"
    trend_arg, sentiment_arg = risk_spy[0]
    assert trend_arg.ticker == "TSLA"
    assert sentiment_arg.data_status == "no_data"

def test_debate_skipped_on_insufficient_with_visible_reason():
    debate_mock = MagicMock()
    fns = _fns("insufficient", debate_mock)
    result = asyncio.run(_run_pipeline("TSLA", "q", "live", 30, None, True, **fns))
    debate_mock.assert_not_called()
    assert result.memo.debate_skipped_reason == "insufficient evidence"
    assert result.debate is None
    assert any("SKIPPED" in line for line in result.pipeline_trace if "[debate]" in line)

def test_debate_runs_normally_on_sufficient():
    ran = []
    async def df(t, res, tr, se, ri):
        ran.append(t)
        from core.schemas import DebateArgumentSchema, DebateOutputSchema
        arg = DebateArgumentSchema(side="bull", arguments=["a"], confidence=0.5,
                                   key_evidence=[])
        bear = DebateArgumentSchema(side="bear", arguments=["b"], confidence=0.5,
                                    key_evidence=[])
        return DebateOutputSchema(ticker=t, as_of=NOW, bull=arg, bear=bear,
                                  coordinator_verdict="v", final_bias="neutral",
                                  memo_update="u")
    fns = _fns("sufficient", MagicMock())
    fns["_debate_fn"] = df
    result = asyncio.run(_run_pipeline("TSLA", "q", "live", 30, None, True, **fns))
    assert ran == ["TSLA"]
    assert result.memo.debate_skipped_reason is None

def test_stream_emits_skipped_event_for_debate():
    fns = _fns("insufficient", MagicMock())
    async def collect():
        events = []
        async for e in stream_pipeline_events("TSLA", "q", "live", 30, None, True, **fns):
            events.append(e)
        return events
    events = asyncio.run(collect())
    skipped = [e for e in events if e.get("event") == "skipped"]
    assert skipped == [{"event": "skipped", "agent": "debate",
                        "message": "insufficient evidence"}]
    running_debate = [e for e in events if e.get("event") == "running"
                      and e.get("agent") == "debate"]
    assert running_debate == []
    complete = [e for e in events if e["event"] == "complete"][0]
    assert complete["data"]["research"]["evidence_status"] == "insufficient"
    assert complete["data"]["memo"]["debate_skipped_reason"] == "insufficient evidence"
