"""
Tests for progressive/streaming analysis — written BEFORE implementation (TDD).

Two layers tested:
  1. stream_pipeline_events() async generator — event ordering, structure, resilience
  2. POST /analyze/stream FastAPI endpoint — HTTP contract (content-type, valid NDJSON)

The generator tests inject fake agents (same pattern as test_coordinator_async.py)
so no real ChromaDB or LLM calls are made.
The endpoint tests patch stream_pipeline_events so the HTTP layer is tested in
isolation from the pipeline logic.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from agents.coordinator_agent import stream_pipeline_events
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


# ── Fixture factories (reused from test_coordinator_async) ────────────────────


def _r(ticker="T"):
    return ResearchOutputSchema(ticker=ticker, question="q", days_back=30, evidence=[], summary="ok")

def _t(ticker="T"):
    return TrendOutputSchema(ticker=ticker, mode="live", as_of=datetime.now(timezone.utc), signals=[], summary="ok")

def _s(ticker="T"):
    return SentimentOutputSchema(ticker=ticker, as_of=datetime.now(timezone.utc), window_days=30, overall_score=0.0, overall_label="neutral", items=[], summary="ok")

def _rk(ticker="T"):
    return RiskOutputSchema(ticker=ticker, as_of=datetime.now(timezone.utc), risk_score=50.0, risk_level="moderate", flags=[], summary="ok")

def _m(ticker="T"):
    return InvestmentMemoSchema(
        ticker=ticker, as_of=datetime.now(timezone.utc), question="q", thesis="t",
        catalysts=[], risks=[],
        action=ActionSignalSchema(signal="hold", confidence=0.5, rationale="r"),
        citations=[], risk_level="moderate", risk_score=50.0, writer_mode="deterministic"
    )

def _db(ticker="T"):
    return DebateOutputSchema(
        ticker=ticker, as_of=datetime.now(timezone.utc),
        bull=DebateArgumentSchema(side="bull", arguments=[], confidence=0.6, key_evidence=[]),
        bear=DebateArgumentSchema(side="bear", arguments=[], confidence=0.4, key_evidence=[]),
        coordinator_verdict="balanced", final_bias="neutral", memo_update=""
    )

def _mc(ticker="T"):
    return MemoryComparisonSchema(ticker=ticker, current_as_of=datetime.now(timezone.utc), signal_changed=False, thesis_changed=False, summary="ok")


def _fakes(with_debate=True):
    async def fake_research(t, q, db, tk): return _r(t)
    async def fake_trend(t, m, fp): return _t(t)
    async def fake_sentiment(t, q, wd, tk): return _s(t)
    async def fake_risk(t, m, fp, q, wd): return _rk(t)
    async def fake_analyst(t, res, tr, se, ri, q): return _m(t)
    async def fake_debate(t, res, tr, se, ri): return _db(t)
    async def fake_compare(memo): return _mc(memo.ticker)
    async def fake_save(memo): pass

    return dict(
        _research_fn=fake_research, _trend_fn=fake_trend,
        _sentiment_fn=fake_sentiment, _risk_fn=fake_risk,
        _analyst_fn=fake_analyst,
        _debate_fn=fake_debate if with_debate else None,
        _compare_fn=fake_compare, _save_fn=fake_save,
    )


def _collect(ticker="TEST", run_debate=True, fakes=None, timeout=10.0):
    """Drain stream_pipeline_events into a list, patching get_store."""
    if fakes is None:
        fakes = _fakes(with_debate=run_debate)

    gen = stream_pipeline_events(
        ticker, "q", "live", 30, None, run_debate, **fakes
    )

    async def _drain():
        return [e async for e in gen]

    with patch("agents.coordinator_agent.get_store", return_value=MagicMock()):
        return asyncio.run(asyncio.wait_for(_drain(), timeout=timeout))


# ── Event structure ───────────────────────────────────────────────────────────


def test_all_events_are_dicts():
    events = _collect()
    assert all(isinstance(e, dict) for e in events)


def test_all_events_have_event_field():
    events = _collect()
    assert all("event" in e for e in events)


def test_all_events_are_json_serialisable():
    events = _collect()
    for e in events:
        json.dumps(e)  # must not raise


# ── Running events emitted before done ────────────────────────────────────────


def _positions(events, field, value):
    return [i for i, e in enumerate(events) if e.get(field) == value]


def test_research_running_emitted_before_research_done():
    events = _collect()
    running = [i for i, e in enumerate(events) if e.get("event") == "running" and e.get("agent") == "research"]
    done    = [i for i, e in enumerate(events) if e.get("event") == "done"    and e.get("agent") == "research"]
    assert running and done
    assert running[0] < done[0]


def test_trend_running_emitted_before_trend_done():
    events = _collect()
    running = [i for i, e in enumerate(events) if e.get("event") == "running" and e.get("agent") == "trend"]
    done    = [i for i, e in enumerate(events) if e.get("event") == "done"    and e.get("agent") == "trend"]
    assert running and done
    assert running[0] < done[0]


def test_sentiment_running_before_done():
    events = _collect()
    running = [i for i, e in enumerate(events) if e.get("event") == "running" and e.get("agent") == "sentiment"]
    done    = [i for i, e in enumerate(events) if e.get("event") == "done"    and e.get("agent") == "sentiment"]
    assert running[0] < done[0]


def test_risk_running_before_done():
    events = _collect()
    running = [i for i, e in enumerate(events) if e.get("event") == "running" and e.get("agent") == "risk"]
    done    = [i for i, e in enumerate(events) if e.get("event") == "done"    and e.get("agent") == "risk"]
    assert running[0] < done[0]


def test_analyst_running_before_done():
    events = _collect()
    running = [i for i, e in enumerate(events) if e.get("event") == "running" and e.get("agent") == "analyst"]
    done    = [i for i, e in enumerate(events) if e.get("event") == "done"    and e.get("agent") == "analyst"]
    assert running[0] < done[0]


def test_memory_running_before_done():
    events = _collect()
    running = [i for i, e in enumerate(events) if e.get("event") == "running" and e.get("agent") == "memory"]
    done    = [i for i, e in enumerate(events) if e.get("event") == "done"    and e.get("agent") == "memory"]
    assert running[0] < done[0]


# ── Done events emitted for every agent ───────────────────────────────────────


def _done_agents(events):
    return {e["agent"] for e in events if e.get("event") == "done"}


def test_research_done_event_emitted():
    assert "research" in _done_agents(_collect())


def test_trend_done_event_emitted():
    assert "trend" in _done_agents(_collect())


def test_sentiment_done_event_emitted():
    assert "sentiment" in _done_agents(_collect())


def test_risk_done_event_emitted():
    assert "risk" in _done_agents(_collect())


def test_analyst_done_event_emitted():
    assert "analyst" in _done_agents(_collect())


def test_debate_done_event_emitted_when_flag_true():
    assert "debate" in _done_agents(_collect(run_debate=True))


def test_debate_done_event_not_emitted_when_flag_false():
    assert "debate" not in _done_agents(_collect(run_debate=False))


def test_memory_done_event_emitted():
    assert "memory" in _done_agents(_collect())


# ── Complete event is last ─────────────────────────────────────────────────────


def test_complete_event_is_emitted():
    events = _collect()
    assert any(e.get("event") == "complete" for e in events)


def test_complete_event_is_the_last_event():
    events = _collect()
    assert events[-1]["event"] == "complete"


def test_complete_event_has_data_field():
    events = _collect()
    complete = next(e for e in events if e["event"] == "complete")
    assert "data" in complete


def test_complete_event_data_contains_ticker():
    events = _collect("AAPL")
    complete = next(e for e in events if e["event"] == "complete")
    assert complete["data"]["ticker"] == "AAPL"


def test_complete_event_data_contains_memo():
    events = _collect()
    complete = next(e for e in events if e["event"] == "complete")
    assert "memo" in complete["data"]


# ── Done events carry data ────────────────────────────────────────────────────


def test_research_done_event_has_data():
    events = _collect()
    ev = next(e for e in events if e.get("event") == "done" and e.get("agent") == "research")
    assert "data" in ev and isinstance(ev["data"], dict)


def test_trend_done_event_has_data():
    events = _collect()
    ev = next(e for e in events if e.get("event") == "done" and e.get("agent") == "trend")
    assert "data" in ev and isinstance(ev["data"], dict)


def test_sentiment_done_event_has_data():
    events = _collect()
    ev = next(e for e in events if e.get("event") == "done" and e.get("agent") == "sentiment")
    assert "data" in ev


def test_risk_done_event_has_data():
    events = _collect()
    ev = next(e for e in events if e.get("event") == "done" and e.get("agent") == "risk")
    assert "data" in ev


# ── Pipeline ordering constraints ─────────────────────────────────────────────


def test_sentiment_done_after_research_done():
    events = _collect()
    r_done = next(i for i, e in enumerate(events) if e.get("event") == "done" and e.get("agent") == "research")
    s_done = next(i for i, e in enumerate(events) if e.get("event") == "done" and e.get("agent") == "sentiment")
    assert r_done < s_done


def test_risk_done_after_trend_done():
    events = _collect()
    t_done = next(i for i, e in enumerate(events) if e.get("event") == "done" and e.get("agent") == "trend")
    rk_done = next(i for i, e in enumerate(events) if e.get("event") == "done" and e.get("agent") == "risk")
    assert t_done < rk_done


def test_analyst_done_after_risk_done():
    events = _collect()
    rk_done = next(i for i, e in enumerate(events) if e.get("event") == "done" and e.get("agent") == "risk")
    an_done = next(i for i, e in enumerate(events) if e.get("event") == "done" and e.get("agent") == "analyst")
    assert rk_done < an_done


def test_memory_done_after_analyst_done():
    events = _collect()
    an_done = next(i for i, e in enumerate(events) if e.get("event") == "done" and e.get("agent") == "analyst")
    mem_done = next(i for i, e in enumerate(events) if e.get("event") == "done" and e.get("agent") == "memory")
    assert an_done < mem_done


# ── Resilience: agent failure produces error event, stream continues ──────────


def test_stream_continues_after_research_failure():
    fakes = _fakes()

    async def exploding_research(t, q, db, tk):
        raise RuntimeError("research down")

    fakes["_research_fn"] = exploding_research
    events = _collect(fakes=fakes)

    agents_done = _done_agents(events)
    # Sentiment, risk, analyst, memory must still complete
    assert "sentiment" in agents_done
    assert "risk" in agents_done
    assert "analyst" in agents_done


def test_stream_emits_error_event_on_research_failure():
    fakes = _fakes()

    async def exploding_research(t, q, db, tk):
        raise RuntimeError("research boom")

    fakes["_research_fn"] = exploding_research
    events = _collect(fakes=fakes)

    error_events = [e for e in events if e.get("event") == "error" and e.get("agent") == "research"]
    assert len(error_events) == 1
    assert "message" in error_events[0]


def test_stream_complete_event_emitted_even_after_agent_failure():
    fakes = _fakes()

    async def exploding_trend(t, m, fp):
        raise RuntimeError("trend boom")

    fakes["_trend_fn"] = exploding_trend
    events = _collect(fakes=fakes)
    assert events[-1]["event"] == "complete"


# ── HTTP endpoint ─────────────────────────────────────────────────────────────


def test_analyze_stream_endpoint_exists():
    from fastapi.testclient import TestClient
    from api.routes import app

    # Patch the generator so no real pipeline runs
    async def fake_gen(*a, **kw):
        yield {"event": "complete", "data": {"ticker": "TEST"}}

    with patch("api.routes.stream_pipeline_events", fake_gen):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/analyze/stream", json={
            "ticker": "TEST", "question": "q", "mode": "live", "days_back": 30, "run_debate": False
        })

    assert response.status_code == 200


def test_analyze_stream_content_type_is_ndjson():
    from fastapi.testclient import TestClient
    from api.routes import app

    async def fake_gen(*a, **kw):
        yield {"event": "complete", "data": {"ticker": "TEST"}}

    with patch("api.routes.stream_pipeline_events", fake_gen):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/analyze/stream", json={
            "ticker": "TEST", "question": "q", "mode": "live", "days_back": 30, "run_debate": False
        })

    assert "ndjson" in response.headers.get("content-type", "").lower()


def test_analyze_stream_response_lines_are_valid_json():
    from fastapi.testclient import TestClient
    from api.routes import app

    async def fake_gen(*a, **kw):
        yield {"event": "running", "agent": "research"}
        yield {"event": "done", "agent": "research", "data": {}}
        yield {"event": "complete", "data": {"ticker": "TEST"}}

    with patch("api.routes.stream_pipeline_events", fake_gen):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/analyze/stream", json={
            "ticker": "TEST", "question": "q", "mode": "live", "days_back": 30, "run_debate": False
        })

    lines = [l for l in response.text.split("\n") if l.strip()]
    assert len(lines) >= 1
    for line in lines:
        json.loads(line)  # each line must be valid JSON


def test_analyze_stream_rejects_empty_ticker():
    from fastapi.testclient import TestClient
    from api.routes import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/analyze/stream", json={
        "ticker": "", "question": "q", "mode": "live", "days_back": 30, "run_debate": False
    })
    assert response.status_code == 422
