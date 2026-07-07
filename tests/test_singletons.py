"""
Tests for core.singletons + store injection through the agent chain.
Written BEFORE implementation (TDD).

Covers:
  - get_store() is a lazy singleton (constructs VectorStoreManager once)
  - reset_store() clears the cache for test isolation
  - run_research uses the injected store instead of constructing one
  - run_sentiment propagates store= to run_research
  - run_risk propagates store= through run_sentiment to run_research
  - coordinator calls get_store() once per pipeline run
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch, call


# ── core.singletons ───────────────────────────────────────────────────────────


def test_get_store_constructs_vector_store_manager_only_once():
    from core import singletons
    from core.singletons import get_store, reset_store

    reset_store()  # start clean

    mock_instance = MagicMock()
    with patch("core.singletons.VectorStoreManager", return_value=mock_instance) as MockVSM:
        s1 = get_store()
        s2 = get_store()
        s3 = get_store()

    assert MockVSM.call_count == 1, "VectorStoreManager should be constructed exactly once"
    assert s1 is s2 is s3, "All calls must return the same instance"


def test_get_store_returns_same_instance_without_patch():
    from core.singletons import get_store, reset_store

    reset_store()

    with patch("core.singletons.VectorStoreManager", return_value=MagicMock()):
        a = get_store()
        b = get_store()

    assert a is b


def test_reset_store_clears_cached_instance():
    from core.singletons import get_store, reset_store

    reset_store()
    first_mock = MagicMock()
    second_mock = MagicMock()

    with patch("core.singletons.VectorStoreManager", side_effect=[first_mock, second_mock]):
        s1 = get_store()
        reset_store()        # ← clear the cache
        s2 = get_store()    # must construct a new one

    assert s1 is first_mock
    assert s2 is second_mock
    assert s1 is not s2


def test_reset_store_with_explicit_instance_injects_that_instance():
    from core.singletons import get_store, reset_store

    injected = MagicMock()
    reset_store(injected)

    assert get_store() is injected


# ── run_research store injection ──────────────────────────────────────────────


def test_run_research_calls_query_on_provided_store():
    from agents.research_agent import run_research

    mock_store = MagicMock()
    mock_store.query.return_value = []

    result = run_research("AAPL", "what happened?", days_back=30, top_k=5, store=mock_store)

    mock_store.query.assert_called_once()
    call_kwargs = mock_store.query.call_args
    assert call_kwargs is not None


def test_run_research_does_not_call_get_store_when_store_is_provided():
    # Store resolution now lives in core.retrieval; the singleton seam is
    # core.singletons.get_store (imported lazily inside retrieve_evidence).
    from agents.research_agent import run_research

    mock_store = MagicMock()
    mock_store.query.return_value = []

    with patch("core.singletons.get_store") as mock_get_store:
        run_research("AAPL", "q", store=mock_store)

    mock_get_store.assert_not_called()


def test_run_research_calls_get_store_when_no_store_provided():
    from agents.research_agent import run_research

    mock_store = MagicMock()
    mock_store.query.return_value = []

    with patch("core.singletons.get_store", return_value=mock_store) as mock_get_store:
        run_research("AAPL", "q")

    mock_get_store.assert_called_once()
    mock_store.query.assert_called_once()


# ── run_sentiment store propagation ──────────────────────────────────────────


def test_run_sentiment_passes_store_to_run_research():
    from agents.sentiment_agent import run_sentiment

    mock_store = MagicMock()

    # Patch run_research inside sentiment_agent so we can inspect the call
    with patch("agents.sentiment_agent.run_research") as mock_run_research:
        mock_run_research.return_value = MagicMock(evidence=[])
        run_sentiment("AAPL", store=mock_store)

    assert mock_run_research.called
    # store= must have been forwarded
    _, kwargs = mock_run_research.call_args
    assert kwargs.get("store") is mock_store, (
        f"run_sentiment did not forward store= to run_research; kwargs={kwargs}"
    )


def test_run_sentiment_does_not_create_store_itself():
    """run_sentiment should not call VectorStoreManager directly."""
    from agents.sentiment_agent import run_sentiment

    mock_store = MagicMock()
    with patch("agents.sentiment_agent.run_research") as mock_run_research:
        mock_run_research.return_value = MagicMock(evidence=[])
        with patch("core.singletons.VectorStoreManager") as MockVSM:
            run_sentiment("AAPL", store=mock_store)

    MockVSM.assert_not_called()


# ── run_risk store propagation ────────────────────────────────────────────────


def test_run_risk_passes_store_to_run_sentiment():
    from agents.risk_agent import run_risk
    from core.schemas import (
        SentimentOutputSchema, TrendOutputSchema, TrendSignalSchema
    )
    from datetime import datetime, timezone

    mock_store = MagicMock()

    fake_trend = TrendOutputSchema(
        ticker="TEST", mode="live",
        as_of=datetime.now(timezone.utc), signals=[], summary="ok"
    )
    fake_sentiment = SentimentOutputSchema(
        ticker="TEST", as_of=datetime.now(timezone.utc),
        window_days=30, overall_score=0.0, overall_label="neutral",
        items=[], summary="ok"
    )

    with patch("agents.risk_agent.run_trend", return_value=fake_trend):
        with patch("agents.risk_agent.run_sentiment") as mock_sentiment:
            mock_sentiment.return_value = fake_sentiment
            run_risk("TEST", store=mock_store)

    assert mock_sentiment.called
    _, kwargs = mock_sentiment.call_args
    assert kwargs.get("store") is mock_store, (
        f"run_risk did not forward store= to run_sentiment; kwargs={kwargs}"
    )


# ── Coordinator uses one store per pipeline run ───────────────────────────────


def test_coordinator_pipeline_calls_get_store_exactly_once():
    """_run_pipeline must retrieve the singleton once, not per-agent."""
    import asyncio
    from agents.coordinator_agent import _run_pipeline
    from datetime import datetime, timezone
    from core.schemas import (
        ActionSignalSchema, InvestmentMemoSchema, MemoryComparisonSchema,
        ResearchOutputSchema, RiskOutputSchema, SentimentOutputSchema,
        TrendOutputSchema,
    )

    def _r(ticker="T"): return ResearchOutputSchema(ticker=ticker, question="q", days_back=30, evidence=[], summary="ok")
    def _t(ticker="T"): return TrendOutputSchema(ticker=ticker, mode="live", as_of=datetime.now(timezone.utc), signals=[], summary="ok")
    def _s(ticker="T"): return SentimentOutputSchema(ticker=ticker, as_of=datetime.now(timezone.utc), window_days=30, overall_score=0.0, overall_label="neutral", items=[], summary="ok")
    def _rk(ticker="T"): return RiskOutputSchema(ticker=ticker, as_of=datetime.now(timezone.utc), risk_score=50.0, risk_level="moderate", flags=[], summary="ok")
    def _m(ticker="T"): return InvestmentMemoSchema(ticker=ticker, as_of=datetime.now(timezone.utc), question="q", thesis="t", catalysts=[], risks=[], action=ActionSignalSchema(signal="hold", confidence=0.5, rationale="r"), citations=[], risk_level="moderate", risk_score=50.0, writer_mode="deterministic")
    def _mc(ticker="T"): return MemoryComparisonSchema(ticker=ticker, current_as_of=datetime.now(timezone.utc), signal_changed=False, thesis_changed=False, summary="ok")

    async def fake_research(ticker, question, days_back, top_k): return _r(ticker)
    async def fake_trend(ticker, mode, filepath): return _t(ticker)
    async def fake_sentiment(ticker, question, window_days, top_k, research): return _s(ticker)
    async def fake_risk(ticker, mode, price_filepath, question, window_days, trend, sentiment): return _rk(ticker)
    async def fake_analyst(ticker, research, trend, sentiment, risk, question): return _m(ticker)
    async def fake_compare(memo): return _mc(memo.ticker)
    async def fake_save(memo): pass

    with patch("agents.coordinator_agent.get_store") as mock_get_store:
        mock_get_store.return_value = MagicMock()
        asyncio.run(_run_pipeline(
            "TEST", "q", "live", 30, None, False,
            _research_fn=fake_research, _trend_fn=fake_trend,
            _sentiment_fn=fake_sentiment, _risk_fn=fake_risk,
            _analyst_fn=fake_analyst, _compare_fn=fake_compare,
            _save_fn=fake_save,
        ))

    assert mock_get_store.call_count == 1, (
        f"Expected get_store() called once, got {mock_get_store.call_count}"
    )
