"""
Tests for the refactored analyst agent — written BEFORE implementation (TDD).

The agent is split into 3 focused LLM calls:
  1. _write_thesis        → investment thesis prose
  2. _write_catalysts_risks → (catalysts list, risks list) via JSON
  3. _write_recommendation → (signal, confidence, rationale) via JSON

Each helper accepts an injected provider so tests never hit a real LLM.
A FakeProvider class is used instead of MagicMock to test real behaviour.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import pytest

from agents.analyst_agent import (
    _MemoContext,
    _write_thesis,
    _write_catalysts_risks,
    _write_recommendation,
    run_analyst_memo,
)
from core.schemas import (
    ActionSignalSchema,
    EvidenceSchema,
    InvestmentMemoSchema,
    ResearchOutputSchema,
    RiskFlagSchema,
    RiskOutputSchema,
    SentimentOutputSchema,
    TrendOutputSchema,
    TrendSignalSchema,
)


# ── FakeProvider ──────────────────────────────────────────────────────────────

class FakeProvider:
    """Real object (not a mock) that returns pre-set responses in order.

    Raises RuntimeError when the response queue is exhausted.
    Pass an Exception instance in the queue to simulate a provider failure.
    """

    def __init__(self, *responses):
        self._queue = list(responses)
        self.call_count = 0
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.call_count += 1
        self.calls.append((system, user))
        if not self._queue:
            raise RuntimeError("FakeProvider: response queue exhausted")
        response = self._queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _ctx(ticker="AAPL") -> _MemoContext:
    return _MemoContext(
        ticker=ticker,
        trend_summary="30d return +2.5%, bullish, vol 18.4%",
        sentiment_label="positive",
        sentiment_score=0.35,
        risk_level="moderate",
        risk_score=42.0,
        risk_flags="None",
        evidence_lines="E1: Apple reported record quarterly revenue of $120B.",
    )


def _research(ticker="AAPL") -> ResearchOutputSchema:
    ev = EvidenceSchema(
        citation_id="E1",
        snippet="Apple reported record quarterly revenue of $120B.",
        filepath="sample.txt",
        source="bloomberg",
        ticker=ticker,
        date=datetime(2024, 1, 15, tzinfo=timezone.utc),
        similarity_score=0.9,
    )
    return ResearchOutputSchema(
        ticker=ticker, question="q", days_back=30, evidence=[ev], summary="ok"
    )


def _trend(ticker="AAPL") -> TrendOutputSchema:
    return TrendOutputSchema(
        ticker=ticker,
        mode="live",
        as_of=datetime.now(timezone.utc),
        signals=[
            TrendSignalSchema(
                horizon="30d",
                return_pct=2.5,
                volatility_pct=18.4,
                max_drawdown_pct=-3.1,
                trend_label="bullish",
            )
        ],
        summary="ok",
    )


def _sentiment(ticker="AAPL") -> SentimentOutputSchema:
    return SentimentOutputSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        window_days=30,
        overall_score=0.35,
        overall_label="positive",
        items=[],
        summary="ok",
    )


def _risk(ticker="AAPL") -> RiskOutputSchema:
    return RiskOutputSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        risk_score=42.0,
        risk_level="moderate",
        flags=[
            RiskFlagSchema(category="volatility", severity="low", message="Vol 18.4%.")
        ],
        summary="ok",
    )


# ── _MemoContext ──────────────────────────────────────────────────────────────


def test_memo_context_can_be_constructed_with_test_data():
    ctx = _ctx()
    assert ctx.ticker == "AAPL"
    assert ctx.risk_level == "moderate"
    assert len(ctx.evidence_lines) > 0


# ── _write_thesis ─────────────────────────────────────────────────────────────


def test_write_thesis_calls_generate_exactly_once():
    provider = FakeProvider("Apple Inc. is well-positioned for growth. " * 5)
    _write_thesis(provider, _ctx())
    assert provider.call_count == 1


def test_write_thesis_returns_non_empty_string():
    thesis_text = "Apple Inc. is well-positioned. Strong revenue growth." * 3
    provider = FakeProvider(thesis_text)
    result = _write_thesis(provider, _ctx())
    assert isinstance(result, str)
    assert len(result) > 0


def test_write_thesis_passes_ticker_to_prompt():
    provider = FakeProvider("Thesis about MSFT goes here.")
    _write_thesis(provider, _ctx("MSFT"))
    _, user_prompt = provider.calls[0]
    assert "MSFT" in user_prompt


def test_write_thesis_returns_deterministic_fallback_on_provider_error():
    provider = FakeProvider(RuntimeError("API down"))
    result = _write_thesis(provider, _ctx())
    assert isinstance(result, str)
    assert len(result) > 0


def test_write_thesis_returns_fallback_when_generate_returns_empty():
    provider = FakeProvider("")
    result = _write_thesis(provider, _ctx())
    assert len(result) > 0


# ── _write_catalysts_risks ────────────────────────────────────────────────────


_VALID_CR_JSON = '{"catalysts": ["AI pipeline growth", "Services revenue"], "risks": ["Supply chain", "Macro risk"]}'


def test_write_catalysts_risks_calls_generate_exactly_once():
    provider = FakeProvider(_VALID_CR_JSON)
    _write_catalysts_risks(provider, _ctx())
    assert provider.call_count == 1


def test_write_catalysts_risks_returns_two_lists():
    provider = FakeProvider(_VALID_CR_JSON)
    catalysts, risks = _write_catalysts_risks(provider, _ctx())
    assert isinstance(catalysts, list)
    assert isinstance(risks, list)


def test_write_catalysts_risks_parses_json_catalysts():
    provider = FakeProvider(_VALID_CR_JSON)
    catalysts, _ = _write_catalysts_risks(provider, _ctx())
    assert "AI pipeline growth" in catalysts


def test_write_catalysts_risks_parses_json_risks():
    provider = FakeProvider(_VALID_CR_JSON)
    _, risks = _write_catalysts_risks(provider, _ctx())
    assert "Supply chain" in risks


def test_write_catalysts_risks_returns_fallback_on_invalid_json():
    provider = FakeProvider("not valid json at all")
    catalysts, risks = _write_catalysts_risks(provider, _ctx())
    assert len(catalysts) > 0
    assert len(risks) > 0


def test_write_catalysts_risks_returns_fallback_on_missing_key():
    provider = FakeProvider('{"catalysts": ["growth"]}')  # missing "risks"
    catalysts, risks = _write_catalysts_risks(provider, _ctx())
    assert len(risks) > 0


def test_write_catalysts_risks_returns_fallback_on_provider_error():
    provider = FakeProvider(RuntimeError("timeout"))
    catalysts, risks = _write_catalysts_risks(provider, _ctx())
    assert len(catalysts) > 0
    assert len(risks) > 0


def test_write_catalysts_risks_accepts_markdown_fenced_json():
    fenced = '```json\n{"catalysts": ["Revenue growth"], "risks": ["Competition"]}\n```'
    provider = FakeProvider(fenced)
    catalysts, risks = _write_catalysts_risks(provider, _ctx())
    assert "Revenue growth" in catalysts
    assert "Competition" in risks


# ── _write_recommendation ─────────────────────────────────────────────────────


_VALID_REC_JSON = '{"signal": "buy", "confidence": 0.75, "rationale": "Strong momentum justifies accumulation."}'


def test_write_recommendation_calls_generate_exactly_once():
    provider = FakeProvider(_VALID_REC_JSON)
    _write_recommendation(provider, _ctx())
    assert provider.call_count == 1


def test_write_recommendation_returns_valid_signal():
    provider = FakeProvider(_VALID_REC_JSON)
    signal, confidence, rationale = _write_recommendation(provider, _ctx())
    assert signal in ("buy", "hold", "sell")


def test_write_recommendation_parses_signal_correctly():
    provider = FakeProvider(_VALID_REC_JSON)
    signal, _, _ = _write_recommendation(provider, _ctx())
    assert signal == "buy"


def test_write_recommendation_parses_confidence_correctly():
    provider = FakeProvider(_VALID_REC_JSON)
    _, confidence, _ = _write_recommendation(provider, _ctx())
    assert confidence == pytest.approx(0.75)


def test_write_recommendation_parses_rationale_correctly():
    provider = FakeProvider(_VALID_REC_JSON)
    _, _, rationale = _write_recommendation(provider, _ctx())
    assert "momentum" in rationale.lower()


def test_write_recommendation_clamps_confidence_to_valid_range():
    provider = FakeProvider('{"signal": "hold", "confidence": 1.5, "rationale": "ok"}')
    _, confidence, _ = _write_recommendation(provider, _ctx())
    assert 0.0 <= confidence <= 1.0


def test_write_recommendation_normalises_invalid_signal_to_hold():
    provider = FakeProvider('{"signal": "strong_buy", "confidence": 0.8, "rationale": "ok"}')
    signal, _, _ = _write_recommendation(provider, _ctx())
    assert signal == "hold"


def test_write_recommendation_returns_fallback_on_invalid_json():
    provider = FakeProvider("i recommend buying")
    signal, confidence, rationale = _write_recommendation(provider, _ctx())
    assert signal in ("buy", "hold", "sell")
    assert len(rationale) > 0


def test_write_recommendation_returns_fallback_on_provider_error():
    provider = FakeProvider(RuntimeError("rate limited"))
    signal, confidence, rationale = _write_recommendation(provider, _ctx())
    assert signal in ("buy", "hold", "sell")


# ── run_analyst_memo integration ──────────────────────────────────────────────


def _make_provider_mock(thesis, catalysts_json, rec_json):
    """Return a mock OpenAI/Groq provider that returns the three responses in order."""
    m = MagicMock()
    m.generate.side_effect = [thesis, catalysts_json, rec_json]
    return m


def test_run_analyst_memo_llm_mode_calls_generate_exactly_three_times():
    mock_provider = _make_provider_mock(
        "Strong thesis for AAPL based on solid fundamentals.",
        _VALID_CR_JSON,
        _VALID_REC_JSON,
    )
    with patch("agents.analyst_agent.OpenAIProvider", return_value=mock_provider):
        run_analyst_memo(
            ticker="AAPL",
            research=_research(),
            trend=_trend(),
            sentiment=_sentiment(),
            risk=_risk(),
            writer_mode="openai",
        )
    assert mock_provider.generate.call_count == 3


def test_run_analyst_memo_llm_provider_constructed_with_max_tokens_1500():
    mock_provider = _make_provider_mock(
        "Solid thesis.",
        _VALID_CR_JSON,
        _VALID_REC_JSON,
    )
    with patch("agents.analyst_agent.OpenAIProvider", return_value=mock_provider) as MockCls:
        run_analyst_memo(
            ticker="AAPL",
            research=_research(),
            trend=_trend(),
            sentiment=_sentiment(),
            risk=_risk(),
            writer_mode="openai",
        )
    _, kwargs = MockCls.call_args
    assert kwargs.get("max_tokens") == 1500, (
        f"Expected max_tokens=1500, got {kwargs.get('max_tokens')}"
    )


def test_run_analyst_memo_returns_investment_memo_schema():
    mock_provider = _make_provider_mock(
        "Thesis text here.",
        _VALID_CR_JSON,
        _VALID_REC_JSON,
    )
    with patch("agents.analyst_agent.OpenAIProvider", return_value=mock_provider):
        result = run_analyst_memo(
            ticker="AAPL",
            research=_research(),
            trend=_trend(),
            sentiment=_sentiment(),
            risk=_risk(),
            writer_mode="openai",
        )
    assert isinstance(result, InvestmentMemoSchema)


def test_run_analyst_memo_thesis_comes_from_first_llm_call():
    thesis_text = "Apple is a uniquely positioned technology giant with durable competitive advantages."
    mock_provider = _make_provider_mock(thesis_text, _VALID_CR_JSON, _VALID_REC_JSON)
    with patch("agents.analyst_agent.OpenAIProvider", return_value=mock_provider):
        result = run_analyst_memo(
            ticker="AAPL",
            research=_research(),
            trend=_trend(),
            sentiment=_sentiment(),
            risk=_risk(),
            writer_mode="openai",
        )
    assert result.thesis == thesis_text


def test_run_analyst_memo_catalysts_come_from_second_llm_call():
    mock_provider = _make_provider_mock(
        "Thesis.",
        '{"catalysts": ["AI growth", "Services"], "risks": ["Competition"]}',
        _VALID_REC_JSON,
    )
    with patch("agents.analyst_agent.OpenAIProvider", return_value=mock_provider):
        result = run_analyst_memo(
            ticker="AAPL",
            research=_research(),
            trend=_trend(),
            sentiment=_sentiment(),
            risk=_risk(),
            writer_mode="openai",
        )
    assert "AI growth" in result.catalysts


def test_run_analyst_memo_risks_come_from_second_llm_call():
    mock_provider = _make_provider_mock(
        "Thesis.",
        '{"catalysts": ["AI growth"], "risks": ["Macro headwinds", "Valuation"]}',
        _VALID_REC_JSON,
    )
    with patch("agents.analyst_agent.OpenAIProvider", return_value=mock_provider):
        result = run_analyst_memo(
            ticker="AAPL",
            research=_research(),
            trend=_trend(),
            sentiment=_sentiment(),
            risk=_risk(),
            writer_mode="openai",
        )
    assert "Macro headwinds" in result.risks


def test_run_analyst_memo_action_signal_from_third_llm_call():
    mock_provider = _make_provider_mock(
        "Thesis.",
        _VALID_CR_JSON,
        '{"signal": "sell", "confidence": 0.65, "rationale": "Overvalued relative to peers."}',
    )
    with patch("agents.analyst_agent.OpenAIProvider", return_value=mock_provider):
        result = run_analyst_memo(
            ticker="AAPL",
            research=_research(),
            trend=_trend(),
            sentiment=_sentiment(),
            risk=_risk(),
            writer_mode="openai",
        )
    assert result.action.signal == "sell"


def test_run_analyst_memo_writer_mode_recorded_as_openai():
    mock_provider = _make_provider_mock("Thesis.", _VALID_CR_JSON, _VALID_REC_JSON)
    with patch("agents.analyst_agent.OpenAIProvider", return_value=mock_provider):
        result = run_analyst_memo(
            ticker="AAPL",
            research=_research(),
            trend=_trend(),
            sentiment=_sentiment(),
            risk=_risk(),
            writer_mode="openai",
        )
    assert result.writer_mode == "openai"


def test_run_analyst_memo_deterministic_mode_does_not_call_any_provider():
    with patch("agents.analyst_agent.OpenAIProvider") as MockOpenAI, \
         patch("agents.analyst_agent.GroqProvider") as MockGroq, \
         patch("agents.analyst_agent.ClaudeProvider") as MockClaude:
        run_analyst_memo(
            ticker="AAPL",
            research=_research(),
            trend=_trend(),
            sentiment=_sentiment(),
            risk=_risk(),
            writer_mode="off",
        )
    MockOpenAI.assert_not_called()
    MockGroq.assert_not_called()
    MockClaude.assert_not_called()


def test_run_analyst_memo_survives_single_llm_call_failure():
    """If one of the 3 calls fails the agent falls back gracefully, no crash."""
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = [
        "Good thesis text here.",
        RuntimeError("API timeout on call 2"),
        _VALID_REC_JSON,
    ]
    with patch("agents.analyst_agent.OpenAIProvider", return_value=mock_provider):
        result = run_analyst_memo(
            ticker="AAPL",
            research=_research(),
            trend=_trend(),
            sentiment=_sentiment(),
            risk=_risk(),
            writer_mode="openai",
        )
    assert isinstance(result, InvestmentMemoSchema)
    assert len(result.catalysts) > 0
    assert len(result.risks) > 0
