"""Risk agent: injected outputs are reused; no_data sentiment is flagged, not scored."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from unittest.mock import patch

from agents.risk_agent import run_risk
from core.schemas import (
    SentimentOutputSchema,
    TrendOutputSchema,
    TrendSignalSchema,
)


def _trend():
    return TrendOutputSchema(
        ticker="TSLA", mode="live", as_of=datetime.now(timezone.utc),
        signals=[TrendSignalSchema(horizon="30d", return_pct=1.0,
                                   volatility_pct=10.0, max_drawdown_pct=-1.0,
                                   trend_label="neutral")],
        summary="calm",
    )


def _sentiment(data_status="ok", label="neutral", score=0.0):
    return SentimentOutputSchema(
        ticker="TSLA", as_of=datetime.now(timezone.utc), window_days=30,
        overall_score=score, overall_label=label, items=[], summary="s",
        data_status=data_status,
    )


def test_injected_outputs_skip_recompute():
    with patch("agents.risk_agent.run_trend") as mock_t, \
         patch("agents.risk_agent.run_sentiment") as mock_s:
        run_risk("TSLA", trend=_trend(), sentiment=_sentiment())
    mock_t.assert_not_called()
    mock_s.assert_not_called()

def test_no_data_sentiment_adds_informational_flag():
    result = run_risk("TSLA", trend=_trend(), sentiment=_sentiment("no_data"))
    sentiment_flags = [f for f in result.flags if f.category == "sentiment"]
    assert len(sentiment_flags) == 1
    assert sentiment_flags[0].severity == "low"
    assert "No sentiment data" in sentiment_flags[0].message

def test_no_data_sentiment_adds_no_risk_points():
    baseline = run_risk("TSLA", trend=_trend(), sentiment=_sentiment("ok"))
    no_data = run_risk("TSLA", trend=_trend(), sentiment=_sentiment("no_data"))
    assert no_data.risk_score == baseline.risk_score

def test_negative_sentiment_still_scores_when_data_ok():
    result = run_risk("TSLA", trend=_trend(),
                      sentiment=_sentiment("ok", label="negative", score=-0.5))
    assert any(f.category == "sentiment" and f.severity == "high" for f in result.flags)

def test_standalone_path_still_computes():
    with patch("agents.risk_agent.run_trend", return_value=_trend()) as mock_t, \
         patch("agents.risk_agent.run_sentiment", return_value=_sentiment()) as mock_s:
        run_risk("TSLA")
    mock_t.assert_called_once()
    mock_s.assert_called_once()
