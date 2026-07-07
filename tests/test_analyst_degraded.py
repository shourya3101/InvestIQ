"""Analyst degradation: no_view on insufficient, capped confidence on partial."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from datetime import datetime, timezone
from unittest.mock import patch

from agents.analyst_agent import run_analyst_memo
from core.schemas import (
    EvidenceSchema,
    ResearchOutputSchema,
    RiskFlagSchema,
    RiskOutputSchema,
    SentimentOutputSchema,
    TrendOutputSchema,
    TrendSignalSchema,
)


class FakeProvider:
    def __init__(self, responses):
        self._queue = list(responses)
        self.calls = []
    def generate(self, system, user):
        self.calls.append((system, user))
        if not self._queue:
            raise RuntimeError("FakeProvider: response queue exhausted")
        return self._queue.pop(0)


def _research(status, n_items=0):
    evidence = [
        EvidenceSchema(citation_id=f"E{i}", snippet=f"Tesla news {i}", filepath="f",
                       source="newsapi", similarity_score=0.6)
        for i in range(1, n_items + 1)
    ]
    return ResearchOutputSchema(ticker="TSLA", question="q", evidence=evidence,
                                summary="s", evidence_status=status,
                                status_reason="30 candidates; 30 rejected; 0 passed.")

def _trend():
    return TrendOutputSchema(ticker="TSLA", mode="live",
                             as_of=datetime.now(timezone.utc),
                             signals=[TrendSignalSchema(horizon="30d", return_pct=-14.7,
                                                        volatility_pct=45.1,
                                                        max_drawdown_pct=-15.4,
                                                        trend_label="bearish")],
                             summary="bearish")

def _sentiment(data_status="ok"):
    return SentimentOutputSchema(ticker="TSLA", as_of=datetime.now(timezone.utc),
                                 window_days=30, overall_score=0.0,
                                 overall_label="neutral", items=[], summary="s",
                                 data_status=data_status)

def _risk():
    return RiskOutputSchema(ticker="TSLA", as_of=datetime.now(timezone.utc),
                            risk_score=95.0, risk_level="high",
                            flags=[RiskFlagSchema(category="volatility", severity="high",
                                                  message="Annualised volatility 45.1%.")],
                            summary="high")


def _memo(status, provider, n_items=0):
    with patch("agents.analyst_agent._build_provider", return_value=provider):
        return run_analyst_memo(
            ticker="TSLA", research=_research(status, n_items), trend=_trend(),
            sentiment=_sentiment("no_data" if status == "insufficient" else "ok"),
            risk=_risk(), writer_mode="openai",
        )


# ── insufficient ─────────────────────────────────────────────────────────────

def test_insufficient_yields_no_view_with_zero_confidence():
    provider = FakeProvider(["Market-data-only thesis. No view is taken."])
    memo = _memo("insufficient", provider)
    assert memo.action.signal == "no_view"
    assert memo.action.confidence == 0.0
    assert memo.evidence_status == "insufficient"
    assert memo.catalysts == []
    assert memo.citations == []

def test_insufficient_makes_only_the_thesis_llm_call():
    provider = FakeProvider(["Market-data-only thesis."])
    _memo("insufficient", provider)
    assert len(provider.calls) == 1          # no catalysts call, no recommendation call
    system, user = provider.calls[0]
    assert "do not invent" in system.lower()
    assert "no trustworthy evidence" in user.lower()

def test_insufficient_thesis_llm_failure_falls_back_deterministically():
    class Boom:
        def generate(self, s, u):
            raise RuntimeError("boom")
    with patch("agents.analyst_agent._build_provider", return_value=Boom()):
        memo = run_analyst_memo(ticker="TSLA", research=_research("insufficient"),
                                trend=_trend(), sentiment=_sentiment("no_data"),
                                risk=_risk(), writer_mode="openai")
    assert memo.action.signal == "no_view"
    assert "market data only" in memo.thesis.lower()

def test_insufficient_deterministic_mode_no_llm():
    memo = run_analyst_memo(ticker="TSLA", research=_research("insufficient"),
                            trend=_trend(), sentiment=_sentiment("no_data"),
                            risk=_risk(), writer_mode="off")
    assert memo.action.signal == "no_view"
    assert memo.evidence_status == "insufficient"


# ── partial ──────────────────────────────────────────────────────────────────

def test_partial_caps_llm_confidence():
    provider = FakeProvider([
        "Thesis noting the limited evidence base.",
        json.dumps({"catalysts": ["c1"], "risks": ["r1"]}),
        json.dumps({"signal": "buy", "confidence": 0.9, "rationale": "r"}),
    ])
    memo = _memo("partial", provider, n_items=2)
    assert memo.action.signal == "buy"
    assert memo.action.confidence == 0.6      # capped from 0.9
    assert memo.evidence_status == "partial"

def test_partial_thesis_prompt_mentions_limited_evidence():
    provider = FakeProvider([
        "Thesis.", json.dumps({"catalysts": ["c"], "risks": ["r"]}),
        json.dumps({"signal": "hold", "confidence": 0.5, "rationale": "r"}),
    ])
    _memo("partial", provider, n_items=2)
    thesis_user_prompt = provider.calls[0][1]
    assert "limited" in thesis_user_prompt.lower()


# ── sufficient (regression) ──────────────────────────────────────────────────

def test_sufficient_confidence_not_capped():
    provider = FakeProvider([
        "Thesis.", json.dumps({"catalysts": ["c"], "risks": ["r"]}),
        json.dumps({"signal": "buy", "confidence": 0.9, "rationale": "r"}),
    ])
    memo = _memo("sufficient", provider, n_items=3)
    assert memo.action.confidence == 0.9
    assert memo.evidence_status == "sufficient"
