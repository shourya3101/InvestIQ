"""Additive schema fields: new defaults + backward-compatible construction."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from core.schemas import (
    ActionSignalSchema,
    DocumentSchema,
    EvidenceSchema,
    InvestmentMemoSchema,
    ResearchOutputSchema,
    SentimentOutputSchema,
)


def _evidence(**over):
    base = dict(citation_id="E1", snippet="s", filepath="f", source="src",
                similarity_score=0.5)
    base.update(over)
    return EvidenceSchema(**base)


def test_evidence_old_style_construction_still_validates():
    e = _evidence()
    assert e.relevance_score is None
    assert e.aboutness_score is None

def test_evidence_new_scores_roundtrip():
    e = _evidence(relevance_score=3.2, aboutness_score=0.75)
    assert e.relevance_score == 3.2
    assert e.aboutness_score == 0.75

def test_research_output_defaults_to_sufficient():
    r = ResearchOutputSchema(ticker="AAPL", question="q", summary="s")
    assert r.evidence_status == "sufficient"
    assert r.status_reason == ""

def test_research_output_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ResearchOutputSchema(ticker="AAPL", question="q", summary="s",
                             evidence_status="dubious")

def test_sentiment_defaults_to_ok():
    s = SentimentOutputSchema(
        ticker="AAPL", as_of=datetime.now(timezone.utc), window_days=30,
        overall_score=0.0, overall_label="neutral", summary="s",
    )
    assert s.data_status == "ok"

def test_action_signal_accepts_no_view():
    a = ActionSignalSchema(signal="no_view", confidence=0.0, rationale="no evidence")
    assert a.signal == "no_view"

def test_memo_gains_status_and_debate_reason():
    memo = InvestmentMemoSchema(
        ticker="AAPL", as_of=datetime.now(timezone.utc), question="q",
        thesis="t", catalysts=[], risks=[],
        action=ActionSignalSchema(signal="hold", confidence=0.5, rationale="r"),
        citations=[], risk_level="moderate", risk_score=50.0,
        writer_mode="deterministic",
    )
    assert memo.evidence_status == "sufficient"
    assert memo.debate_skipped_reason is None

def test_document_schema_about_score_defaults_none():
    d = DocumentSchema(content="c", source="s", filepath="f")
    assert d.about_score is None
