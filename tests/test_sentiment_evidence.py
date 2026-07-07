"""Sentiment scores the provided research pack; no_data on insufficient."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

from agents.sentiment_agent import run_sentiment
from core.schemas import EvidenceSchema, ResearchOutputSchema


class FakeScorer:
    def __init__(self):
        self.calls = []
    def __call__(self, text):
        self.calls.append(text)
        return [{"label": "Positive", "score": 0.9}]


def _research(status, n_items):
    evidence = [
        EvidenceSchema(citation_id=f"E{i}", snippet=f"Tesla news {i}", filepath="f",
                       source="newsapi", similarity_score=0.6)
        for i in range(1, n_items + 1)
    ]
    return ResearchOutputSchema(ticker="TSLA", question="q", evidence=evidence,
                                summary="s", evidence_status=status,
                                status_reason="reason")


def test_provided_research_is_used_without_retrieval():
    scorer = FakeScorer()
    with patch("agents.sentiment_agent.run_research") as mock_rr:
        out = run_sentiment("TSLA", research=_research("sufficient", 3), _scorer=scorer)
    mock_rr.assert_not_called()
    assert out.data_status == "ok"
    assert len(out.items) == 3
    assert len(scorer.calls) == 3

def test_insufficient_research_yields_no_data_and_no_scoring():
    scorer = FakeScorer()
    out = run_sentiment("TSLA", research=_research("insufficient", 0), _scorer=scorer)
    assert out.data_status == "no_data"
    assert out.overall_score == 0.0
    assert out.overall_label == "neutral"
    assert out.items == []
    assert scorer.calls == []
    assert "No sentiment data" in out.summary

def test_empty_evidence_with_ok_status_still_no_data():
    scorer = FakeScorer()
    out = run_sentiment("TSLA", research=_research("sufficient", 0), _scorer=scorer)
    assert out.data_status == "no_data"

def test_standalone_path_still_runs_research():
    with patch("agents.sentiment_agent.run_research",
               return_value=_research("sufficient", 2)) as mock_rr:
        out = run_sentiment("TSLA", _scorer=FakeScorer())
    mock_rr.assert_called_once()
    assert out.data_status == "ok"
    assert len(out.items) == 2
