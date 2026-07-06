"""run_research delegates to retrieve_evidence and surfaces the typed status."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch

from agents.research_agent import run_research
from core.retrieval import RetrievalResult
from core.schemas import EvidenceSchema


def _result(status, n_items, reason="3 candidates retrieved; ..."):
    evidence = [
        EvidenceSchema(citation_id=f"E{i}", snippet=f"snippet {i}", filepath="f",
                       source="newsapi", similarity_score=0.6,
                       aboutness_score=1.0, relevance_score=5.0)
        for i in range(1, n_items + 1)
    ]
    return RetrievalResult(ticker="TSLA", query_text="Tesla (TSLA): q",
                           evidence=evidence, evidence_status=status,
                           status_reason=reason)


def test_delegates_with_args_and_maps_fields():
    with patch("agents.research_agent.retrieve_evidence",
               return_value=_result("sufficient", 3)) as mock_ret:
        out = run_research("TSLA", "q", days_back=30, top_k=5)
    kwargs = mock_ret.call_args.kwargs
    assert kwargs["ticker"] == "TSLA" and kwargs["question"] == "q"
    assert kwargs["days_back"] == 30 and kwargs["top_k"] == 5
    assert out.evidence_status == "sufficient"
    assert out.status_reason.startswith("3 candidates")
    assert len(out.evidence) == 3
    assert out.evidence[0].relevance_score == 5.0

def test_insufficient_summary_is_honest():
    with patch("agents.research_agent.retrieve_evidence",
               return_value=_result("insufficient", 0)):
        out = run_research("TSLA", "q")
    assert out.evidence_status == "insufficient"
    assert "No trustworthy evidence" in out.summary

def test_partial_summary_mentions_partial():
    with patch("agents.research_agent.retrieve_evidence",
               return_value=_result("partial", 1)):
        out = run_research("TSLA", "q")
    assert "Partial evidence" in out.summary

def test_store_kwarg_forwarded():
    sentinel = object()
    with patch("agents.research_agent.retrieve_evidence",
               return_value=_result("sufficient", 3)) as mock_ret:
        run_research("TSLA", "q", store=sentinel)
    assert mock_ret.call_args.kwargs["store"] is sentinel
