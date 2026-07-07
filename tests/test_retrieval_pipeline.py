"""retrieve_evidence: company-aware query, gates, re-rank, typed status."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from core.company_registry import CompanyInfo
from core.retrieval import RetrievalResult, retrieve_evidence

TSLA = CompanyInfo(ticker="TSLA", name="Tesla", aliases=["Tesla"], source="config")

RECENT = (datetime.now() - timedelta(days=3)).isoformat()
STALE = (datetime.now() - timedelta(days=400)).isoformat()

ON_TOPIC = "Tesla reported record deliveries. Tesla shares rose after the report."
OFF_TOPIC = "Intel stock surged 190% in 2026 on foundry wins."
PASSING_MENTION = "Bybit lists contracts for TSLA, AMZN, META and others."


def _cand(text, distance=0.4, date=RECENT, about_meta=None):
    meta = {"ticker": "TSLA", "source": "newsapi", "filepath": "f", "date": date}
    if about_meta is not None:
        meta["about_score"] = about_meta
    return {"text": text, "metadata": meta, "distance": distance}


def _store(candidates):
    store = MagicMock()
    store.query.return_value = candidates
    return store


class FakeReranker:
    """Scores pairs by keyword: 'Tesla' in passage → high, else low."""
    def __init__(self, high=8.0, low=-8.0):
        self.high, self.low = high, low
        self.calls = []
    def predict(self, pairs):
        self.calls.append(pairs)
        return [self.high if "Tesla" in p[1] else self.low for p in pairs]


def _run(candidates, reranker=None, **kw):
    kw.setdefault("days_back", 30)
    return retrieve_evidence(
        "TSLA", "What are the key catalysts and risks?",
        store=_store(candidates), _company=TSLA,
        _reranker=reranker if reranker is not None else FakeReranker(),
        **kw,
    )


# ── query construction ────────────────────────────────────────────────────────

def test_query_text_names_the_company():
    store = _store([])
    retrieve_evidence("TSLA", "What are the key catalysts and risks?",
                      store=store, _company=TSLA, _reranker=FakeReranker())
    kwargs = store.query.call_args.kwargs
    assert kwargs["query_text"] == "Tesla (TSLA): What are the key catalysts and risks?"

def test_fetches_wide_candidate_pool_without_store_time_filter():
    store = _store([])
    retrieve_evidence("TSLA", "q", store=store, _company=TSLA, _reranker=FakeReranker())
    kwargs = store.query.call_args.kwargs
    assert kwargs["top_k"] == 30          # RETRIEVAL_FETCH_N
    assert kwargs["days_back"] is None    # time handled by policy, not store


# ── gates ─────────────────────────────────────────────────────────────────────

def test_aboutness_gate_drops_passing_mentions_and_off_topic():
    result = _run([_cand(ON_TOPIC), _cand(PASSING_MENTION), _cand(OFF_TOPIC)])
    texts = [e.snippet for e in result.evidence]
    assert len(texts) == 1 and "Tesla" in texts[0]

def test_article_level_metadata_rescues_pronoun_chunk():
    # Chunk text never names Tesla, but ingestion stored article about_score=1.0
    pronoun_chunk = "The company reiterated its production guidance for the quarter."
    result = _run([_cand(pronoun_chunk, about_meta=1.0)],
                  reranker=FakeReranker(high=8.0, low=8.0))  # reranker passes it
    assert len(result.evidence) == 1

def test_rerank_gate_drops_low_scoring_survivors():
    # Aboutness passes both (both name Tesla); reranker only likes the first.
    # Threshold injected explicitly — the gate must not depend on config values.
    class SplitReranker:
        def predict(self, pairs):
            return [8.0] + [-8.0] * (len(pairs) - 1)
    result = _run([_cand(ON_TOPIC), _cand("Tesla mentioned in unrelated crypto piece.")],
                  reranker=SplitReranker(), rerank_threshold=0.0)
    assert len(result.evidence) == 1

def test_evidence_ordered_by_rerank_score_and_capped_at_top_k():
    class Descending:
        def predict(self, pairs):
            return [float(len(pairs) - i) for i in range(len(pairs))]
    cands = [_cand(f"Tesla item {i}. Tesla news.", distance=0.1 * i) for i in range(8)]
    result = _run(cands, reranker=Descending(), top_k=5)
    assert len(result.evidence) == 5
    scores = [e.relevance_score for e in result.evidence]
    assert scores == sorted(scores, reverse=True)


# ── scores on evidence ────────────────────────────────────────────────────────

def test_evidence_carries_all_three_scores_and_citation_ids():
    result = _run([_cand(ON_TOPIC, distance=0.4)])
    e = result.evidence[0]
    assert e.citation_id == "E1"
    assert e.similarity_score == 0.6
    assert e.aboutness_score == 1.0
    assert e.relevance_score == 8.0


# ── status rules ──────────────────────────────────────────────────────────────

def test_zero_survivors_is_insufficient():
    result = _run([_cand(OFF_TOPIC), _cand(PASSING_MENTION)])
    assert result.evidence_status == "insufficient"
    assert result.evidence == []

def test_one_or_two_survivors_is_partial():
    result = _run([_cand(ON_TOPIC), _cand(ON_TOPIC + " More Tesla.")])
    assert result.evidence_status == "partial"

def test_three_fresh_survivors_is_sufficient():
    result = _run([_cand(ON_TOPIC + f" v{i}") for i in range(3)])
    assert result.evidence_status == "sufficient"

def test_stale_fallback_caps_at_partial():
    result = _run([_cand(ON_TOPIC + f" v{i}", date=STALE) for i in range(3)])
    assert result.evidence_status == "partial"
    assert "older than 30 days" in result.status_reason

def test_fresh_survivors_exclude_stale_ones():
    result = _run([_cand(ON_TOPIC), _cand(ON_TOPIC + " old. Tesla.", date=STALE)])
    assert len(result.evidence) == 1

def test_no_reranker_caps_at_partial():
    # _reranker=None → gates-only fallback (bypasses the _UNSET singleton path)
    result = retrieve_evidence("TSLA", "q", store=_store([_cand(ON_TOPIC + f" v{i}") for i in range(3)]),
                               _company=TSLA, _reranker=None, days_back=30)
    assert result.evidence_status == "partial"
    assert "re-ranker unavailable" in result.status_reason
    assert all(e.relevance_score is None for e in result.evidence)

def test_days_back_none_means_no_time_preference():
    result = _run([_cand(ON_TOPIC + f" v{i}", date=STALE) for i in range(3)], days_back=None)
    assert result.evidence_status == "sufficient"

def test_reranker_singleton_not_resolved_when_nothing_survives_gate():
    # No survivors → nothing to rank → the 22MB model must never be loaded.
    from unittest.mock import patch
    with patch("core.singletons.get_reranker") as mock_get:
        result = retrieve_evidence("TSLA", "q", store=_store([_cand(OFF_TOPIC)]),
                                   _company=TSLA, days_back=30)
    mock_get.assert_not_called()
    assert result.evidence_status == "insufficient"
    assert "re-ranker unavailable" not in result.status_reason


# ── status_reason accounting ──────────────────────────────────────────────────

def test_status_reason_accounts_for_every_rejection():
    class SplitReranker:
        def predict(self, pairs):
            return [8.0] + [-8.0] * (len(pairs) - 1)
    result = _run(
        [_cand(ON_TOPIC), _cand("Tesla in a weak crypto piece."),
         _cand(OFF_TOPIC), _cand(PASSING_MENTION)],
        reranker=SplitReranker(), rerank_threshold=0.0,
    )
    r = result.status_reason
    assert "4 candidates" in r
    assert "2 rejected by aboutness gate" in r
    assert "1 by relevance threshold" in r
    assert "1 passed" in r

def test_fallback_company_source_noted_in_reason():
    fb = CompanyInfo(ticker="TSLA", name="TSLA", aliases=["TSLA"], source="fallback")
    result = retrieve_evidence("TSLA", "q", store=_store([]), _company=fb,
                               _reranker=FakeReranker())
    assert "aliases unavailable" in result.status_reason
