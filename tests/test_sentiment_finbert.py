"""
Tests for the FinBERT-backed sentiment agent — written BEFORE implementation (TDD).

Design:
  - _finbert_polarity(label, score) maps FinBERT output to [-1, +1] polarity
  - _score_snippet(text, scorer) calls scorer or falls back to VADER when scorer is None
  - get_finbert_scorer() is a lazy singleton; _load_finbert() is patchable
  - run_sentiment() accepts _scorer= for test injection (never loads real model in tests)

FakeScorer mimics the transformers pipeline interface without loading any model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from agents.sentiment_agent import _finbert_polarity, _score_snippet, run_sentiment
from core.singletons import get_finbert_scorer, reset_finbert_scorer
from core.schemas import (
    EvidenceSchema,
    ResearchOutputSchema,
    SentimentOutputSchema,
)


# ── FakeScorer ────────────────────────────────────────────────────────────────


class FakeScorer:
    """Mimics a transformers text-classification pipeline."""

    def __init__(self, label: str = "Positive", score: float = 0.95):
        self.label = label
        self.score = score
        self.call_count = 0
        self.last_text: str | None = None

    def __call__(self, text: str):
        self.call_count += 1
        self.last_text = text
        return [{"label": self.label, "score": self.score}]


# ── Fixture helpers ───────────────────────────────────────────────────────────


def _evidence(snippet: str, idx: int = 1) -> EvidenceSchema:
    return EvidenceSchema(
        citation_id=f"E{idx}",
        snippet=snippet,
        filepath="test.txt",
        source="test",
        ticker="AAPL",
        date=datetime(2024, 1, 15, tzinfo=timezone.utc),
        similarity_score=0.9,
    )


def _research(snippets: list[str]) -> ResearchOutputSchema:
    return ResearchOutputSchema(
        ticker="AAPL",
        question="q",
        days_back=30,
        evidence=[_evidence(s, i + 1) for i, s in enumerate(snippets)],
        summary="ok",
    )


# ── _finbert_polarity ─────────────────────────────────────────────────────────


def test_finbert_polarity_positive_returns_positive_score():
    assert _finbert_polarity("Positive", 0.9) == pytest.approx(0.9)


def test_finbert_polarity_negative_returns_negative_score():
    assert _finbert_polarity("Negative", 0.8) == pytest.approx(-0.8)


def test_finbert_polarity_neutral_returns_zero():
    assert _finbert_polarity("Neutral", 0.95) == pytest.approx(0.0)


def test_finbert_polarity_is_case_insensitive_positive():
    assert _finbert_polarity("POSITIVE", 0.7) == pytest.approx(0.7)


def test_finbert_polarity_is_case_insensitive_negative():
    assert _finbert_polarity("negative", 0.6) == pytest.approx(-0.6)


def test_finbert_polarity_clamps_positive_above_one():
    assert _finbert_polarity("Positive", 1.5) == pytest.approx(1.0)


def test_finbert_polarity_clamps_negative_below_minus_one():
    assert _finbert_polarity("Negative", 1.5) == pytest.approx(-1.0)


def test_finbert_polarity_unknown_label_returns_zero():
    assert _finbert_polarity("uncertain", 0.9) == pytest.approx(0.0)


# ── _score_snippet ────────────────────────────────────────────────────────────


def test_score_snippet_with_positive_scorer_returns_positive_polarity():
    polarity, label = _score_snippet("Great results.", FakeScorer("Positive", 0.9))
    assert polarity > 0
    assert label == "positive"


def test_score_snippet_with_negative_scorer_returns_negative_polarity():
    polarity, label = _score_snippet("Terrible losses.", FakeScorer("Negative", 0.85))
    assert polarity < 0
    assert label == "negative"


def test_score_snippet_with_neutral_scorer_returns_zero_polarity():
    polarity, label = _score_snippet("Results in line.", FakeScorer("Neutral", 0.92))
    assert polarity == pytest.approx(0.0)
    assert label == "neutral"


def test_score_snippet_passes_text_to_scorer():
    scorer = FakeScorer()
    _score_snippet("Apple earnings beat.", scorer)
    assert scorer.last_text == "Apple earnings beat."


def test_score_snippet_label_is_lowercase():
    _, label = _score_snippet("text", FakeScorer("Positive", 0.9))
    assert label == label.lower()


def test_score_snippet_truncates_long_text_to_512_chars():
    long_text = "x" * 2000
    scorer = FakeScorer()
    _score_snippet(long_text, scorer)
    assert len(scorer.last_text) <= 512


def test_score_snippet_with_none_scorer_falls_back_to_vader():
    # VADER on a clearly positive text should return non-zero compound
    # We just check it doesn't raise and returns valid types
    polarity, label = _score_snippet("Excellent record revenue and strong earnings.", None)
    assert isinstance(polarity, float)
    assert label in ("positive", "negative", "neutral")


def test_score_snippet_vader_label_matches_polarity_sign():
    # VADER neutral text
    polarity, label = _score_snippet("The company released its results.", None)
    if polarity > 0.05:
        assert label == "positive"
    elif polarity < -0.05:
        assert label == "negative"
    else:
        assert label == "neutral"


# ── get_finbert_scorer singleton ──────────────────────────────────────────────


def test_get_finbert_scorer_calls_load_only_once():
    reset_finbert_scorer()  # clear cache so it will re-initialize

    fake = FakeScorer()
    with patch("core.singletons._load_finbert", return_value=fake) as mock_load:
        s1 = get_finbert_scorer()
        s2 = get_finbert_scorer()
        s3 = get_finbert_scorer()

    assert mock_load.call_count == 1
    assert s1 is s2 is s3


def test_get_finbert_scorer_returns_same_instance_on_repeated_calls():
    fake = FakeScorer()
    reset_finbert_scorer(fake)  # inject directly

    assert get_finbert_scorer() is fake
    assert get_finbert_scorer() is fake


def test_reset_finbert_scorer_with_explicit_scorer_injects_it():
    fake = FakeScorer("Neutral", 0.8)
    reset_finbert_scorer(fake)
    assert get_finbert_scorer() is fake


def test_reset_finbert_scorer_no_args_forces_reinitialisation():
    fake1 = FakeScorer()
    fake2 = FakeScorer()
    reset_finbert_scorer(fake1)
    assert get_finbert_scorer() is fake1

    reset_finbert_scorer()  # clear — next call must re-initialize
    with patch("core.singletons._load_finbert", return_value=fake2):
        result = get_finbert_scorer()
    assert result is fake2


def test_get_finbert_scorer_returns_none_when_transformers_unavailable():
    reset_finbert_scorer()
    with patch("core.singletons._load_finbert", side_effect=ImportError("no transformers")):
        result = get_finbert_scorer()
    assert result is None


def test_get_finbert_scorer_caches_none_after_import_failure():
    reset_finbert_scorer()
    with patch("core.singletons._load_finbert", side_effect=ImportError) as mock_load:
        get_finbert_scorer()
        get_finbert_scorer()  # second call must NOT retry loading
    assert mock_load.call_count == 1


# ── run_sentiment integration ─────────────────────────────────────────────────


def test_run_sentiment_uses_injected_scorer_not_singleton():
    scorer = FakeScorer("Positive", 0.95)

    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["Apple revenue soared."])):
        run_sentiment("AAPL", _scorer=scorer)

    assert scorer.call_count == 1


def test_run_sentiment_returns_sentiment_output_schema():
    scorer = FakeScorer("Positive", 0.9)

    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["Strong earnings."])):
        result = run_sentiment("AAPL", _scorer=scorer)

    assert isinstance(result, SentimentOutputSchema)


def test_run_sentiment_positive_evidence_yields_positive_polarity():
    scorer = FakeScorer("Positive", 0.97)

    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["Record profits, best quarter ever."])):
        result = run_sentiment("AAPL", _scorer=scorer)

    assert result.items[0].polarity > 0
    assert result.items[0].label == "positive"


def test_run_sentiment_negative_evidence_yields_negative_polarity():
    scorer = FakeScorer("Negative", 0.95)

    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["Severe loss and restructuring charges."])):
        result = run_sentiment("AAPL", _scorer=scorer)

    assert result.items[0].polarity < 0
    assert result.items[0].label == "negative"


def test_run_sentiment_rationale_mentions_finbert():
    scorer = FakeScorer("Positive", 0.9)

    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["Good quarter."])):
        result = run_sentiment("AAPL", _scorer=scorer)

    assert "FinBERT" in result.items[0].rationale


def test_run_sentiment_overall_score_is_mean_of_item_polarities():
    # Two items: +0.9 and -0.8 → mean = +0.05
    call_no = [0]

    class AlternatingScorer:
        def __call__(self, text):
            call_no[0] += 1
            if call_no[0] == 1:
                return [{"label": "Positive", "score": 0.9}]
            return [{"label": "Negative", "score": 0.8}]

    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["text one", "text two"])):
        result = run_sentiment("AAPL", _scorer=AlternatingScorer())

    expected = round((0.9 + (-0.8)) / 2, 4)
    assert result.overall_score == pytest.approx(expected, abs=1e-3)


def test_run_sentiment_with_none_scorer_uses_vader_fallback():
    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["The company reported quarterly results."])):
        result = run_sentiment("AAPL", _scorer=None)

    assert isinstance(result, SentimentOutputSchema)
    assert result.overall_label in ("positive", "negative", "neutral")


def test_run_sentiment_vader_rationale_mentions_vader():
    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["Quarterly results released."])):
        result = run_sentiment("AAPL", _scorer=None)

    assert "VADER" in result.items[0].rationale


def test_run_sentiment_with_no_evidence_returns_neutral():
    with patch("agents.sentiment_agent.run_research",
               return_value=_research([])):
        result = run_sentiment("AAPL", _scorer=FakeScorer())

    assert result.overall_label == "neutral"
    assert result.overall_score == 0.0


def test_run_sentiment_scorer_called_once_per_evidence_item():
    scorer = FakeScorer("Neutral", 0.9)

    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["a", "b", "c"])):
        run_sentiment("AAPL", _scorer=scorer)

    assert scorer.call_count == 3


def test_run_sentiment_does_not_call_get_finbert_scorer_when_scorer_injected():
    scorer = FakeScorer()

    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["text"])):
        with patch("agents.sentiment_agent.get_finbert_scorer") as mock_get:
            run_sentiment("AAPL", _scorer=scorer)

    mock_get.assert_not_called()


def test_run_sentiment_calls_get_finbert_scorer_when_no_scorer_injected():
    fake_scorer = FakeScorer()

    with patch("agents.sentiment_agent.run_research",
               return_value=_research(["text"])):
        with patch("agents.sentiment_agent.get_finbert_scorer", return_value=fake_scorer) as mock_get:
            run_sentiment("AAPL")  # no _scorer injected

    mock_get.assert_called_once()
