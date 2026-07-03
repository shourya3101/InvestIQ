"""aboutness_score: deterministic company-mention scoring in [0, 1]."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.company_registry import CompanyInfo
from core.retrieval import aboutness_score

TSLA = CompanyInfo(ticker="TSLA", name="Tesla", aliases=["Tesla", "Tesla, Inc."], source="config")
FALLBACK = CompanyInfo(ticker="ZZZZ", name="ZZZZ", aliases=["ZZZZ"], source="fallback")

# Real failure case from the RQ1 eval: TSLA listed once among seven symbols.
BYBIT_CHUNK = (
    "Bybit Introduces 24/7 TradFi Perpetual Contracts Trading for Dozens of US "
    "Stocks and Global ETFs. Bybit has expanded perpetual contracts offerings to "
    "include seven new TradFi assets: TSLA, AMZN, META, GOOGL, MSFT, AVGO, LLY."
)

TESLA_CHUNK = (
    "Tesla reported record quarterly deliveries as the company expanded Model Y "
    "production. Tesla shares rose 4% after the announcement, and TSLA remains "
    "the most traded EV name."
)


def test_empty_text_scores_zero():
    assert aboutness_score("", TSLA) == 0.0

def test_no_mention_scores_zero():
    assert aboutness_score("Intel stock surged 190% in 2026.", TSLA) == 0.0

def test_single_bare_ticker_mention_scores_low():
    # 1 ticker hit → 1/4 = 0.25: below the 0.3 provisional gate
    assert aboutness_score("Watchlist: TSLA among others.", TSLA) == 0.25

def test_single_name_mention_scores_half():
    # 1 name hit → 2/4 = 0.5
    assert aboutness_score("Tesla announced a new factory.", TSLA) == 0.5

def test_score_saturates_at_one():
    assert aboutness_score(TESLA_CHUNK, TSLA) == 1.0

def test_bybit_regression_chunk_scores_below_gate():
    from config import ABOUTNESS_THRESHOLD
    assert aboutness_score(BYBIT_CHUNK, TSLA) < ABOUTNESS_THRESHOLD

def test_genuine_tesla_chunk_scores_above_gate():
    from config import ABOUTNESS_THRESHOLD
    assert aboutness_score(TESLA_CHUNK, TSLA) >= ABOUTNESS_THRESHOLD

def test_ticker_match_is_case_sensitive():
    # lowercase 'tsla' (e.g. in a URL slug) is not a ticker mention
    assert aboutness_score("read more at example.com/tsla-news", TSLA) == 0.0

def test_name_match_is_case_insensitive():
    assert aboutness_score("TESLA results beat estimates.", TSLA) == 0.5

def test_word_boundary_no_substring_match():
    # 'Teslaphile' must not count as a 'Tesla' mention
    assert aboutness_score("The Teslaphile community cheered.", TSLA) == 0.0

def test_fallback_alias_equal_to_ticker_counts_once_as_ticker():
    # alias list is just ["ZZZZ"]: one mention = 1 ticker hit → 0.25, not 0.75
    assert aboutness_score("ZZZZ is listed here.", FALLBACK) == 0.25

def test_monotonic_in_mentions():
    one = aboutness_score("Tesla did a thing.", TSLA)
    two = aboutness_score("Tesla did a thing. Tesla did another.", TSLA)
    assert two >= one
