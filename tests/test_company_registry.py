"""Company registry: ticker -> company name + aliases resolution chain."""

import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from unittest.mock import patch

from core.company_registry import CompanyInfo, get_company, _strip_suffixes


# ── suffix stripping ──────────────────────────────────────────────────────────

def test_strip_suffixes_removes_inc():
    assert _strip_suffixes("Tesla, Inc.") == "Tesla"

def test_strip_suffixes_removes_stacked_suffixes():
    assert _strip_suffixes("Example Holdings Ltd.") == "Example"

def test_strip_suffixes_leaves_plain_name():
    assert _strip_suffixes("Apple") == "Apple"


# ── resolution: config override ───────────────────────────────────────────────

def test_config_override_wins(tmp_path):
    info = get_company("TSLA", cache_path=tmp_path / "aliases.json")
    assert info.source == "config"
    assert info.name == "Tesla"
    assert "Tesla" in info.aliases

def test_ticker_normalised_to_upper(tmp_path):
    info = get_company(" tsla ", cache_path=tmp_path / "aliases.json")
    assert info.ticker == "TSLA"


# ── resolution: cache ─────────────────────────────────────────────────────────

def test_cache_hit_skips_yfinance(tmp_path):
    cache = tmp_path / "aliases.json"
    cache.write_text(json.dumps({"NFLX": ["Netflix", "Netflix, Inc."]}))
    with patch("core.company_registry._fetch_yfinance_names") as mock_yf:
        info = get_company("NFLX", cache_path=cache)
    mock_yf.assert_not_called()
    assert info.source == "cache"
    assert info.name == "Netflix"


# ── resolution: yfinance ──────────────────────────────────────────────────────

def test_yfinance_result_is_stripped_and_cached(tmp_path):
    cache = tmp_path / "aliases.json"
    with patch(
        "core.company_registry._fetch_yfinance_names",
        return_value=["Netflix, Inc.", "Netflix Inc"],
    ):
        info = get_company("NFLX", cache_path=cache)
    assert info.source == "yfinance"
    assert info.name == "Netflix"                     # stripped form first
    assert "Netflix, Inc." in info.aliases            # raw form kept too
    saved = json.loads(cache.read_text())
    assert saved["NFLX"][0] == "Netflix"              # cached for next time


# ── resolution: offline fallback ──────────────────────────────────────────────

def test_offline_fallback_uses_bare_ticker(tmp_path):
    with patch("core.company_registry._fetch_yfinance_names", return_value=[]):
        info = get_company("ZZZZ", cache_path=tmp_path / "aliases.json")
    assert info.source == "fallback"
    assert info.aliases == ["ZZZZ"]
    assert info.name == "ZZZZ"

def test_corrupt_cache_is_ignored(tmp_path):
    cache = tmp_path / "aliases.json"
    cache.write_text("{not json")
    with patch("core.company_registry._fetch_yfinance_names", return_value=[]):
        info = get_company("ZZZZ", cache_path=cache)
    assert info.source == "fallback"
