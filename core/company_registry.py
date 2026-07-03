"""
Company registry — resolves ticker → company name + aliases.

Resolution order:
  1. COMPANY_ALIASES override map in config.py
  2. JSON cache at data/company_aliases.json
  3. yfinance Ticker.info shortName/longName (result cached)
  4. Fallback: the bare ticker, flagged so retrieval can note reduced
     disambiguation power in its status_reason.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from config import COMPANY_ALIASES, COMPANY_ALIASES_CACHE

_SUFFIX_RE = re.compile(
    r",?\s+(Inc\.?|Incorporated|Corp\.?|Corporation|Ltd\.?|Limited"
    r"|PLC|Co\.?|Company|Holdings|Group)\s*$",
    re.IGNORECASE,
)


class CompanyInfo(BaseModel):
    """Resolved company identity used for query building and aboutness."""

    ticker: str
    name: str
    aliases: list[str]
    source: Literal["config", "cache", "yfinance", "fallback"]


def _strip_suffixes(name: str) -> str:
    """Remove trailing corporate suffixes: 'Tesla, Inc.' → 'Tesla'."""
    prev = None
    while prev != name:
        prev = name
        name = _SUFFIX_RE.sub("", name).strip()
    return name


def _read_cache(cache_path: Path) -> dict:
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache(cache_path: Path, cache: dict) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError:
        pass  # the cache is an optimisation; failure to write is never fatal


def _fetch_yfinance_names(ticker: str) -> list[str]:
    """Raw company-name candidates from yfinance; [] on any failure."""
    try:
        import yfinance as yf  # noqa: PLC0415 — lazy: offline paths never import it
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return []
    names: list[str] = []
    for key in ("shortName", "longName"):
        value = info.get(key)
        if value and isinstance(value, str) and value not in names:
            names.append(value)
    return names


def get_company(ticker: str, cache_path: Optional[Path] = None) -> CompanyInfo:
    """Resolve *ticker* to a CompanyInfo via config → cache → yfinance → fallback."""
    ticker = ticker.upper().strip()
    path = cache_path or COMPANY_ALIASES_CACHE

    if ticker in COMPANY_ALIASES:
        aliases = list(COMPANY_ALIASES[ticker])
        return CompanyInfo(ticker=ticker, name=aliases[0], aliases=aliases, source="config")

    cache = _read_cache(path)
    if cache.get(ticker):
        aliases = list(cache[ticker])
        return CompanyInfo(ticker=ticker, name=aliases[0], aliases=aliases, source="cache")

    aliases = []
    for raw in _fetch_yfinance_names(ticker):
        stripped = _strip_suffixes(raw)
        for candidate in (stripped, raw):
            if candidate and candidate not in aliases:
                aliases.append(candidate)
    if aliases:
        cache[ticker] = aliases
        _write_cache(path, cache)
        return CompanyInfo(ticker=ticker, name=aliases[0], aliases=aliases, source="yfinance")

    return CompanyInfo(ticker=ticker, name=ticker, aliases=[ticker], source="fallback")
