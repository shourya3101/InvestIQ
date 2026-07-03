"""
Retrieval policy for the trustworthy retrieval layer.

VectorStoreManager stays a thin Chroma wrapper; every gating/ranking/status
decision lives here.  Spec: docs/superpowers/specs/2026-07-03-trustworthy-
retrieval-design.md.
"""

from __future__ import annotations

import re

from core.company_registry import CompanyInfo

_WORD_RE_CACHE: dict[str, re.Pattern] = {}


def _word_pattern(term: str, case_sensitive: bool) -> re.Pattern:
    key = f"{'cs' if case_sensitive else 'ci'}:{term}"
    if key not in _WORD_RE_CACHE:
        flags = 0 if case_sensitive else re.IGNORECASE
        _WORD_RE_CACHE[key] = re.compile(rf"\b{re.escape(term)}\b", flags)
    return _WORD_RE_CACHE[key]


def aboutness_score(text: str, company: CompanyInfo) -> float:
    """Score how much *text* is about *company*, in [0, 1].

    Deterministic mention counting: company-name aliases match
    case-insensitively, the ticker symbol matches case-sensitively (so a
    lowercase URL slug is not a mention).  A name mention is worth two
    ticker mentions; the score saturates at 1.0.

    A passing mention (one bare ticker in a list) scores 0.25; a single
    name mention 0.5; two or more name mentions 1.0.
    """
    if not text:
        return 0.0
    name_hits = 0
    for alias in company.aliases:
        if alias.upper() == company.ticker:
            continue  # fallback alias == ticker: counted below as ticker hits
        name_hits += len(_word_pattern(alias, case_sensitive=False).findall(text))
    ticker_hits = len(_word_pattern(company.ticker, case_sensitive=True).findall(text))
    return min(1.0, (2 * name_hits + ticker_hits) / 4.0)
