# Trustworthy Retrieval Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieval that gates evidence on company-aboutness and cross-encoder relevance, and a typed insufficient-evidence state that propagates so downstream agents refuse to fabricate.

**Architecture:** New `core/retrieval.py` policy module (dense fetch 30 → aboutness gate → cross-encoder re-rank → time preference → typed status) on top of the untouched `VectorStoreManager`. Additive schema fields carry status through research → sentiment → risk → analyst → coordinator → frontend. Spec: `docs/superpowers/specs/2026-07-03-trustworthy-retrieval-design.md`.

**Tech Stack:** Python 3.11, Pydantic v2, ChromaDB, sentence-transformers (`CrossEncoder`), yfinance, pytest, vanilla JS frontend.

## Global Constraints

- Run all commands from project root `C:\Users\shour\OneDrive\Desktop\URECA`; use the venv interpreter: `venv\Scripts\python.exe -m pytest ...`
- TDD: write the failing test first, watch it fail, then implement.
- Full suite (`venv\Scripts\python.exe -m pytest tests/ -v`) must pass after every task; 234 existing tests stay green.
- All schema changes are additive with defaults — constructing any schema with pre-change arguments must still validate.
- Never call real yfinance, NewsAPI, ChromaDB, SentenceTransformer, or CrossEncoder in tests — inject or patch.
- Locked decisions (do not revisit): cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, candidate pool 30, query-time gating, no BM25, no corpus migration.
- Two USER REVIEW GATES: Task 13 (thresholds — show FRR on positives, surface any negatives-vs-positives tradeoff to the user, never pick a compromise silently) and Task 14 (user sees the TSLA degraded report render before done).
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: Config constants + company registry

**Files:**
- Modify: `config.py` (append after the `OPENAI_MODEL` block)
- Create: `core/company_registry.py`
- Test: `tests/test_company_registry.py`

**Interfaces:**
- Produces: `get_company(ticker: str, cache_path: Optional[Path] = None) -> CompanyInfo`; `CompanyInfo(ticker: str, name: str, aliases: list[str], source: Literal["config","cache","yfinance","fallback"])`; config constants `RETRIEVAL_FETCH_N`, `RERANK_MODEL_NAME`, `ABOUTNESS_THRESHOLD`, `RERANK_THRESHOLD`, `MIN_SUFFICIENT_EVIDENCE`, `COMPANY_ALIASES`, `COMPANY_ALIASES_CACHE`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_company_registry.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_company_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.company_registry'`

- [ ] **Step 3: Append retrieval constants to `config.py`**

Append after the `OPENAI_MODEL = "gpt-4o-mini"` line (keep `llm_enabled()` last):

```python
# ── Retrieval trust layer ────────────────────────────────────────────────────
RETRIEVAL_FETCH_N = 30
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# PROVISIONAL thresholds — locked only after evaluation/calibrate_retrieval.py
# is run and the user reviews the false-reject rate (spec §7). Do not tune by hand.
ABOUTNESS_THRESHOLD = 0.3
RERANK_THRESHOLD = 0.0

MIN_SUFFICIENT_EVIDENCE = 3

# Manual alias overrides checked before cache/yfinance (core/company_registry.py)
COMPANY_ALIASES: dict[str, list[str]] = {
    "AAPL": ["Apple", "Apple Inc"],
    "MSFT": ["Microsoft", "Microsoft Corporation"],
    "NVDA": ["Nvidia", "NVIDIA", "Nvidia Corporation"],
    "GOOGL": ["Google", "Alphabet", "Alphabet Inc"],
    "TSLA": ["Tesla", "Tesla, Inc."],
}
COMPANY_ALIASES_CACHE = DATA_DIR / "company_aliases.json"
```

- [ ] **Step 4: Create `core/company_registry.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_company_registry.py -v`
Expected: 8 PASS

- [ ] **Step 6: Full suite, then commit**

Run: `venv\Scripts\python.exe -m pytest tests/ -q` — expected: all pass.

```bash
git add config.py core/company_registry.py tests/test_company_registry.py
git commit -m "feat: company registry (config->cache->yfinance->fallback) + retrieval config"
```

---

### Task 2: Aboutness score (pure function)

**Files:**
- Create: `core/retrieval.py` (module started here; grown in Task 5)
- Test: `tests/test_aboutness.py`

**Interfaces:**
- Consumes: `CompanyInfo` from Task 1.
- Produces: `aboutness_score(text: str, company: CompanyInfo) -> float` in `[0,1]`. Formula: `min(1.0, (2*name_hits + ticker_hits) / 4.0)` — name matches case-insensitive word-boundary, ticker matches case-sensitive word-boundary; an alias equal to the ticker counts as ticker, not name.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aboutness.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_aboutness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.retrieval'`

- [ ] **Step 3: Create `core/retrieval.py` with the scorer**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_aboutness.py -v`
Expected: 12 PASS

- [ ] **Step 5: Full suite, then commit**

```bash
git add core/retrieval.py tests/test_aboutness.py
git commit -m "feat: deterministic company-aboutness scorer with TSLA regression fixtures"
```

---

### Task 3: Cross-encoder re-ranker singleton

**Files:**
- Modify: `core/singletons.py` (append)
- Test: `tests/test_reranker_singleton.py`

**Interfaces:**
- Produces: `get_reranker() -> Optional[CrossEncoder]` (None = load failed → gates-only fallback); `reset_reranker(reranker=_RERANKER_NOT_INIT)` test hook. Mirrors the FinBERT singleton pattern exactly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reranker_singleton.py
"""Cross-encoder singleton: lazy load, cached failure, test injection."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

from core import singletons
from core.singletons import get_reranker, reset_reranker


def teardown_function():
    reset_reranker()  # never leak state between tests


def test_loads_once_and_caches():
    fake = MagicMock(name="cross_encoder")
    with patch.object(singletons, "_load_reranker", return_value=fake) as loader:
        assert get_reranker() is fake
        assert get_reranker() is fake
    loader.assert_called_once()


def test_load_failure_caches_none():
    with patch.object(singletons, "_load_reranker", side_effect=RuntimeError("no model")) as loader:
        assert get_reranker() is None
        assert get_reranker() is None   # failure cached, not retried
    loader.assert_called_once()


def test_reset_injects_fake():
    fake = MagicMock()
    reset_reranker(fake)
    assert get_reranker() is fake


def test_reset_no_args_forces_reload():
    reset_reranker(MagicMock())
    reset_reranker()
    with patch.object(singletons, "_load_reranker", return_value="fresh"):
        assert get_reranker() == "fresh"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_reranker_singleton.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_reranker'`

- [ ] **Step 3: Append to `core/singletons.py`**

```python
# ── Cross-encoder re-ranker singleton ─────────────────────────────────────────

# Sentinel: distinguishes "not yet attempted" from None ("load failed / gates-only")
_RERANKER_NOT_INIT = object()
_reranker = _RERANKER_NOT_INIT


def _load_reranker():
    """Load and return the CrossEncoder re-ranker.

    Separated from get_reranker() so tests can patch this function without
    importing sentence_transformers at module load time.
    """
    from sentence_transformers import CrossEncoder  # noqa: PLC0415
    from config import RERANK_MODEL_NAME  # noqa: PLC0415
    return CrossEncoder(RERANK_MODEL_NAME)


def get_reranker():
    """Return the shared CrossEncoder, loading it on first call.

    Returns None when loading fails, which signals retrieval to fall back
    to aboutness + cosine gating and cap evidence_status at "partial".
    """
    global _reranker
    if _reranker is _RERANKER_NOT_INIT:
        try:
            _reranker = _load_reranker()
        except Exception:
            _reranker = None  # cache the failure → gates-only on every call
    return _reranker


def reset_reranker(reranker=_RERANKER_NOT_INIT) -> None:
    """Replace or reset the cached re-ranker.  Intended for tests only."""
    global _reranker
    _reranker = reranker
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_reranker_singleton.py -v`
Expected: 4 PASS

- [ ] **Step 5: Full suite, then commit**

```bash
git add core/singletons.py tests/test_reranker_singleton.py
git commit -m "feat: lazy CrossEncoder re-ranker singleton with gates-only fallback"
```

---

### Task 4: Additive schema fields

**Files:**
- Modify: `core/schemas.py`
- Test: `tests/test_schema_additions.py`

**Interfaces:**
- Produces (all additive, spec §4):
  - `EvidenceSchema.relevance_score: Optional[float] = None`, `EvidenceSchema.aboutness_score: Optional[float] = None`
  - `ResearchOutputSchema.evidence_status: Literal["sufficient","partial","insufficient"] = "sufficient"`, `ResearchOutputSchema.status_reason: str = ""`
  - `SentimentOutputSchema.data_status: Literal["ok","no_data"] = "ok"`
  - `ActionSignalSchema.signal` Literal gains `"no_view"`
  - `InvestmentMemoSchema.evidence_status: Literal[...] = "sufficient"`, `InvestmentMemoSchema.debate_skipped_reason: Optional[str] = None`
  - `DocumentSchema.about_score: Optional[float] = None` (article-level, ingestion; needed by spec §6)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schema_additions.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_schema_additions.py -v`
Expected: FAIL — `AttributeError` / `ValidationError` on the new fields.

- [ ] **Step 3: Apply the schema edits**

In `core/schemas.py` make exactly these changes:

```python
class DocumentSchema(BaseModel):
    """Schema for a document ingested into the vector store."""

    content: str
    source: str
    ticker: Optional[str] = None
    date: Optional[datetime] = None
    filepath: str
    about_score: Optional[float] = None  # article-level aboutness (ingestion)


class EvidenceSchema(BaseModel):
    """Schema for a single piece of retrieved evidence."""

    citation_id: str
    snippet: str
    filepath: str
    source: str
    ticker: Optional[str] = None
    date: Optional[datetime] = None
    similarity_score: float
    aboutness_score: Optional[float] = None   # company-mention gate score
    relevance_score: Optional[float] = None   # cross-encoder score


class ResearchOutputSchema(BaseModel):
    """Schema for the full output of a research query."""

    ticker: str
    question: str
    days_back: Optional[int] = None
    evidence: List[EvidenceSchema] = Field(default_factory=list)
    summary: str
    evidence_status: Literal["sufficient", "partial", "insufficient"] = "sufficient"
    status_reason: str = ""
```

In `SentimentOutputSchema`, add as the last field:

```python
    data_status: Literal["ok", "no_data"] = "ok"
```

In `ActionSignalSchema`, change the signal line to:

```python
    signal: Literal["buy", "hold", "sell", "watch", "no_view"]
```

In `InvestmentMemoSchema`, add after `memory`:

```python
    evidence_status: Literal["sufficient", "partial", "insufficient"] = "sufficient"
    debate_skipped_reason: Optional[str] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_schema_additions.py -v`
Expected: 8 PASS

- [ ] **Step 5: Full suite (backward-compat proof), then commit**

Run: `venv\Scripts\python.exe -m pytest tests/ -q` — all 234+ pass, proving additivity.

```bash
git add core/schemas.py tests/test_schema_additions.py
git commit -m "feat: additive schema fields for evidence status, no_view, debate skip"
```

---

### Task 5: Retrieval pipeline (`retrieve_evidence`)

**Files:**
- Modify: `core/retrieval.py`
- Test: `tests/test_retrieval_pipeline.py`

**Interfaces:**
- Consumes: `aboutness_score` (Task 2), `get_reranker` (Task 3), `get_company` (Task 1), schema fields (Task 4), `VectorStoreManager.query(ticker, query_text, top_k, days_back, allow_fallback)` returning `[{"text", "metadata", "distance", "time_filtered"}]` (unchanged).
- Produces:
  ```python
  retrieve_evidence(
      ticker: str, question: str, days_back: Optional[int] = 30, top_k: int = 5,
      store: Optional[VectorStoreManager] = None,
      aboutness_threshold: Optional[float] = None,   # None → config
      rerank_threshold: Optional[float] = None,      # None → config
      _reranker=_UNSET,                              # test hook; None forces gates-only
      _company: Optional[CompanyInfo] = None,        # test hook, skips registry
  ) -> RetrievalResult
  ```
  `RetrievalResult(ticker, query_text, evidence: list[EvidenceSchema], evidence_status, status_reason)`. Citation ids `E1..En` assigned here.
- Status rules: 0 items → `insufficient`; 1–2 → `partial`; ≥3 but stale-fallback used or re-ranker unavailable → `partial`; else `sufficient`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieval_pipeline.py
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
    chunk = "The company reiterated its production guidance for the quarter and Tesla margins."
    pronoun_chunk = "The company reiterated its production guidance for the quarter."
    result = _run([_cand(pronoun_chunk, about_meta=1.0)],
                  reranker=FakeReranker(high=8.0, low=8.0))  # reranker passes it
    assert len(result.evidence) == 1

def test_rerank_gate_drops_low_scoring_survivors():
    # Aboutness passes both (both name Tesla); reranker only likes the first.
    class SplitReranker:
        def predict(self, pairs):
            return [8.0] + [-8.0] * (len(pairs) - 1)
    result = _run([_cand(ON_TOPIC), _cand("Tesla mentioned in unrelated crypto piece.")],
                  reranker=SplitReranker())
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


# ── status_reason accounting ──────────────────────────────────────────────────

def test_status_reason_accounts_for_every_rejection():
    class SplitReranker:
        def predict(self, pairs):
            return [8.0] + [-8.0] * (len(pairs) - 1)
    result = _run(
        [_cand(ON_TOPIC), _cand("Tesla in a weak crypto piece."),
         _cand(OFF_TOPIC), _cand(PASSING_MENTION)],
        reranker=SplitReranker(),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_retrieval_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'retrieve_evidence'`

- [ ] **Step 3: Implement in `core/retrieval.py`**

Append below `aboutness_score` (add the new imports at the top of the file):

```python
from datetime import datetime, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, Field

from config import (
    ABOUTNESS_THRESHOLD,
    MIN_SUFFICIENT_EVIDENCE,
    RERANK_THRESHOLD,
    RETRIEVAL_FETCH_N,
)
from core.company_registry import get_company
from core.schemas import EvidenceSchema

_UNSET = object()


class RetrievalResult(BaseModel):
    """Gated evidence plus the typed sufficiency verdict."""

    ticker: str
    query_text: str
    evidence: list[EvidenceSchema] = Field(default_factory=list)
    evidence_status: Literal["sufficient", "partial", "insufficient"]
    status_reason: str


def _parse_meta_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


def _to_evidence(idx: int, cand: dict) -> EvidenceSchema:
    meta = cand["metadata"]
    similarity = max(0.0, min(1.0, 1.0 - cand["distance"]))
    return EvidenceSchema(
        citation_id=f"E{idx}",
        snippet=cand["text"].strip()[:280],
        filepath=meta.get("filepath", ""),
        source=meta.get("source", ""),
        ticker=meta.get("ticker", "") or None,
        date=_parse_meta_date(meta.get("date", "")),
        similarity_score=round(similarity, 4),
        aboutness_score=cand.get("aboutness"),
        relevance_score=cand.get("relevance"),
    )


def retrieve_evidence(
    ticker: str,
    question: str,
    days_back: Optional[int] = 30,
    top_k: int = 5,
    store=None,
    aboutness_threshold: Optional[float] = None,
    rerank_threshold: Optional[float] = None,
    _reranker=_UNSET,
    _company=None,
) -> RetrievalResult:
    """Dense fetch → aboutness gate → cross-encoder re-rank → time preference →
    typed evidence status.  See module docstring for the spec reference."""
    ticker = ticker.upper().strip()
    about_min = ABOUTNESS_THRESHOLD if aboutness_threshold is None else aboutness_threshold
    rerank_min = RERANK_THRESHOLD if rerank_threshold is None else rerank_threshold

    if store is None:
        from core.singletons import get_store  # noqa: PLC0415 — avoid heavy import at module load
        store = get_store()
    company = _company if _company is not None else get_company(ticker)
    if _reranker is _UNSET:
        from core.singletons import get_reranker  # noqa: PLC0415
        reranker = get_reranker()
    else:
        reranker = _reranker

    query_text = f"{company.name} ({ticker}): {question}"

    candidates = store.query(
        ticker=ticker,
        query_text=query_text,
        top_k=RETRIEVAL_FETCH_N,
        days_back=None,  # time preference applied here, after the quality gates
    )
    n_total = len(candidates)

    # ── aboutness gate ────────────────────────────────────────────────────────
    survivors: list[dict] = []
    for cand in candidates:
        chunk_score = aboutness_score(cand["text"], company)
        meta_score = cand["metadata"].get("about_score")
        effective = max(chunk_score, float(meta_score)) if meta_score is not None else chunk_score
        if effective >= about_min:
            cand["aboutness"] = round(effective, 4)
            survivors.append(cand)
    n_about_rejected = n_total - len(survivors)

    # ── cross-encoder re-rank ────────────────────────────────────────────────
    reranker_used = reranker is not None
    n_rerank_rejected = 0
    if reranker_used and survivors:
        scores = reranker.predict([(query_text, c["text"]) for c in survivors])
        for cand, score in zip(survivors, scores):
            cand["relevance"] = round(float(score), 4)
        before = len(survivors)
        survivors = [c for c in survivors if c["relevance"] >= rerank_min]
        n_rerank_rejected = before - len(survivors)
        survivors.sort(key=lambda c: c["relevance"], reverse=True)
    else:
        survivors.sort(key=lambda c: c["distance"])

    # ── time preference (never silent) ───────────────────────────────────────
    stale_fallback = False
    if days_back is not None and survivors:
        cutoff = datetime.now() - timedelta(days=days_back)
        fresh = [
            c for c in survivors
            if (d := _parse_meta_date(c["metadata"].get("date", ""))) is not None and d >= cutoff
        ]
        if fresh:
            survivors = fresh
        else:
            stale_fallback = True

    final = survivors[:top_k]
    evidence = [_to_evidence(i, c) for i, c in enumerate(final, 1)]

    # ── status ────────────────────────────────────────────────────────────────
    if not evidence:
        status = "insufficient"
    elif len(evidence) < MIN_SUFFICIENT_EVIDENCE or stale_fallback or not reranker_used:
        status = "partial"
    else:
        status = "sufficient"

    parts = [
        f"{n_total} candidates retrieved; "
        f"{n_about_rejected} rejected by aboutness gate, "
        f"{n_rerank_rejected} by relevance threshold; "
        f"{len(evidence)} passed — {status} evidence."
    ]
    if stale_fallback:
        parts.append(f"All surviving evidence is older than {days_back} days (stale fallback).")
    if not reranker_used:
        parts.append("Cross-encoder re-ranker unavailable — cosine ordering only.")
    if company.source == "fallback":
        parts.append("Company aliases unavailable (offline) — ticker-only matching.")

    return RetrievalResult(
        ticker=ticker,
        query_text=query_text,
        evidence=evidence,
        evidence_status=status,
        status_reason=" ".join(parts),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_retrieval_pipeline.py -v`
Expected: 16 PASS

- [ ] **Step 5: Full suite, then commit**

```bash
git add core/retrieval.py tests/test_retrieval_pipeline.py
git commit -m "feat: gated retrieval pipeline with typed evidence status"
```

---

### Task 6: Research agent delegates to the pipeline

**Files:**
- Modify: `agents/research_agent.py` (replace `run_research` body; keep signature and the `__main__` block)
- Test: `tests/test_research_agent_status.py`

**Interfaces:**
- Consumes: `retrieve_evidence` (Task 5).
- Produces: `run_research(ticker, question, days_back=30, top_k=5, store=None) -> ResearchOutputSchema` — unchanged signature; output now carries `evidence_status`/`status_reason` and per-item scores.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_research_agent_status.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_research_agent_status.py -v`
Expected: FAIL — `AttributeError: ... does not have the attribute 'retrieve_evidence'`

- [ ] **Step 3: Rewrite `run_research`**

Replace the entire body of `run_research` in `agents/research_agent.py` (and slim the imports — `hashlib`-era helpers go away):

```python
"""
Research Agent v3 — delegates retrieval policy to core.retrieval.

Packages gated evidence into ResearchOutputSchema, carrying the typed
evidence_status so downstream agents can refuse to fabricate.
No LLM calls — this agent only retrieves and structures evidence.
"""

from typing import Optional

from config import DEFAULT_TICKER
from core.retrieval import retrieve_evidence
from core.schemas import ResearchOutputSchema
from core.vector_store_manager import VectorStoreManager


def run_research(
    ticker: str,
    question: str,
    days_back: Optional[int] = 30,
    top_k: int = 5,
    store: Optional[VectorStoreManager] = None,
) -> ResearchOutputSchema:
    """Run gated retrieval and return a validated ResearchOutputSchema."""
    result = retrieve_evidence(
        ticker=ticker,
        question=question,
        days_back=days_back,
        top_k=top_k,
        store=store,
    )

    if result.evidence_status == "insufficient":
        summary = f"No trustworthy evidence found for {result.ticker}. {result.status_reason}"
    elif result.evidence_status == "partial":
        summary = f"Partial evidence for {result.ticker}. {result.status_reason}"
    else:
        summary = f"Sufficient evidence for {result.ticker}. {result.status_reason}"

    return ResearchOutputSchema(
        ticker=result.ticker,
        question=question,
        days_back=days_back,
        evidence=result.evidence,
        summary=summary,
        evidence_status=result.evidence_status,
        status_reason=result.status_reason,
    )
```

Keep the existing `__main__` smoke block as-is (it only calls `run_research`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_research_agent_status.py -v`
Expected: 4 PASS

- [ ] **Step 5: Full suite, then commit**

```bash
git add agents/research_agent.py tests/test_research_agent_status.py
git commit -m "feat: research agent delegates to gated retrieval, surfaces status"
```

---

### Task 7: Sentiment consumes the research evidence pack

**Files:**
- Modify: `agents/sentiment_agent.py`
- Test: `tests/test_sentiment_evidence.py`

**Interfaces:**
- Produces: `run_sentiment(ticker, question=..., window_days=365, top_k=5, store=None, research: Optional[ResearchOutputSchema] = None, _scorer=_UNSET) -> SentimentOutputSchema`. When `research` is provided, no retrieval happens. `data_status="no_data"` when `research.evidence_status == "insufficient"` or the evidence list is empty; FinBERT/VADER never invoked in that case.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sentiment_evidence.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_sentiment_evidence.py -v`
Expected: FAIL — `TypeError: run_sentiment() got an unexpected keyword argument 'research'`

- [ ] **Step 3: Modify `run_sentiment`**

Add the import `from core.schemas import ..., ResearchOutputSchema` and change the signature + retrieval section of `run_sentiment` in `agents/sentiment_agent.py`:

```python
def run_sentiment(
    ticker: str,
    question: str = "What are the key catalysts and risks?",
    window_days: int = 365,
    top_k: int = 5,
    store: Optional[VectorStoreManager] = None,
    research: Optional[ResearchOutputSchema] = None,
    _scorer=_UNSET,
) -> SentimentOutputSchema:
```

Replace the `# ── Retrieve evidence ──` block with:

```python
    # ── Resolve evidence pack (coordinator passes research; standalone retrieves) ──
    if research is None:
        research = run_research(
            ticker=ticker,
            question=question,
            days_back=window_days,
            top_k=top_k,
            store=store,
        )

    # ── Refuse to score when there is nothing trustworthy to score ────────────
    if research.evidence_status == "insufficient" or not research.evidence:
        return SentimentOutputSchema(
            ticker=ticker,
            as_of=datetime.now(timezone.utc),
            window_days=window_days,
            overall_score=0.0,
            overall_label="neutral",
            items=[],
            summary="No sentiment data — no trustworthy evidence to score.",
            data_status="no_data",
        )
```

(The scorer resolution lines move *below* this guard so FinBERT is never
loaded on the no-data path; the rest of the function is unchanged and its
final `SentimentOutputSchema(...)` keeps the default `data_status="ok"`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_sentiment_evidence.py tests/test_sentiment_finbert.py -v`
Expected: new 4 PASS + all existing FinBERT tests PASS.

- [ ] **Step 5: Full suite, then commit**

```bash
git add agents/sentiment_agent.py tests/test_sentiment_evidence.py
git commit -m "feat: sentiment consumes research pack, no_data on insufficient evidence"
```

---

### Task 8: Risk agent — accept shared outputs, flag missing sentiment

**Files:**
- Modify: `agents/risk_agent.py`
- Test: `tests/test_risk_no_sentiment.py`

**Interfaces:**
- Produces: `run_risk(ticker, mode="live", price_filepath=None, question=..., window_days=365, top_k=5, store=None, trend: Optional[TrendOutputSchema] = None, sentiment: Optional[SentimentOutputSchema] = None) -> RiskOutputSchema`. Provided outputs are used as-is (no recompute); `data_status=="no_data"` adds a low-severity informational sentiment flag and never adds sentiment risk points (the existing scorer only adds points for negative sentiment, so no reweighting is required — the flag prevents fabricated neutrality from passing unremarked).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_risk_no_sentiment.py
"""Risk agent: injected outputs are reused; no_data sentiment is flagged, not scored."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from unittest.mock import patch

from agents.risk_agent import run_risk
from core.schemas import (
    SentimentOutputSchema,
    TrendOutputSchema,
    TrendSignalSchema,
)


def _trend():
    return TrendOutputSchema(
        ticker="TSLA", mode="live", as_of=datetime.now(timezone.utc),
        signals=[TrendSignalSchema(horizon="30d", return_pct=1.0,
                                   volatility_pct=10.0, max_drawdown_pct=-1.0,
                                   trend_label="neutral")],
        summary="calm",
    )


def _sentiment(data_status="ok", label="neutral", score=0.0):
    return SentimentOutputSchema(
        ticker="TSLA", as_of=datetime.now(timezone.utc), window_days=30,
        overall_score=score, overall_label=label, items=[], summary="s",
        data_status=data_status,
    )


def test_injected_outputs_skip_recompute():
    with patch("agents.risk_agent.run_trend") as mock_t, \
         patch("agents.risk_agent.run_sentiment") as mock_s:
        run_risk("TSLA", trend=_trend(), sentiment=_sentiment())
    mock_t.assert_not_called()
    mock_s.assert_not_called()

def test_no_data_sentiment_adds_informational_flag():
    result = run_risk("TSLA", trend=_trend(), sentiment=_sentiment("no_data"))
    sentiment_flags = [f for f in result.flags if f.category == "sentiment"]
    assert len(sentiment_flags) == 1
    assert sentiment_flags[0].severity == "low"
    assert "No sentiment data" in sentiment_flags[0].message

def test_no_data_sentiment_adds_no_risk_points():
    baseline = run_risk("TSLA", trend=_trend(), sentiment=_sentiment("ok"))
    no_data = run_risk("TSLA", trend=_trend(), sentiment=_sentiment("no_data"))
    assert no_data.risk_score == baseline.risk_score

def test_negative_sentiment_still_scores_when_data_ok():
    result = run_risk("TSLA", trend=_trend(),
                      sentiment=_sentiment("ok", label="negative", score=-0.5))
    assert any(f.category == "sentiment" and f.severity == "high" for f in result.flags)

def test_standalone_path_still_computes():
    with patch("agents.risk_agent.run_trend", return_value=_trend()) as mock_t, \
         patch("agents.risk_agent.run_sentiment", return_value=_sentiment()) as mock_s:
        run_risk("TSLA")
    mock_t.assert_called_once()
    mock_s.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_risk_no_sentiment.py -v`
Expected: FAIL — `TypeError: run_risk() got an unexpected keyword argument 'trend'`

- [ ] **Step 3: Modify `run_risk`**

Add imports `from core.schemas import ..., SentimentOutputSchema, TrendOutputSchema`, extend the signature, and replace the gather block:

```python
def run_risk(
    ticker: str,
    mode: str = "live",
    price_filepath: str | None = None,
    question: str = "What are the key catalysts and risks?",
    window_days: int = 365,
    top_k: int = 5,
    store: Optional[VectorStoreManager] = None,
    trend: Optional[TrendOutputSchema] = None,
    sentiment: Optional[SentimentOutputSchema] = None,
) -> RiskOutputSchema:
```

```python
    # ── Gather sub-agent outputs (reused when the coordinator injects them) ──
    if trend is None:
        trend = run_trend(
            ticker,
            mode=mode,
            filepath=price_filepath if mode == "offline" else None,
        )
    sent = sentiment
    if sent is None:
        sent = run_sentiment(
            ticker,
            question=question,
            window_days=window_days,
            top_k=top_k,
            store=store,
        )
```

Replace the `# 4) Sentiment` block with:

```python
    # 4) Sentiment — absence of data is flagged, never scored as neutrality
    if sent.data_status == "no_data":
        flags.append(
            RiskFlagSchema(
                category="sentiment",
                severity="low",
                message="No sentiment data — insufficient trustworthy evidence.",
            )
        )
    elif sent.overall_label == "negative":
        sev = "high" if sent.overall_score <= -0.3 else "moderate"
        flags.append(
            RiskFlagSchema(
                category="sentiment",
                severity=sev,
                message=f"Overall sentiment is negative (score {sent.overall_score:.2f}).",
            )
        )
        risk_score += _SENTIMENT_POINTS[sev]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_risk_no_sentiment.py -v`
Expected: 5 PASS

- [ ] **Step 5: Full suite, then commit**

```bash
git add agents/risk_agent.py tests/test_risk_no_sentiment.py
git commit -m "feat: risk agent reuses injected outputs, flags missing sentiment data"
```

---

### Task 9: Analyst — no-view on insufficient, capped confidence on partial

**Files:**
- Modify: `agents/analyst_agent.py`
- Test: `tests/test_analyst_degraded.py`

**Interfaces:**
- Consumes: `research.evidence_status` / `research.status_reason` (Task 4/6).
- Produces: unchanged signature. New module constant `_PARTIAL_CONFIDENCE_CAP = 0.6`. New helpers `_write_market_data_thesis(provider, ctx, status_reason) -> str` and `_insufficient_memo(...) -> InvestmentMemoSchema`. Every returned memo sets `evidence_status=research.evidence_status`. On `insufficient`: `signal="no_view"`, `confidence=0.0`, recommendation/catalysts LLM calls are NOT made, `catalysts=[]`, `citations=[]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_analyst_degraded.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_analyst_degraded.py -v`
Expected: FAIL — no_view path missing (`AssertionError` on signal / call counts).

- [ ] **Step 3: Implement in `agents/analyst_agent.py`**

Add constant near `_MAX_TOKENS`:

```python
_PARTIAL_CONFIDENCE_CAP = 0.6
```

Add two functions after `_write_thesis`:

```python
def _write_market_data_thesis(provider, ctx: _MemoContext, status_reason: str) -> str:
    """Thesis for the insufficient-evidence path: market data only, no fabrication."""
    system = (
        "You are a senior equity analyst. Write only from the market data provided. "
        "You have NO news or document evidence — do not invent company events, "
        "products, or fundamentals. Use complete sentences, no bullets or headers."
    )
    user = (
        f"Retrieval found no trustworthy evidence for {ctx.ticker} ({status_reason}).\n"
        f"Write a 3-4 sentence market-data-only summary for {ctx.ticker}. "
        f"State first that no reliable evidence was found and no investment view is taken.\n\n"
        f"Market data:\n"
        f"- Trend: {ctx.trend_summary}\n"
        f"- Risk: {ctx.risk_level} (score {ctx.risk_score:.0f}/100)\n"
        f"- Risk flags: {ctx.risk_flags}\n"
    )
    return provider.generate(system, user).strip()


def _insufficient_memo(
    ticker: str,
    research: ResearchOutputSchema,
    trend: TrendOutputSchema,
    sentiment: SentimentOutputSchema,
    risk: RiskOutputSchema,
    question: str,
    mode: str,
) -> InvestmentMemoSchema:
    """Degraded memo: genuine no-view, market-data claims only, no fabrication."""
    ctx = _build_memo_context(ticker, research, trend, sentiment, risk)

    thesis = ""
    if mode in ("groq", "claude", "auto", "openai"):
        try:
            provider = _build_provider(mode)
            thesis = _write_market_data_thesis(provider, ctx, research.status_reason)
        except Exception:
            thesis = ""
    if not thesis:
        thesis = (
            f"No trustworthy evidence was retrieved for {ticker}; this report is "
            f"based on market data only and takes no investment view. "
            f"Trend: {ctx.trend_summary}. "
            f"Risk: {ctx.risk_level} (score {ctx.risk_score:.0f}/100)."
        )

    return InvestmentMemoSchema(
        ticker=ticker,
        as_of=datetime.now(timezone.utc),
        question=question,
        thesis=thesis,
        catalysts=[],
        risks=[f.message for f in risk.flags] or ["No material risk flags triggered."],
        action=ActionSignalSchema(
            signal="no_view",
            confidence=0.0,
            rationale=f"No trustworthy evidence retrieved for {ticker}; declining to take a view.",
        ),
        citations=[],
        risk_level=risk.risk_level,
        risk_score=risk.risk_score,
        writer_mode=mode,
        evidence_status="insufficient",
    )
```

In `_write_thesis`, add a parameter and prompt line for the partial case —
change the signature to `def _write_thesis(provider, ctx, evidence_note: str = "") -> str:`
and insert into the `user` string, right before the final `"Write ONLY the thesis paragraph..."` sentence:

```python
        f"{evidence_note}"
```

In `run_analyst_memo`, immediately after `ctx = _build_memo_context(...)`:

```python
    status = research.evidence_status
    if status == "insufficient":
        return _insufficient_memo(ticker, research, trend, sentiment, risk, question, mode)

    evidence_note = (
        f"NOTE: the evidence base is limited ({research.status_reason}) — "
        f"explicitly acknowledge the limited evidence in the thesis.\n\n"
        if status == "partial" else ""
    )
```

Change the LLM-path thesis call to `_write_thesis(provider, ctx, evidence_note)`, and after
`signal, confidence, rationale = _write_recommendation(provider, ctx)` add:

```python
            if status == "partial":
                confidence = min(confidence, _PARTIAL_CONFIDENCE_CAP)
```

Add `evidence_status=status,` to BOTH `InvestmentMemoSchema(...)` constructions (LLM path
and deterministic path). In the deterministic path, after
`signal, confidence, rationale = _deterministic_recommendation(ctx)` add:

```python
    if status == "partial":
        confidence = min(confidence, _PARTIAL_CONFIDENCE_CAP)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_analyst_degraded.py tests/test_analyst_agent.py -v`
Expected: new 7 PASS + all existing analyst tests PASS.

- [ ] **Step 5: Full suite, then commit**

```bash
git add agents/analyst_agent.py tests/test_analyst_degraded.py
git commit -m "feat: analyst no_view on insufficient evidence, confidence cap on partial"
```

---

### Task 10: Coordinator propagation + debate skip (both pipeline variants)

**Files:**
- Modify: `agents/coordinator_agent.py` (`_run_pipeline` AND `stream_pipeline_events`)
- Modify: `tests/test_coordinator_async.py`, `tests/test_streaming.py` (injected-fake arity only)
- Test: `tests/test_coordinator_propagation.py`

**Interfaces:**
- Injected-callable signatures change (update every existing fake):
  - sentiment: `sf(ticker, question, window_days, top_k, research)` → default wrapper calls `run_sentiment(..., research=research)`
  - risk: `rkf(ticker, mode, price_filepath, question, window_days, trend, sentiment)` → default wrapper calls `run_risk(..., trend=trend, sentiment=sentiment)`
- Debate skip on `research.evidence_status == "insufficient"`: debate never invoked, `memo.debate_skipped_reason = "insufficient evidence"`, trace line `[debate] SKIPPED: insufficient evidence`, stream event `{"event": "skipped", "agent": "debate", "message": "insufficient evidence"}` (in place of its `running` event).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_coordinator_propagation.py
"""End-to-end status propagation through _run_pipeline and stream_pipeline_events."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from agents.coordinator_agent import _run_pipeline, stream_pipeline_events
from core.schemas import (
    ActionSignalSchema,
    InvestmentMemoSchema,
    MemoryComparisonSchema,
    ResearchOutputSchema,
    RiskOutputSchema,
    SentimentOutputSchema,
    TrendOutputSchema,
    TrendSignalSchema,
)

NOW = datetime.now(timezone.utc)


def _research(status):
    return ResearchOutputSchema(ticker="TSLA", question="q", evidence=[],
                                summary="s", evidence_status=status,
                                status_reason="reason")

def _trend():
    return TrendOutputSchema(ticker="TSLA", mode="live", as_of=NOW,
                             signals=[TrendSignalSchema(horizon="30d", return_pct=0.0,
                                                        volatility_pct=1.0,
                                                        max_drawdown_pct=0.0,
                                                        trend_label="neutral")],
                             summary="s")

def _sentiment():
    return SentimentOutputSchema(ticker="TSLA", as_of=NOW, window_days=30,
                                 overall_score=0.0, overall_label="neutral",
                                 items=[], summary="s", data_status="no_data")

def _risk():
    return RiskOutputSchema(ticker="TSLA", as_of=NOW, risk_score=50.0,
                            risk_level="moderate", flags=[], summary="s")

def _memo(status="insufficient"):
    return InvestmentMemoSchema(ticker="TSLA", as_of=NOW, question="q", thesis="t",
                                catalysts=[], risks=[],
                                action=ActionSignalSchema(signal="no_view",
                                                          confidence=0.0, rationale="r"),
                                citations=[], risk_level="moderate", risk_score=50.0,
                                writer_mode="off", evidence_status=status)

def _memory():
    return MemoryComparisonSchema(ticker="TSLA", current_as_of=NOW,
                                  signal_changed=False, thesis_changed=False,
                                  summary="first analysis")


def _fns(status, debate_mock, sentiment_spy=None, risk_spy=None):
    async def rf(t, q, db, tk):
        return _research(status)
    async def tf(t, m, fp):
        return _trend()
    async def sf(t, q, wd, tk, research):
        if sentiment_spy is not None:
            sentiment_spy.append(research)
        return _sentiment()
    async def rkf(t, m, fp, q, wd, trend, sentiment):
        if risk_spy is not None:
            risk_spy.append((trend, sentiment))
        return _risk()
    async def af(t, res, tr, se, ri, q):
        return _memo(status)
    async def df(t, res, tr, se, ri):
        debate_mock(t)
        raise AssertionError("debate must not run on insufficient evidence")
    async def cf(memo):
        return _memory()
    async def svf(memo):
        pass
    return dict(_research_fn=rf, _trend_fn=tf, _sentiment_fn=sf, _risk_fn=rkf,
                _analyst_fn=af, _debate_fn=df, _compare_fn=cf, _save_fn=svf)


def test_research_output_reaches_sentiment_and_risk():
    sentiment_spy, risk_spy = [], []
    fns = _fns("insufficient", MagicMock(), sentiment_spy, risk_spy)
    result = asyncio.run(_run_pipeline("TSLA", "q", "live", 30, None, False, **fns))
    assert sentiment_spy[0].evidence_status == "insufficient"
    trend_arg, sentiment_arg = risk_spy[0]
    assert trend_arg.ticker == "TSLA"
    assert sentiment_arg.data_status == "no_data"

def test_debate_skipped_on_insufficient_with_visible_reason():
    debate_mock = MagicMock()
    fns = _fns("insufficient", debate_mock)
    result = asyncio.run(_run_pipeline("TSLA", "q", "live", 30, None, True, **fns))
    debate_mock.assert_not_called()
    assert result.memo.debate_skipped_reason == "insufficient evidence"
    assert result.debate is None
    assert any("SKIPPED" in line for line in result.pipeline_trace if "[debate]" in line)

def test_debate_runs_normally_on_sufficient():
    ran = []
    async def df(t, res, tr, se, ri):
        ran.append(t)
        from core.schemas import DebateArgumentSchema, DebateOutputSchema
        arg = DebateArgumentSchema(side="bull", arguments=["a"], confidence=0.5,
                                   key_evidence=[])
        bear = DebateArgumentSchema(side="bear", arguments=["b"], confidence=0.5,
                                    key_evidence=[])
        return DebateOutputSchema(ticker=t, as_of=NOW, bull=arg, bear=bear,
                                  coordinator_verdict="v", final_bias="neutral",
                                  memo_update="u")
    fns = _fns("sufficient", MagicMock())
    fns["_debate_fn"] = df
    result = asyncio.run(_run_pipeline("TSLA", "q", "live", 30, None, True, **fns))
    assert ran == ["TSLA"]
    assert result.memo.debate_skipped_reason is None

def test_stream_emits_skipped_event_for_debate():
    fns = _fns("insufficient", MagicMock())
    async def collect():
        events = []
        async for e in stream_pipeline_events("TSLA", "q", "live", 30, None, True, **fns):
            events.append(e)
        return events
    events = asyncio.run(collect())
    skipped = [e for e in events if e.get("event") == "skipped"]
    assert skipped == [{"event": "skipped", "agent": "debate",
                        "message": "insufficient evidence"}]
    running_debate = [e for e in events if e.get("event") == "running"
                      and e.get("agent") == "debate"]
    assert running_debate == []
    complete = [e for e in events if e["event"] == "complete"][0]
    assert complete["data"]["research"]["evidence_status"] == "insufficient"
    assert complete["data"]["memo"]["debate_skipped_reason"] == "insufficient evidence"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_coordinator_propagation.py -v`
Expected: FAIL — `TypeError` (fake arity) once wired; initially fails because coordinator passes 4 args to `sf`.

- [ ] **Step 3: Modify `agents/coordinator_agent.py`**

In BOTH `_run_pipeline` and `stream_pipeline_events`, apply the same four changes:

1. Default wrappers gain the pass-through arguments:

```python
    async def _default_sentiment(t, q, wd, tk, research):
        return await asyncio.to_thread(
            run_sentiment, t, question=q, window_days=wd, top_k=tk,
            store=store, research=research,
        )

    async def _default_risk(t, m, fp, q, wd, trend, sentiment):
        return await asyncio.to_thread(
            run_risk, t, mode=m, price_filepath=fp, question=q, window_days=wd,
            store=store, trend=trend, sentiment=sentiment,
        )
```

2. Call sites pass the computed outputs:

```python
    sentiment, s_t, s_err = await _guarded(
        sf(ticker, question, days_back, 5, research), _fallback_sentiment
    )
```

```python
    risk, rk_t, rk_err = await _guarded(
        rkf(ticker, mode, price_filepath, question, days_back, trend, sentiment),
        _fallback_risk,
    )
```

3. Debate skip — in `_run_pipeline`, replace the Stage 4 debate branching:

```python
    analyst_coro = af(ticker, research, trend, sentiment, risk, question)
    skip_debate = run_debate_flag and research.evidence_status == "insufficient"

    if run_debate_flag and not skip_debate:
        debate_coro = df(ticker, research, trend, sentiment, risk)
        (memo, an_t, an_err), (debate, db_t, db_err) = await asyncio.gather(
            _guarded(analyst_coro, None),
            _guarded(debate_coro, None),
        )
    else:
        memo, an_t, an_err = await _guarded(analyst_coro, None)
        debate, db_t, db_err = None, 0.0, None

    if an_err or memo is None:
        raise RuntimeError(f"Analyst agent failed critically: {an_err}")

    if skip_debate:
        memo.debate_skipped_reason = "insufficient evidence"
        pipeline_trace.append(_trace("debate", "SKIPPED: insufficient evidence", 0.0))
```

(The existing `if run_debate_flag:` reporting block below becomes
`if run_debate_flag and not skip_debate:`.)

4. In `stream_pipeline_events`, same skip logic, plus events: replace the
unconditional `yield {"event": "running", "agent": "debate"}` with

```python
    skip_debate = run_debate_flag and research.evidence_status == "insufficient"
    if run_debate_flag and not skip_debate:
        yield {"event": "running", "agent": "debate"}
    elif skip_debate:
        yield {"event": "skipped", "agent": "debate", "message": "insufficient evidence"}
```

and gate the debate coroutine/reporting on `run_debate_flag and not skip_debate`; after
the analyst memo exists, set `memo.debate_skipped_reason` + trace exactly as in
`_run_pipeline`. The final `FullAnalysisSchema` construction is unchanged
(`debate` is already `None` when skipped).

- [ ] **Step 4: Update existing fakes' arity**

In `tests/test_coordinator_async.py` and `tests/test_streaming.py`, every injected
sentiment fake gains a trailing `research` parameter and every risk fake gains
trailing `trend, sentiment` parameters (values may be ignored). Mechanical change —
signatures only, no behavior edits.

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_coordinator_propagation.py tests/test_coordinator_async.py tests/test_streaming.py -v`
Expected: all PASS.

- [ ] **Step 6: Full suite, then commit**

```bash
git add agents/coordinator_agent.py tests/test_coordinator_propagation.py tests/test_coordinator_async.py tests/test_streaming.py
git commit -m "feat: coordinator propagates evidence status, skips debate visibly on insufficient"
```

---

### Task 11: Ingestion fix (`scripts/fetch_news.py` + metadata plumbing)

**Files:**
- Modify: `scripts/fetch_news.py`, `core/vector_store_manager.py` (`add_documents` metadata), `core/document_loader.py` (`chunk_documents` propagates `about_score`)
- Test: `tests/test_fetch_news_aboutness.py`

**Interfaces:**
- Consumes: `get_company` (Task 1), `aboutness_score` (Task 2), `DocumentSchema.about_score` (Task 4).
- Produces: `fetch_articles(ticker, api_key, page_size=20, company: Optional[CompanyInfo] = None)` queries `q='"<name>" OR "<ticker>"'`; `articles_to_documents(articles, ticker, company: Optional[CompanyInfo] = None)` computes article-level aboutness (title counted twice), skips articles below `ABOUTNESS_THRESHOLD`, stamps `about_score` on every chunk; `VectorStoreManager.add_documents` writes `about_score` metadata **only when not None** (legacy docs keep no key); `chunk_documents` copies `about_score` to chunks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fetch_news_aboutness.py
"""Ingestion: company-name query, article-level aboutness floor, metadata plumbing."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

from core.company_registry import CompanyInfo
from core.document_loader import DocumentLoader
from core.schemas import DocumentSchema
from scripts.fetch_news import articles_to_documents, fetch_articles

TSLA = CompanyInfo(ticker="TSLA", name="Tesla", aliases=["Tesla"], source="config")


def _article(title, description="", content="", url="https://x.com/a"):
    return {"title": title, "description": description, "content": content,
            "publishedAt": "2026-06-30T12:00:00Z", "url": url}


# ── query construction ────────────────────────────────────────────────────────

def test_query_uses_company_name_or_ticker():
    with patch("scripts.fetch_news.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "ok", "articles": []}
        fetch_articles("TSLA", "key", company=TSLA)
    params = mock_get.call_args.kwargs["params"]
    assert params["q"] == '"Tesla" OR "TSLA"'


# ── article-level floor ───────────────────────────────────────────────────────

def test_passing_mention_article_is_skipped():
    arts = [_article("Bybit lists new contracts",
                     content="Contracts for TSLA, AMZN, META and other symbols.")]
    docs = articles_to_documents(arts, "TSLA", company=TSLA)
    assert docs == []

def test_on_topic_article_is_kept_with_about_score():
    arts = [_article("Tesla beats delivery estimates",
                     description="Tesla shares rose.",
                     content="Tesla reported record quarterly deliveries.")]
    docs = articles_to_documents(arts, "TSLA", company=TSLA)
    assert len(docs) >= 1
    assert all(d.about_score is not None and d.about_score >= 0.3 for d in docs)

def test_title_mention_weighted_double():
    # Name only in the title: title counted twice → 2 name hits → score 1.0
    arts = [_article("Tesla expands Berlin plant", content="The factory will grow.")]
    docs = articles_to_documents(arts, "TSLA", company=TSLA)
    assert docs and docs[0].about_score == 1.0


# ── chunk propagation ─────────────────────────────────────────────────────────

def test_chunk_documents_propagates_about_score():
    doc = DocumentSchema(content="x" * 2000, source="newsapi", ticker="TSLA",
                         filepath="f", about_score=0.8)
    chunks = DocumentLoader().chunk_documents([doc])
    assert len(chunks) > 1
    assert all(c.about_score == 0.8 for c in chunks)


# ── vector store metadata ─────────────────────────────────────────────────────

def _vsm():
    # NOTE: patch the consuming module, not sentence_transformers itself —
    # VectorStoreManager binds the name at import time (from X import Y).
    from core.vector_store_manager import VectorStoreManager
    with patch("chromadb.PersistentClient") as mock_client, \
         patch("core.vector_store_manager.SentenceTransformer") as mock_st:
        mock_client.return_value.get_or_create_collection.return_value = MagicMock()
        mock_st.return_value.encode.return_value.tolist.return_value = [0.0]
        store = VectorStoreManager(persist_dir="/tmp/t", collection_name="t")
    return store

def test_add_documents_writes_about_score_when_present():
    store = _vsm()
    doc = DocumentSchema(content="c", source="s", ticker="TSLA", filepath="f",
                         about_score=0.75)
    store.add_documents([doc])
    metadata = store.collection.upsert.call_args.kwargs["metadatas"][0]
    assert metadata["about_score"] == 0.75

def test_add_documents_omits_about_score_when_none():
    store = _vsm()
    doc = DocumentSchema(content="c", source="s", ticker="TSLA", filepath="f")
    store.add_documents([doc])
    metadata = store.collection.upsert.call_args.kwargs["metadatas"][0]
    assert "about_score" not in metadata
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_fetch_news_aboutness.py -v`
Expected: FAIL — `TypeError: fetch_articles() got an unexpected keyword argument 'company'`

- [ ] **Step 3: Implement the three changes**

`scripts/fetch_news.py` — add imports and modify:

```python
from config import CHROMA_DIR, CHROMA_COLLECTION, ABOUTNESS_THRESHOLD
from core.company_registry import CompanyInfo, get_company
from core.retrieval import aboutness_score
```

```python
def fetch_articles(
    ticker: str,
    api_key: Optional[str],
    page_size: int = 20,
    company: Optional[CompanyInfo] = None,
) -> list[dict]:
    ...
    if company is None:
        company = get_company(ticker)
    params = {
        "q": f'"{company.name}" OR "{ticker}"',
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }
    ...  # rest unchanged
```

```python
def articles_to_documents(
    articles: list[dict],
    ticker: str,
    company: Optional[CompanyInfo] = None,
) -> list[DocumentSchema]:
    """Convert NewsAPI articles into chunked DocumentSchema objects.

    Articles scoring below ABOUTNESS_THRESHOLD at the article level (title
    weighted double) are skipped entirely — passing mentions never enter
    the corpus.  Every kept chunk carries the article-level about_score.
    """
    if company is None:
        company = get_company(ticker)

    docs: list[DocumentSchema] = []
    for article in articles:
        title = article.get("title", "") or ""
        raw_content = " ".join(filter(None, [
            title,
            article.get("description", ""),
            article.get("content", ""),
        ]))

        # Title counted twice: a headline mention is a strong aboutness signal.
        about = aboutness_score(f"{title} {raw_content}", company)
        if about < ABOUTNESS_THRESHOLD:
            continue

        ...  # date/url/filepath parsing unchanged

        for chunk in chunk_text(raw_content):
            docs.append(
                DocumentSchema(
                    content=chunk,
                    source="newsapi",
                    ticker=ticker,
                    date=date,
                    filepath=filepath,
                    about_score=round(about, 4),
                )
            )
    return docs
```

`ingest_news` passes one resolved company to both calls:

```python
    company = get_company(ticker)
    articles = fetch_articles(ticker, api_key, page_size=page_size, company=company)
    ...
    docs = articles_to_documents(articles, ticker, company=company)
```

`core/document_loader.py` — in `chunk_documents`, add to the chunk constructor:

```python
                        about_score=doc.about_score,
```

`core/vector_store_manager.py` — in `add_documents`, after building `metadata`:

```python
            # Chroma rejects None values; only write the key when present so
            # legacy documents remain distinguishable from score-0 documents.
            if doc.about_score is not None:
                metadata["about_score"] = float(doc.about_score)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_fetch_news_aboutness.py tests/test_fetch_news.py tests/test_document_chunking.py -v`
Expected: new 7 PASS + all existing fetch_news/chunking tests PASS.

- [ ] **Step 5: Full suite, then commit**

```bash
git add scripts/fetch_news.py core/document_loader.py core/vector_store_manager.py tests/test_fetch_news_aboutness.py
git commit -m "feat: ingestion aboutness floor, company-name NewsAPI query, about_score metadata"
```

---

### Task 12: Frontend degraded rendering + legacy fallback

**Files:**
- Modify: `frontend/index.html`

No JS test harness exists (repo practice); verification is the manual checklist in Step 4 plus the Task 14 user gate. Keep every new field access behind a fallback so legacy payloads render exactly as today.

- [ ] **Step 1: Add CSS (after the `/* MEMORY */` block)**

```css
/* EVIDENCE STATUS / DEGRADED MODE */
.gap-card{border:1px solid rgba(245,158,11,0.55);background:rgba(245,158,11,0.07);}
.gap-card.gap-insufficient{border-color:rgba(239,68,68,0.65);background:rgba(239,68,68,0.07);}
.gap-title{font-size:1.15rem;font-weight:700;margin-bottom:8px;display:flex;align-items:center;gap:10px;}
.gap-insufficient .gap-title{color:var(--red);}
.gap-card:not(.gap-insufficient) .gap-title{color:var(--amber);}
.gap-body{font-size:0.9rem;line-height:1.6;color:var(--text);}
.gap-reason{margin-top:10px;font-size:0.78rem;color:var(--text2);font-family:'Courier New',monospace;}
.action-no_view{background:rgba(148,163,184,0.15);color:var(--text2);border:1px dashed var(--text2);}
[data-evidence-status="insufficient"] .muted-degraded{opacity:0.55;}
[data-evidence-status="insufficient"] .card-title .mkt-tag{display:inline-block;}
.mkt-tag{display:none;margin-left:8px;padding:2px 8px;border-radius:5px;font-size:0.65rem;font-weight:600;background:rgba(148,163,184,0.15);color:var(--text2);vertical-align:middle;}
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;height:180px;color:var(--text2);font-size:0.9rem;gap:6px;text-align:center;}
.empty-state .big{font-size:1.6rem;}
```

- [ ] **Step 2: Add the gap card + tags to the HTML**

Insert directly after `<button class="back-btn" ...>` inside `#resultsPanel`:

```html
  <!-- EVIDENCE GAP NOTICE (hidden unless status is partial/insufficient) -->
  <div class="card gap-card" id="gapCard" style="display:none;">
    <div class="gap-title" id="gapTitle"></div>
    <div class="gap-body" id="gapBody"></div>
    <div class="gap-reason" id="gapReason"></div>
  </div>
```

Add `muted-degraded` to the trend and risk card classes and a tag inside their titles:

```html
    <div class="card muted-degraded">
      <div class="card-title">&#128200; Trend Signals<span class="mkt-tag">market data only</span></div>
```
```html
    <div class="card muted-degraded">
      <div class="card-title">&#9888;&#65039; Risk Assessment<span class="mkt-tag">market data only</span></div>
```

- [ ] **Step 3: Add/modify JS**

New function (near the render helpers) — single source of truth for status:

```javascript
// ── Evidence status (degraded rendering) ───────────────
// Legacy payloads have no evidence_status → status null → render as before.

function setEvidenceStatus(status, reason) {
  const panel = document.getElementById('resultsPanel');
  const gap = document.getElementById('gapCard');
  if (!status || status === 'sufficient') {
    panel.removeAttribute('data-evidence-status');
    gap.style.display = 'none';
    return;
  }
  panel.setAttribute('data-evidence-status', status);
  gap.style.display = 'block';
  gap.classList.toggle('gap-insufficient', status === 'insufficient');
  if (status === 'insufficient') {
    document.getElementById('gapTitle').innerHTML = '&#9888;&#65039; No trustworthy evidence found';
    document.getElementById('gapBody').textContent =
      'Retrieval rejected all candidate documents for this ticker — none were verifiably ' +
      'about the company and relevant to the question. No investment view is taken. ' +
      'The sections below reflect market data (price/volatility) only.';
  } else {
    document.getElementById('gapTitle').innerHTML = '&#9888;&#65039; Limited evidence';
    document.getElementById('gapBody').textContent =
      'Only a small amount of trustworthy evidence passed retrieval gates. ' +
      'Confidence is capped and conclusions should be read with caution.';
  }
  document.getElementById('gapReason').textContent = reason || '';
}
```

Wire it into the stream handler (inside `if (type === 'done')`, after `_streamData[agent] = data;`):

```javascript
    if (agent === 'research' && data) {
      setEvidenceStatus(data.evidence_status || null, data.status_reason || '');
    }
```

and into `renderResults`, first line after destructuring:

```javascript
  setEvidenceStatus(
    (research && research.evidence_status) || (memo && memo.evidence_status) || null,
    (research && research.status_reason) || ''
  );
```

`renderMemoSections` — replace the action/confidence/reco lines:

```javascript
  const sig = (memo.action || {}).signal || 'hold';
  const sigClass = {buy:'action-buy',hold:'action-hold',sell:'action-sell',
                    watch:'action-watch',no_view:'action-no_view'}[sig] || 'action-hold';
  const sigText = sig === 'no_view' ? 'NO VIEW' : sig.toUpperCase();
  document.getElementById('heroAction').innerHTML =
    `<span class="action-badge ${sigClass}">${sigText}</span>`;
  document.getElementById('heroConf').textContent = sig === 'no_view'
    ? 'No view taken — insufficient evidence'
    : `Confidence: ${Math.round(((memo.action || {}).confidence || 0) * 100)}%`;
```

```javascript
  const recoMap = {
    buy:"Conditions appear favorable for a position.",
    hold:"Current position is justified. Wait for clearer signals.",
    sell:"Risk factors outweigh potential upside.",
    watch:"Elevated risk detected. Monitor closely.",
    no_view:"No recommendation — retrieval found no trustworthy evidence for this ticker."
  };
```

and at the end of `renderMemoSections` (debate skip visible in the report):

```javascript
  if (memo.debate_skipped_reason) {
    document.getElementById('debateContent').innerHTML =
      `<div class="empty-state"><span class="big">&#9878;&#65039;</span>` +
      `Debate skipped: ${memo.debate_skipped_reason}</div>`;
  }
```

`renderSentimentGauge` — guard at the top (before any chart work):

```javascript
function renderSentimentGauge(sentiment) {
  if (sentChartInst) { sentChartInst.destroy(); sentChartInst = null; }
  if (sentiment && sentiment.data_status === 'no_data') {
    // Explicit empty state — a neutral-looking dial at 0.00 would be a lie.
    document.getElementById('sentimentChart').style.display = 'none';
    document.getElementById('sentValue').textContent = '';
    document.getElementById('sentLabel').innerHTML =
      '<div class="empty-state"><span class="big">&#128683;</span>No sentiment data<br>' +
      '<span style="font-size:0.78rem;">no trustworthy evidence to score</span></div>';
    return;
  }
  document.getElementById('sentimentChart').style.display = '';
  ...  // existing gauge code unchanged
}
```

`renderEvidenceTable` — score columns + honest empty state:

```javascript
function renderEvidenceTable(evidence, sentiment, statusReason) {
  const sentItems = (sentiment && sentiment.items) ? sentiment.items : [];
  const sentMap = {};
  sentItems.forEach(s => { sentMap[s.citation_id] = s.label; });
  if (!evidence.length) {
    const reason = statusReason
      ? `<div class="gap-reason" style="margin-top:8px;">${statusReason}</div>` : '';
    document.getElementById('evidenceArea').innerHTML =
      '<div style="color:var(--text2);font-size:0.85rem;">No trustworthy evidence passed retrieval gates</div>' + reason;
    return;
  }
  const hasScores = evidence.some(e => e.relevance_score !== null && e.relevance_score !== undefined);
  const scoreHead = hasScores ? '<th>Relevance</th><th>Aboutness</th>' : '';
  const rows = evidence.map(e => {
    const s = sentMap[e.citation_id] || 'neutral';
    const bc = {positive:'badge-positive',neutral:'badge-neutral',negative:'badge-negative'}[s];
    const src = (e.source || e.filepath || 'Unknown').substring(0, 20);
    const snippet = (e.snippet || '').substring(0, 100);
    const dateDisplay = (!e.date || e.date === 'Unknown')
      ? '<span style="color:#94a3b8;font-style:italic;">No date</span>' : formatDate(e.date);
    const scoreCells = hasScores
      ? `<td>${e.relevance_score != null ? e.relevance_score.toFixed(2) : '—'}</td>` +
        `<td>${e.aboutness_score != null ? e.aboutness_score.toFixed(2) : '—'}</td>`
      : '';
    return `<tr><td class="ev-id">${e.citation_id}</td><td>${dateDisplay}</td><td>${src}</td>${scoreCells}<td><span class="badge ${bc}">${s}</span></td><td class="ev-snippet">${snippet}</td></tr>`;
  }).join('');
  document.getElementById('evidenceArea').innerHTML =
    `<table class="ev-table"><thead><tr><th>ID</th><th>Date</th><th>Source</th>${scoreHead}<th>Sentiment</th><th>Snippet</th></tr></thead><tbody>${rows}</tbody></table>`;
}
```

Update its two call sites to pass the reason:
in `_handleStreamEvent`: `renderEvidenceTable(data.evidence || [], {}, data.status_reason || '')` (research branch) and
`renderEvidenceTable(_streamData.research.evidence || [], _streamData.sentiment, (_streamData.research || {}).status_reason || '')`;
in `renderResults`: `renderEvidenceTable((research || {}).evidence || [], sentiment || {}, (research || {}).status_reason || '')`.

`_handleStreamEvent` — handle the new `skipped` event (before the `done` branch):

```javascript
  if (type === 'skipped') {
    setAgentStatus(agent, 'done');   // grey→green dot; the card explains the skip
    return;
  }
```

`resetToInput` — clear degraded state:

```javascript
  setEvidenceStatus(null, '');
```

- [ ] **Step 4: Manual verification checklist (run the API, use fake statuses if needed)**

1. Legacy payload replay: feed a pre-change `FullAnalysisSchema` JSON (no new fields) through `renderResults` via the console — renders identically to today, no console errors.
2. Sufficient run (AAPL): report looks like today + score columns in evidence table.
3. Insufficient run: gap card dominant, NO VIEW badge, no confidence %, muted trend/risk with "market data only" tags, sentiment empty-state (no gauge), debate "skipped" message, evidence card shows rejection accounting.
4. Partial run: amber gap notice, nothing muted, capped confidence shown.

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html
git commit -m "feat: structurally-visible degraded report rendering with legacy fallback"
```

---

### Task 13: Calibration script + USER REVIEW GATE 1 (thresholds)

**Files:**
- Create: `evaluation/calibrate_retrieval.py`
- Modify (after user approval only): `config.py` threshold values + comment

**Interfaces:**
- Consumes: live Chroma corpus, `retrieve_evidence` internals (store.query, aboutness_score, get_reranker), `evaluation/results/annotation_TSLA.json` (the 5 labeled negatives).

- [ ] **Step 1: Write `evaluation/calibrate_retrieval.py`**

```python
"""
Threshold calibration for the trustworthy retrieval layer (spec §7).

Runs the retrieval stages WITHOUT gates over the live corpus and reports
per-candidate (aboutness, cosine similarity, cross-encoder) scores for:
  - labeled negatives: the 5 TSLA off-topic items from the RQ1 eval
    (matched by snippet prefix against annotation_TSLA.json)
  - positive candidates: top candidates for AAPL/NVDA/MSFT/GOOGL, printed
    for manual on-topic verification before the FRR is computed.

Then sweeps a threshold grid and reports, per (aboutness, rerank) pair:
  negatives admitted | positives rejected (false-reject rate).

Output: evaluation/results/calibration_scores.csv + printed sweep table.
Run from project root:  venv\\Scripts\\python.exe evaluation/calibrate_retrieval.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RETRIEVAL_FETCH_N
from core.company_registry import get_company
from core.retrieval import aboutness_score
from core.singletons import get_reranker, get_store

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
QUESTION = "What are the key catalysts and risks?"
RESULTS_DIR = Path("evaluation/results")
OUT_CSV = RESULTS_DIR / "calibration_scores.csv"

ABOUT_GRID = [0.2, 0.25, 0.3, 0.4, 0.5]
RERANK_GRID = [-5.0, -2.0, 0.0, 2.0, 5.0]


def load_tsla_negative_prefixes() -> list[str]:
    """First 80 chars of each labeled off-topic TSLA evidence snippet."""
    with open(RESULTS_DIR / "annotation_TSLA.json", encoding="utf-8") as f:
        record = json.load(f)
    return [e["snippet"][:80] for e in record["evidence"]]


def main() -> None:
    store = get_store()
    reranker = get_reranker()
    if reranker is None:
        print("FATAL: cross-encoder failed to load — calibration needs real scores.")
        sys.exit(1)

    neg_prefixes = load_tsla_negative_prefixes()
    rows = []

    for ticker in TICKERS:
        company = get_company(ticker)
        query_text = f"{company.name} ({ticker}): {QUESTION}"
        candidates = store.query(ticker=ticker, query_text=query_text,
                                 top_k=RETRIEVAL_FETCH_N, days_back=None)
        if not candidates:
            print(f"[{ticker}] no candidates in corpus")
            continue
        scores = reranker.predict([(query_text, c["text"]) for c in candidates])
        for cand, rr in zip(candidates, scores):
            text = cand["text"]
            is_neg = ticker == "TSLA" and any(text.strip().startswith(p[:40]) or p[:40] in text
                                              for p in neg_prefixes)
            rows.append({
                "ticker": ticker,
                "label": "negative" if is_neg else "unlabeled",
                "aboutness": round(aboutness_score(text, company), 4),
                "rerank": round(float(rr), 4),
                "cosine_sim": round(max(0.0, min(1.0, 1.0 - cand["distance"])), 4),
                "date": cand["metadata"].get("date", ""),
                "snippet": text.strip()[:100].replace("\n", " "),
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {OUT_CSV} ({len(rows)} candidates)\n")

    # ── print per-ticker tables for manual positive labeling ─────────────────
    for ticker in TICKERS:
        print(f"── {ticker} " + "─" * 60)
        for r in [r for r in rows if r["ticker"] == ticker]:
            tag = "NEG" if r["label"] == "negative" else "   "
            print(f"  {tag} about={r['aboutness']:.2f} rerank={r['rerank']:+6.2f} "
                  f"cos={r['cosine_sim']:.2f} | {r['snippet'][:70]}")
        print()

    # ── threshold sweep against negatives (positives applied after labeling) ─
    negatives = [r for r in rows if r["label"] == "negative"]
    print("=== SWEEP: negatives admitted (want 0/5) ===")
    print(f"{'about_thr':>9} {'rerank_thr':>10} {'neg_admitted':>12}")
    for a_thr in ABOUT_GRID:
        for r_thr in RERANK_GRID:
            admitted = sum(1 for r in negatives
                           if r["aboutness"] >= a_thr and r["rerank"] >= r_thr)
            print(f"{a_thr:>9} {r_thr:>10} {admitted:>12}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the live corpus**

Run: `venv\Scripts\python.exe evaluation/calibrate_retrieval.py`
Expected: CSV written; per-ticker score tables printed; sweep table printed.
(First run downloads the ~22MB cross-encoder — allow a minute.)

- [ ] **Step 3: Label positives, compute FRR**

Manually verify on-topic AAPL and NVDA candidates from the printed tables
(mark rows whose snippets are genuinely about the company). Compute, for each
candidate threshold pair: `FRR = positives_rejected / positives_total`.

- [ ] **Step 4: USER REVIEW GATE 1 — do not proceed without approval**

Present to the user: score distributions, the sweep table, the FRR on
verified positives for the recommended pair, and — if no pair achieves
0/5 negatives admitted AND FRR 0 — the explicit tradeoff options. **The user
picks; do not pick a balanced number silently.**

- [ ] **Step 5: Lock thresholds (after approval) and commit**

Update `config.py` values and replace the PROVISIONAL comment:

```python
# Calibrated 2026-07-03 via evaluation/calibrate_retrieval.py:
# TSLA negatives admitted 0/5; FRR on verified AAPL/NVDA positives = <measured>.
ABOUTNESS_THRESHOLD = <approved value>
RERANK_THRESHOLD = <approved value>
```

Run: `venv\Scripts\python.exe -m pytest tests/ -q` — all pass (tests inject
thresholds explicitly, so locked values must not break them; if any test
hardcoded the provisional values, fix the test to inject).

```bash
git add config.py evaluation/calibrate_retrieval.py evaluation/results/calibration_scores.csv
git commit -m "feat: retrieval threshold calibration — locked after user review (FRR reported)"
```

---

### Task 14: Acceptance + USER REVIEW GATE 2 (TSLA render) + docs

**Files:**
- Modify: `CLAUDE.md` (Fix 9 checklist entry, Known Issues update)
- Create: `tests/test_retrieval_integration.py` (slow-marked, real cross-encoder)

- [ ] **Step 1: Slow integration test with the real cross-encoder**

```python
# tests/test_retrieval_integration.py
"""Real CrossEncoder smoke test — marked slow, excluded from default dev loop."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.company_registry import CompanyInfo
from core.retrieval import retrieve_evidence

TSLA = CompanyInfo(ticker="TSLA", name="Tesla", aliases=["Tesla"], source="config")


@pytest.mark.slow
def test_real_cross_encoder_separates_on_topic_from_off_topic():
    from sentence_transformers import CrossEncoder
    from config import RERANK_MODEL_NAME
    reranker = CrossEncoder(RERANK_MODEL_NAME)

    query = "Tesla (TSLA): What are the key catalysts and risks?"
    on_topic = "Tesla reported record quarterly deliveries and raised its production guidance."
    off_topic = "Bybit expands perpetual contracts to include seven new TradFi assets."
    scores = reranker.predict([(query, on_topic), (query, off_topic)])
    assert scores[0] > scores[1]
```

Register the marker in `pytest.ini` / `pyproject.toml` if not present
(check first; add `markers = slow: slow tests needing real models` to whichever
config file the repo uses, creating `pytest.ini` only if neither exists).

Run: `venv\Scripts\python.exe -m pytest tests/test_retrieval_integration.py -v -m slow`
Expected: 1 PASS (model download on first run).

- [ ] **Step 2: End-to-end TSLA acceptance run**

Run the API (`venv\Scripts\python.exe -m api.main`) and
trigger a TSLA analysis from the frontend. Verify against the spec acceptance:
- `research.evidence_status` is `insufficient` or `partial`
- zero Meta/Intel/Bybit items in the evidence table
- memo action is `no_view` (if insufficient) with the degraded rendering

- [ ] **Step 3: USER REVIEW GATE 2 — show the user**

Present the rendered TSLA degraded report (screenshot or live) alongside a
sufficient-evidence report (AAPL). **The user confirms the two are impossible
to confuse at a glance before this feature is called done.**

- [ ] **Step 4: Docs + final commit**

In `CLAUDE.md`: add to Fixes Completed
`- [x] Fix 9: Trustworthy retrieval — aboutness gate, cross-encoder re-rank, evidence_status propagation, no_view memos, degraded frontend (spec: docs/superpowers/specs/2026-07-03-trustworthy-retrieval-design.md)`;
under Known Issues, replace the BAC/TSLA-era retrieval notes with a line noting the TSLA
off-topic case is now gated (link the eval).

Run: `venv\Scripts\python.exe -m pytest tests/ -q` — full suite green.

```bash
git add CLAUDE.md tests/test_retrieval_integration.py pytest.ini
git commit -m "feat: Fix 9 complete — trustworthy retrieval acceptance + docs"
```
