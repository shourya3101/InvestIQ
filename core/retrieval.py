"""
Retrieval policy for the trustworthy retrieval layer.

VectorStoreManager stays a thin Chroma wrapper; every gating/ranking/status
decision lives here.  Spec: docs/superpowers/specs/2026-07-03-trustworthy-
retrieval-design.md.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, Field

from config import (
    ABOUTNESS_THRESHOLD,
    MIN_SUFFICIENT_EVIDENCE,
    RERANK_THRESHOLD,
    RETRIEVAL_FETCH_N,
)
from core.company_registry import CompanyInfo, get_company
from core.schemas import EvidenceSchema

_UNSET = object()

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
    # Resolve the singleton lazily: never load the model when nothing survived
    # the aboutness gate (and keep unit tests free of real model loads).
    if _reranker is _UNSET:
        if survivors:
            from core.singletons import get_reranker  # noqa: PLC0415
            reranker = get_reranker()
        else:
            reranker = None
    else:
        reranker = _reranker

    had_rank_pool = bool(survivors)
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
    if had_rank_pool and not reranker_used:
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
