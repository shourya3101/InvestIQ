"""
Process-level singletons for expensive shared resources.

VectorStoreManager (which owns a SentenceTransformer) is constructed once
and reused across all agent calls in a pipeline run.  Call reset_store()
in tests to inject a mock or force a fresh construction.

FinBERT pipeline is similarly lazy-loaded once.  Call reset_finbert_scorer()
in tests to inject a FakeScorer or force re-initialisation.
"""

from __future__ import annotations

from typing import Optional

from core.vector_store_manager import VectorStoreManager

# ── VectorStoreManager singleton ─────────────────────────────────────────────

_store: Optional[VectorStoreManager] = None


def get_store() -> VectorStoreManager:
    """Return the shared VectorStoreManager, constructing it on first call."""
    global _store
    if _store is None:
        _store = VectorStoreManager()
    return _store


def reset_store(store: Optional[VectorStoreManager] = None) -> None:
    """Replace or clear the cached singleton.  Intended for tests only."""
    global _store
    _store = store


# ── FinBERT pipeline singleton ────────────────────────────────────────────────

# Sentinel: distinguishes "not yet attempted" from None ("load failed / VADER")
_FINBERT_NOT_INIT = object()
_finbert_scorer = _FINBERT_NOT_INIT


def _load_finbert():
    """Load and return the FinBERT text-classification pipeline.

    Separated from get_finbert_scorer() so tests can patch this function
    without importing transformers at module load time.
    """
    from transformers import pipeline  # noqa: PLC0415
    return pipeline(
        "text-classification",
        model="yiyanghkust/finbert-tone",
        truncation=True,
        max_length=512,
    )


def get_finbert_scorer():
    """Return the shared FinBERT scorer, loading it on first call.

    Returns None when transformers is not installed or loading fails,
    which signals the agent to fall back to VADER.
    """
    global _finbert_scorer
    if _finbert_scorer is _FINBERT_NOT_INIT:
        try:
            _finbert_scorer = _load_finbert()
        except Exception:
            _finbert_scorer = None  # cache failure → VADER on every call
    return _finbert_scorer


def reset_finbert_scorer(scorer=_FINBERT_NOT_INIT) -> None:
    """Replace or reset the cached FinBERT scorer.  Intended for tests only.

    Call with no arguments to force re-initialisation on next access.
    Call with a FakeScorer (or None) to inject a test double.
    """
    global _finbert_scorer
    _finbert_scorer = scorer
