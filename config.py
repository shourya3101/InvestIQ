"""
Centralised configuration for the URECA research system.

All magic strings and paths live here so every module
imports from a single source of truth.

Environment is controlled by APP_ENV:
  dev  (default) — DEBUG logging, verbose errors
  prod           — INFO logging, minimal output
"""

import logging as _logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Environment ───────────────────────────────────────────────────────────────

APP_ENV: str = os.getenv("APP_ENV", "dev")  # "dev" | "prod" | "production"

_PROD_ENVS = {"prod", "production"}


def get_log_level(env: str) -> int:
    """Return the logging level for *env*. Unknown envs default to DEBUG."""
    return _logging.INFO if env in _PROD_ENVS else _logging.DEBUG


def is_debug(env: str) -> bool:
    """Return True when the environment is NOT production."""
    return env not in _PROD_ENVS


LOG_LEVEL: int = get_log_level(APP_ENV)
DEBUG: bool = is_debug(APP_ENV)

# ── API keys (read from .env) ─────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
MEMO_HISTORY_DIR = DATA_DIR / "memo_history"
LOG_DIR = PROJECT_ROOT / "logs"

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "investment_docs"

# ── Embedding model ──────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Defaults for test / demo runs ────────────────────────────────────────────
DEFAULT_TICKER = "AAPL"
DEFAULT_DAYS_BACK = 365

# ── LLM provider ─────────────────────────────────────────────────────────────
LLM_MODE = "openai"  # "off" | "groq" | "claude" | "auto" | "openai"
GROQ_MODEL = "llama-3.3-70b-versatile"
CLAUDE_MODEL = "claude-3-5-sonnet-latest"
OPENAI_MODEL = "gpt-4o-mini"

# ── Retrieval trust layer ────────────────────────────────────────────────────
RETRIEVAL_FETCH_N = 30
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Calibrated 2026-07-03 via evaluation/calibrate_retrieval.py (user-approved):
# TSLA negatives admitted 0/5 (all score aboutness 0.00 — the aboutness gate
# alone rejects them). Rerank −10.0 splits the observed junk cluster (≤ −10.2)
# from genuine content (≥ −9.25); item-level FRR ≈ 12% (redundant AAPL items
# only), zero change to any ticker's final evidence_status. Do not tune by hand.
ABOUTNESS_THRESHOLD = 0.3
RERANK_THRESHOLD = -10.0

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


def llm_enabled() -> bool:
    """Return True when an LLM provider is configured."""
    return LLM_MODE in ("groq", "claude", "auto", "openai")
