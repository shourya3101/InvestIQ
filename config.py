"""
Centralised configuration for the URECA research system.

All magic strings and paths live here so every module
imports from a single source of truth.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API keys (read from .env) ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
MEMO_HISTORY_DIR = DATA_DIR / "memo_history"

# ── ChromaDB ─────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "investment_docs"

# ── Embedding model ──────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Defaults for test / demo runs ────────────────────────────────────────
DEFAULT_TICKER = "AAPL"
DEFAULT_DAYS_BACK = 365

# ── LLM provider ────────────────────────────────────────────────────────
LLM_MODE = "openai"  # "off" | "groq" | "claude" | "auto" | "openai"
GROQ_MODEL = "llama-3.3-70b-versatile"
CLAUDE_MODEL = "claude-3-5-sonnet-latest"
OPENAI_MODEL = "gpt-4o-mini"


def llm_enabled() -> bool:
    """Return True when an LLM provider is configured."""
    return LLM_MODE in ("groq", "claude", "auto", "openai")
