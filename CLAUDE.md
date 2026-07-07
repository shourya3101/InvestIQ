# InvestIQ — Agent Context File

## What This Project Is
Multi-agent investment research system (NTU URECA project).
User enters a ticker → 7 agents run → institutional-grade report in ~20s.

## Tech Stack
- Python 3.11, FastAPI, ChromaDB, sentence-transformers
- GPT-4o-mini (primary), Groq/Claude (fallback via LLMRouter)
- VADER sentiment, yfinance market data, Pydantic v2 schemas
- Vanilla HTML/CSS/JS frontend with Chart.js

## Project Structure
agents/          ← 7 specialized agents
core/            ← schemas, vector store, document loader, market data
llm/             ← LLMRouter, OpenAI/Groq/Claude providers
api/             ← FastAPI routes + uvicorn launcher
frontend/        ← single-file dashboard (index.html)
scripts/         ← fetch_news.py (NewsAPI ingestion)
tests/           ← pytest test suite (62 tests, all passing)
data/chroma/     ← ChromaDB persistent vector store
data/memo_history/ ← per-ticker JSONL memory

## Agent Pipeline (async DAG)
gather(Research + Trend) → Sentiment → Risk → gather(Analyst + Debate) → Memory

## Fixes Completed
- [x] Fix 1: Real news ingestion (scripts/fetch_news.py, 38 tests)
- [x] Fix 2: Async parallel execution (asyncio.gather, 17 tests)
- [x] Fix 3: Singleton pattern for VectorStoreManager + SentenceTransformer, 
             parse_llm_json extracted to core/utils.py (62 tests total)

## Fixes Remaining
- [x] Fix 4: Thesis cutoff — split analyst into 3 smaller LLM calls (max_tokens=1500 each, 33 tests)
- [x] Fix 5: Swap VADER for FinBERT sentiment (singleton, VADER fallback, 34 tests)
- [x] Fix 6: Better document chunking (sliding window 800 chars, 100 overlap, 29 tests)
- [x] Fix 7: Progressive frontend loading — NDJSON streaming, per-agent progress bar (195 tests)
- [x] Fix 8: Production readiness — structured logging, yfinance cache, slowapi rate limit, ticker validation, dev/prod config (234 tests)
- [x] Fix 9: Trustworthy retrieval — aboutness gate, cross-encoder re-rank, evidence_status propagation, no_view memos, degraded frontend (316 tests; spec: docs/superpowers/specs/2026-07-03-trustworthy-retrieval-design.md)

## Known Issues
- BAC ticker had duplicate ID error on ingestion
- TSLA off-topic retrieval (RQ1 eval: Meta/Intel/Bybit items) is now gated — Fix 9
  rejects them at query time (see evaluation/results/calibration_scores.csv)

## Key Files
- agents/coordinator_agent.py ← async pipeline orchestrator
- agents/analyst_agent.py ← memo generation, no_view/partial degradation
- core/retrieval.py ← retrieval policy: aboutness gate, re-rank, evidence_status
- core/company_registry.py ← ticker → company name/aliases (config→cache→yfinance)
- core/vector_store_manager.py ← ChromaDB wrapper
- core/singletons.py ← shared VectorStoreManager singleton
- core/utils.py ← shared parse_llm_json utility
- core/schemas.py ← all Pydantic models
- llm/providers.py ← LLMRouter, max_tokens=2000
- config.py ← all constants, dotenv loading

## Rules
- Always use TDD (write failing tests first)
- Never break existing Pydantic schemas
- Run full test suite after every change: python -m pytest tests/ -v
- Activate venv first: venv\Scripts\activate
- After each fix, update the checklist in this file
