# InvestIQ — Multi-Agent Investment RAG System
**URECA Research Project — Nanyang Technological University**

A time-aware, explainable multi-agent system for investment research and decision support.

## Architecture

7 specialized agents collaborate in sequence:
- **Research Agent** — retrieves evidence via vector embedding search with time-aware filtering
- **Trend Agent** — computes 7d/30d/90d price signals, volatility, and drawdown
- **Sentiment Agent** — VADER-based sentiment scoring over retrieved evidence
- **Risk Agent** — rule-based risk flags and 0-100 risk score
- **Debate Agent** — Bull vs Bear LLM debate with coordinator verdict
- **Analyst Agent** — generates investment memo (deterministic or LLM-powered)
- **Memory Agent** — tracks memos over time, detects signal/risk changes

## Setup

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Configure environment
```
cp .env.example .env
# Add your API keys to .env (optional for LLM features)
```

### 3. Ingest sample data
```
python main.py ingest --file data/sample_docs/sample_document.txt --ticker AAPL
```

## Running

### Option A — Windows
Double-click `start.bat`
Then open `frontend/index.html` in your browser.

### Option B — Command line
```
uvicorn api.routes:app --host 0.0.0.0 --port 8000
```
Then open `frontend/index.html` in your browser.

### Option C — CLI
```
python main.py analyze --ticker AAPL
python main.py analyze --ticker AAPL --days 30
python main.py history --ticker AAPL
```

## API

Interactive docs: http://localhost:8000/docs
Health check:    http://localhost:8000/health

## Adding Bloomberg Data

1. Export news/prices from Bloomberg Terminal as CSV
2. Run: `python main.py ingest --file your_export.csv --ticker AAPL`
3. Run analysis — system automatically uses Bloomberg data

## Project Structure

```
agents/          — 7 specialized AI agents
core/            — schemas, data loaders, vector store
llm/             — Groq + Claude LLM provider abstraction
api/             — FastAPI REST backend
frontend/        — Single-file web application
data/            — Vector DB, memo history, raw files
```
