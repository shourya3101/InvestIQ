from __future__ import annotations

import json as _json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import DATA_DIR
from agents.coordinator_agent import run_full_analysis, stream_pipeline_events
from agents.memory_agent import load_history


# ── Rate limiter ──────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── App setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="InvestIQ API",
    description="Multi-Agent Investment RAG System",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Input validation ──────────────────────────────────────────────────

_TICKER_RE = re.compile(r"^[A-Z0-9]{1,5}$")


def validate_ticker(raw: str) -> str:
    """Uppercase and validate *raw* ticker. Raises HTTP 422 on bad input."""
    ticker = raw.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(
            status_code=422,
            detail="Ticker must be 1-5 alphanumeric characters (A-Z, 0-9).",
        )
    return ticker


# ── Request models ────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker: str
    question: str = "What are the key catalysts and risks?"
    mode: str = "live"
    days_back: int = 365
    run_debate: bool = True


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "InvestIQ API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agents": [
            "research", "trend", "sentiment",
            "risk", "debate", "analyst", "memory",
        ],
    }


@app.post("/analyze")
@limiter.limit("10/minute")
def analyze(request: Request, req: AnalyzeRequest):
    ticker = validate_ticker(req.ticker)

    if req.mode not in ("live", "offline"):
        raise HTTPException(
            status_code=422,
            detail="mode must be 'live' or 'offline'",
        )
    if req.mode == "offline":
        raise HTTPException(
            status_code=422,
            detail="Use /analyze/upload for offline mode",
        )

    try:
        result = run_full_analysis(
            ticker=ticker,
            question=req.question,
            mode="live",
            days_back=req.days_back,
            run_debate_flag=req.run_debate,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/stream")
@limiter.limit("10/minute")
async def analyze_stream(request: Request, req: AnalyzeRequest):
    """Stream analysis results as NDJSON, one event per agent completion."""
    ticker = validate_ticker(req.ticker)

    if req.mode not in ("live", "offline"):
        raise HTTPException(status_code=422, detail="mode must be 'live' or 'offline'")
    if req.mode == "offline":
        raise HTTPException(status_code=422, detail="Use /analyze/upload for offline mode")

    async def _generate():
        async for event in stream_pipeline_events(
            ticker=ticker,
            question=req.question,
            mode="live",
            days_back=req.days_back,
            run_debate_flag=req.run_debate,
        ):
            yield _json.dumps(event) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


@app.post("/analyze/upload")
def analyze_upload(
    ticker: str = Form(...),
    question: str = Form(default="What are the key catalysts and risks?"),
    days_back: int = Form(default=365),
    run_debate: bool = Form(default=True),
    price_file: Optional[UploadFile] = File(default=None),
    news_file: Optional[UploadFile] = File(default=None),
):
    ticker = validate_ticker(ticker)

    raw_dir = Path(DATA_DIR) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    price_filepath: Optional[str] = None

    try:
        if price_file is not None:
            price_path = raw_dir / f"{ticker}_{int(time.time())}_prices.csv"
            with open(price_path, "wb") as f:
                shutil.copyfileobj(price_file.file, f)
            price_filepath = str(price_path)
            notes.append(f"Saved price file: {price_path.name}")

        if news_file is not None:
            news_path = raw_dir / f"{ticker}_{int(time.time())}_news.csv"
            with open(news_path, "wb") as f:
                shutil.copyfileobj(news_file.file, f)

            from core.document_loader import DocumentLoader
            from core.vector_store_manager import VectorStoreManager

            loader = DocumentLoader()
            docs = loader.load_csv(str(news_path), ticker=ticker)
            store = VectorStoreManager()
            count = store.add_documents(docs)
            notes.append(f"Ingested {count} news docs")

        result = run_full_analysis(
            ticker=ticker,
            question=question,
            mode="offline" if price_filepath else "live",
            days_back=days_back,
            price_filepath=price_filepath,
            run_debate_flag=run_debate,
        )
        payload = result.model_dump(mode="json")
        if notes:
            payload["_upload_notes"] = notes
        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{ticker}")
def history(ticker: str):
    ticker = ticker.upper().strip()
    entries = load_history(ticker, n=10)
    if not entries:
        return {"ticker": ticker, "entries": [], "message": "No history found"}
    return {
        "ticker": ticker,
        "entries": [e.model_dump(mode="json") for e in entries],
        "count": len(entries),
    }
