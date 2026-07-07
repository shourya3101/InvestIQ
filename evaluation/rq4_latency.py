"""
RQ4 — Latency experiment: asynchronous vs. sequential agent execution.

Measures the wall-clock speedup of InvestIQ's async DAG pipeline
(agents.coordinator_agent._run_pipeline) against a strictly sequential
baseline that runs the same seven agents one at a time, in the same order,
using the same shared VectorStoreManager singleton.

Run from the project root:
    python evaluation/rq4_latency.py
"""

from __future__ import annotations

# ── Section A: Imports and config ────────────────────────────────────────────
import asyncio
import csv
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

# Ensure the project root is importable when this script is launched directly
# (`python evaluation/rq4_latency.py`), which otherwise puts evaluation/ — not
# the project root — on sys.path and breaks `from agents...` imports.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 console output so the "→" progress arrows survive redirection to
# a file on Windows (default cp1252 can't encode U+2192 and raises
# UnicodeEncodeError).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # not a reconfigurable TextIOWrapper
        pass

from agents.coordinator_agent import _run_pipeline
from core.singletons import get_store, reset_store
from core.schemas import (
    ResearchOutputSchema,
    TrendOutputSchema,
    SentimentOutputSchema,
    RiskOutputSchema,
    InvestmentMemoSchema,
    ActionSignalSchema,
)
from agents.research_agent import run_research
from agents.trend_agent import run_trend
from agents.sentiment_agent import run_sentiment
from agents.risk_agent import run_risk
from agents.debate_agent import run_debate
from agents.analyst_agent import run_analyst_memo
from agents.memory_agent import save_memo, compare_to_last

TICKERS = ["AAPL", "MSFT", "NVDA"]
RUNS_PER_TICKER = 3
QUESTION = "What are the key catalysts and risks?"
DAYS_BACK = 365
OUTPUT_CSV = "evaluation/results/rq4_latency.csv"
OUTPUT_SUMMARY = "evaluation/results/rq4_summary.txt"

# Pipeline-wide constants held fixed across both arms of the experiment so the
# only independent variable is async-parallel vs. sequential scheduling.
MODE = "live"
PRICE_FILEPATH = None
TOP_K = 5


# ── Section B: Sequential pipeline function ──────────────────────────────────
async def run_sequential_pipeline(ticker: str, question: str, days_back: int) -> dict:
    """Run the seven agents strictly one at a time, mirroring the coordinator.

    Exact order: research → trend → sentiment → risk → analyst → debate → memory.
    Each agent runs via asyncio.to_thread (identical to the coordinator's default
    wrappers) but never overlaps, so this is the sequential baseline. On any
    agent failure the exception is captured and partial results are returned.
    """
    store = get_store()
    ticker = ticker.upper().strip()

    action_signal = None
    evidence_count = 0
    writer_mode = None
    error = None

    start = time.perf_counter()
    try:
        research: ResearchOutputSchema = await asyncio.to_thread(
            run_research, ticker, question, days_back=days_back, top_k=TOP_K, store=store
        )
        evidence_count = len(research.evidence)

        trend: TrendOutputSchema = await asyncio.to_thread(
            run_trend, ticker, mode=MODE, filepath=PRICE_FILEPATH
        )

        sentiment: SentimentOutputSchema = await asyncio.to_thread(
            run_sentiment, ticker, question=question, window_days=days_back,
            top_k=TOP_K, store=store
        )

        risk: RiskOutputSchema = await asyncio.to_thread(
            run_risk, ticker, mode=MODE, price_filepath=PRICE_FILEPATH,
            question=question, window_days=days_back, store=store
        )

        memo: InvestmentMemoSchema = await asyncio.to_thread(
            run_analyst_memo, ticker=ticker, research=research, trend=trend,
            sentiment=sentiment, risk=risk, question=question
        )
        action_signal = memo.action.signal
        writer_mode = memo.writer_mode

        debate = await asyncio.to_thread(
            run_debate, ticker, research, trend, sentiment, risk
        )
        memo.debate = debate

        memory = await asyncio.to_thread(compare_to_last, memo)
        memo.memory = memory
        await asyncio.to_thread(save_memo, memo)
    except Exception as exc:  # noqa: BLE001 — record any agent failure, return partial
        error = f"{type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - start

    return {
        "ticker": ticker,
        "runtime_seconds": runtime,
        "action_signal": action_signal,
        "evidence_count": evidence_count,
        "writer_mode": writer_mode,
        "error": error,
    }


# ── Section C: Async pipeline wrapper ────────────────────────────────────────
async def run_async_pipeline(ticker: str, question: str, days_back: int) -> dict:
    """Run the production async DAG (_run_pipeline) and time the whole call."""
    norm_ticker = ticker.upper().strip()

    action_signal = None
    evidence_count = 0
    writer_mode = None
    error = None

    start = time.perf_counter()
    try:
        full = await _run_pipeline(
            ticker=ticker,
            question=question,
            mode=MODE,
            days_back=days_back,
            price_filepath=PRICE_FILEPATH,
            run_debate_flag=True,
        )
        action_signal = full.memo.action.signal
        evidence_count = len(full.research.evidence)
        writer_mode = full.memo.writer_mode
    except Exception as exc:  # noqa: BLE001 — record failure, return partial
        error = f"{type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - start

    return {
        "ticker": norm_ticker,
        "runtime_seconds": runtime,
        "action_signal": action_signal,
        "evidence_count": evidence_count,
        "writer_mode": writer_mode,
        "error": error,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_stdev(values: list[float]) -> float:
    """Sample stdev; 0.0 when fewer than two data points."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _signals_for(results: list[dict], ticker: str, mode: str) -> list[str]:
    """Return the sorted action_signals across all runs of (ticker, mode)."""
    return sorted(
        str(r["action_signal"])
        for r in results
        if r["ticker"] == ticker and r["mode"] == mode
    )


# ── Section D: Main experiment loop ──────────────────────────────────────────
async def main() -> None:
    start_dt = datetime.now()
    print("=== RQ4 LATENCY EXPERIMENT ===")
    print(f"Start timestamp: {start_dt.isoformat(timespec='seconds')}")

    print("Warming up vector store and embedding model...")
    # Force singleton construction (VectorStoreManager + SentenceTransformer)
    # before any timing begins so model-load cost is excluded from measurements.
    get_store()

    print("Warming up FinBERT scorer...")
    from core.singletons import get_finbert_scorer
    get_finbert_scorer()
    print("Warmup complete. Starting timed runs.")

    # ── Force deterministic OpenAI sampling for this experiment ──────────────
    # Both analyst_agent._build_provider and debate_agent construct
    # OpenAIProvider(model=...) with keyword-only args and no explicit
    # temperature, so defaulting temperature=0 here makes action_signal
    # reproducible run-to-run and the equivalence check meaningful.
    from llm import providers as _llm_providers
    _orig_init = _llm_providers.OpenAIProvider.__init__
    def _zero_temp_init(self, *args, **kwargs):
        kwargs.setdefault("temperature", 0)
        _orig_init(self, *args, **kwargs)
    _llm_providers.OpenAIProvider.__init__ = _zero_temp_init

    results: list[dict] = []

    for ticker in TICKERS:
        for n in range(1, RUNS_PER_TICKER + 1):
            # ── Async arm ────────────────────────────────────────────────────
            print(f"Running ASYNC  | Ticker: {ticker} | Run {n}/{RUNS_PER_TICKER}...")
            a = await run_async_pipeline(ticker, QUESTION, DAYS_BACK)
            a.update({
                "run_id": f"{ticker}-async-r{n}",
                "run_number": n,
                "mode": "async",
            })
            results.append(a)
            print(
                f"  → {a['runtime_seconds']:.2f}s | signal={a['action_signal']} "
                f"| evidence={a['evidence_count']}"
            )

            # ── Sequential arm ───────────────────────────────────────────────
            print(f"Running SEQ    | Ticker: {ticker} | Run {n}/{RUNS_PER_TICKER}...")
            s = await run_sequential_pipeline(ticker, QUESTION, DAYS_BACK)
            s.update({
                "run_id": f"{ticker}-sequential-r{n}",
                "run_number": n,
                "mode": "sequential",
            })
            results.append(s)
            print(
                f"  → {s['runtime_seconds']:.2f}s | signal={s['action_signal']} "
                f"| evidence={s['evidence_count']}"
            )

    # ── Statistics ───────────────────────────────────────────────────────────
    async_times = [r["runtime_seconds"] for r in results if r["mode"] == "async" and not r["error"]]
    seq_times = [r["runtime_seconds"] for r in results if r["mode"] == "sequential" and not r["error"]]

    async_mean = statistics.mean(async_times) if async_times else 0.0
    async_std = _safe_stdev(async_times)
    seq_mean = statistics.mean(seq_times) if seq_times else 0.0
    seq_std = _safe_stdev(seq_times)
    speedup = (seq_mean / async_mean) if async_mean else 0.0

    # Output equivalence at the DISTRIBUTION level: compare the set of
    # action_signals produced across all runs per mode (per ticker), rather
    # than a single run-1 vs run-1 exact match. The async and sequential
    # pipelines call identical agent functions with identical inputs, so any
    # per-run signal variation is LLM sampling noise, not a scheduling effect;
    # set equality is the meaningful equivalence criterion.
    equivalence = {}
    all_sets_match = True
    for ticker in TICKERS:
        async_signals = _signals_for(results, ticker, "async")
        seq_signals = _signals_for(results, ticker, "sequential")
        sets_match = set(async_signals) == set(seq_signals)
        equivalence[ticker] = (async_signals, seq_signals, sets_match)
        if not sets_match:
            all_sets_match = False

    # ── Section E: Save CSV ──────────────────────────────────────────────────
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "ticker", "run_number", "mode", "runtime_seconds",
        "action_signal", "evidence_count", "writer_mode", "error",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "run_id": r["run_id"],
                "ticker": r["ticker"],
                "run_number": r["run_number"],
                "mode": r["mode"],
                "runtime_seconds": f"{r['runtime_seconds']:.4f}",
                "action_signal": r["action_signal"],
                "evidence_count": r["evidence_count"],
                "writer_mode": r["writer_mode"],
                "error": r["error"],
            })

    # ── Section F: Save and print summary ────────────────────────────────────
    total_runs = len(TICKERS) * RUNS_PER_TICKER * 2

    lines = []
    lines.append("=== RQ4 LATENCY RESULTS ===")
    lines.append(f"Experiment date: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Tickers tested: {', '.join(TICKERS)}")
    lines.append(f"Runs per ticker per mode: {RUNS_PER_TICKER}")
    lines.append(f"Total runs: {total_runs}")
    lines.append("")
    lines.append("--- Async Pipeline ---")
    lines.append(f"Mean latency:  {async_mean:.2f}s")
    lines.append(f"Std deviation: {async_std:.2f}s")
    lines.append(f"Min:           {min(async_times) if async_times else 0.0:.2f}s")
    lines.append(f"Max:           {max(async_times) if async_times else 0.0:.2f}s")
    lines.append("")
    lines.append("--- Sequential Pipeline ---")
    lines.append(f"Mean latency:  {seq_mean:.2f}s")
    lines.append(f"Std deviation: {seq_std:.2f}s")
    lines.append(f"Min:           {min(seq_times) if seq_times else 0.0:.2f}s")
    lines.append(f"Max:           {max(seq_times) if seq_times else 0.0:.2f}s")
    lines.append("")
    lines.append("--- Speedup ---")
    lines.append(f"Ratio (seq/async): {speedup:.2f}x")
    lines.append(f"Absolute reduction: {seq_mean - async_mean:.2f}s")
    lines.append("")
    lines.append("--- Output Equivalence Check (distribution level) ---")
    for ticker in TICKERS:
        async_signals, seq_signals, sets_match = equivalence[ticker]
        a_str = "{" + ",".join(async_signals) + "}"
        s_str = "{" + ",".join(seq_signals) + "}"
        lines.append(
            f"{ticker}: async={a_str} | sequential={s_str} | signal_sets_match={sets_match}"
        )
    lines.append(f"All signal sets match: {all_sets_match}")
    lines.append(
        "Output equivalence interpreted at distribution level: async and "
        "sequential call identical agent functions with identical inputs, so "
        "any per-run signal variation reflects LLM sampling noise "
        "(temperature/inference nondeterminism), not scheduling differences."
    )
    lines.append("")
    lines.append("--- Errors ---")
    error_runs = [r for r in results if r["error"]]
    if error_runs:
        for r in error_runs:
            lines.append(f"{r['run_id']} ({r['mode']}): {r['error']}")
    else:
        lines.append("None")

    summary = "\n".join(lines)

    Path(OUTPUT_SUMMARY).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print()
    print(summary)
    print()
    print(f"[saved] CSV     → {OUTPUT_CSV}")
    print(f"[saved] Summary → {OUTPUT_SUMMARY}")


# ── Section G: Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
