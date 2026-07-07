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

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

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
