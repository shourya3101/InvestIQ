"""
RQ1 (grounding) + RQ2 (auditability) compilation from manual annotation.

Reads the hand-annotated annotation_{ticker}.json files (single annotator,
sentence-by-sentence supported/unsupported + citation faithfulness), aggregates
the counts, and writes evaluation/results/rq1_rq2_summary.txt. This script does
NOT judge grounding — it only arithmetic-aggregates the annotator's counts.

Run from the project root:
    python evaluation/rq1_rq2_compile.py
"""

from __future__ import annotations

import json
from pathlib import Path

# Ensure the project root is importable when launched directly, and force UTF-8
# console output so dashes/figures survive redirection on Windows (cp1252).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
RESULTS_DIR = Path("evaluation/results")
OUT_SUMMARY = RESULTS_DIR / "rq1_rq2_summary.txt"
SYSTEMS = ["investiq", "baseline"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_records() -> dict[str, dict]:
    recs = {}
    for t in TICKERS:
        recs[t] = json.loads((RESULTS_DIR / f"annotation_{t}.json").read_text(encoding="utf-8"))
    return recs


def aggregate(recs: dict[str, dict], system: str) -> dict:
    """Sum the annotation counts and coverage for one system across tickers."""
    keys = ["total_claims", "unsupported_claims", "contradicted_claims",
            "cited_claims", "cite_faithful", "cite_unfaithful"]
    agg = {k: sum(recs[t][system][k] for t in TICKERS) for k in keys}
    n_theses = len(TICKERS)

    total = agg["total_claims"]
    ungrounded = agg["unsupported_claims"] + agg["contradicted_claims"]
    agg["unsupported_rate"] = ungrounded / total if total else 0.0
    agg["mean_unsupported_per_thesis"] = agg["unsupported_claims"] / n_theses
    # Structural/inline citation coverage = mean of per-ticker citation_coverage.
    agg["structural_coverage"] = sum(recs[t][system]["citation_coverage"] for t in TICKERS) / n_theses
    denom = agg["cite_faithful"] + agg["cite_unfaithful"]
    agg["faithfulness"] = (agg["cite_faithful"] / denom) if denom else None
    return agg


def row(label: str, c1: str, c2: str, width: int = 24, colw: int = 12) -> str:
    return f"{label:<{width}}{c1:<{colw}}{c2}"


def fmt_f(x, places: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    recs = load_records()
    iq = aggregate(recs, "investiq")
    bl = aggregate(recs, "baseline")

    total_annotated = iq["total_claims"] + bl["total_claims"]

    # ── Per-ticker breakdown rows ─────────────────────────────────────────────
    breakdown = ["Per-ticker unsupported-claim rate ((unsupported + contradicted) / total):",
                 f"{'Ticker':<10}{'InvestIQ':<22}{'Baseline'}"]
    for t in TICKERS:
        r_iq = recs[t]["investiq"]
        r_bl = recs[t]["baseline"]
        iq_un = r_iq["unsupported_claims"] + r_iq["contradicted_claims"]
        bl_un = r_bl["unsupported_claims"] + r_bl["contradicted_claims"]
        iq_cell = f"{iq_un}/{r_iq['total_claims']} ({iq_un / r_iq['total_claims']:.3f})"
        bl_cell = f"{bl_un}/{r_bl['total_claims']} ({bl_un / r_bl['total_claims']:.3f})"
        breakdown.append(f"{t:<10}{iq_cell:<22}{bl_cell}")

    # ── Note on TSLA (retrieval-corpus limitation) ────────────────────────────
    tsla_note = (
        "For TSLA, none of the 5 retrieved evidence items (E1-E5) are about Tesla — "
        "they cover Meta, Intel, general semiconductor earnings, and Bybit crypto news. "
        "This retrieval-corpus mismatch (not pipeline structure) drove TSLA's high "
        f"unsupported-claim rate for BOTH systems (InvestIQ "
        f"{recs['TSLA']['investiq']['unsupported_claims']}/{recs['TSLA']['investiq']['total_claims']}, "
        f"Baseline {recs['TSLA']['baseline']['unsupported_claims']}/{recs['TSLA']['baseline']['total_claims']}) "
        "and the single unfaithful baseline citation. It should be reported as a "
        "retrieval-corpus limitation, not a pipeline-structure failure."
    )

    # ── Interpretation (grounded only in these numbers) ───────────────────────
    interpretation = (
        f"Raw unsupported-claim rates are similar between the two systems "
        f"(InvestIQ {iq['unsupported_rate']:.3f} vs Baseline {bl['unsupported_rate']:.3f} "
        f"over {total_annotated} annotated claims), so the multi-agent structure does not by "
        f"itself reduce ungrounded claims. Where they diverge is auditability: InvestIQ's "
        f"structural citation field always exposes the full retrieved evidence set "
        f"(coverage {iq['structural_coverage']:.3f}), honestly surfacing the evidence actually "
        f"used even when it is weak — as in TSLA, where coverage stays {recs['TSLA']['investiq']['citation_coverage']:.3f} "
        f"over off-topic evidence — whereas the baseline only exposes the inline citations it "
        f"chose to embed (coverage {bl['structural_coverage']:.3f}). When the baseline does cite it is "
        f"mostly faithful ({bl['cite_faithful']}/{bl['cite_faithful'] + bl['cite_unfaithful']} = "
        f"{bl['faithfulness']:.3f}), but the {bl['cite_unfaithful']} unfaithful cases are silent "
        f"overreach — a real E-id attached to a claim the cited evidence does not support."
    )

    # ── Build summary ─────────────────────────────────────────────────────────
    L = []
    L.append("=== RQ1 GROUNDING RESULTS ===")
    L.append(f"Tickers: {len(TICKERS)} | Total claims annotated: {total_annotated}")
    L.append(row("", "InvestIQ", "Baseline", width=25))
    L.append(row("Unsupported-claim rate:", fmt_f(iq["unsupported_rate"]), fmt_f(bl["unsupported_rate"]), width=25))
    L.append(row("Mean unsupported/thesis:", fmt_f(iq["mean_unsupported_per_thesis"], 1),
                 fmt_f(bl["mean_unsupported_per_thesis"], 1), width=25))
    L.append(row("Contradicted claims:", str(iq["contradicted_claims"]), str(bl["contradicted_claims"]), width=25))
    L.append(f"{'Inter-annotator kappa:':<25}single annotator — not computed")
    L.append("")
    L.extend(breakdown)
    L.append("")
    L.append("=== RQ2 AUDITABILITY RESULTS ===")
    L.append(row("", "InvestIQ", "Baseline", width=26))
    L.append(row("Structural cite coverage:", fmt_f(iq["structural_coverage"]), fmt_f(bl["structural_coverage"]), width=26))
    L.append(row("Citation faithfulness:", fmt_f(iq["faithfulness"]), fmt_f(bl["faithfulness"]), width=26))
    L.append(row("Cited claims (n):", str(iq["cited_claims"]), str(bl["cited_claims"]), width=26))
    L.append(row("Unfaithful citations (n):", str(iq["cite_unfaithful"]), str(bl["cite_unfaithful"]), width=26))
    L.append("")
    L.append("--- Note on TSLA ---")
    L.append(tsla_note)
    L.append("")
    L.append("--- Interpretation ---")
    L.append(interpretation)

    summary = "\n".join(L)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(summary + "\n", encoding="utf-8")

    # Step 4 — explicit annotator note to stdout.
    print("Single annotator — Cohen's kappa not computed (no second annotator).")
    print()
    print(summary)
    print()
    print(f"[saved] {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
