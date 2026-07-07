"""
RQ1 / RQ2 — Grounding & auditability generation harness.

Produces material for MANUAL annotation (do NOT auto-judge grounding here):
for each ticker it generates two investment theses over the SAME evidence —
  • InvestIQ : the structured multi-agent pipeline (run_full_analysis)
  • Baseline : a single monolithic LLM call given the same evidence pack
Isolating pipeline structure as the only variable between the two.

Outputs per ticker:
  evaluation/results/annotation_{ticker}.json   (machine-readable)
and one shared:
  evaluation/results/annotation_worksheet.txt   (human annotation sheet)

Run from the project root:
    python evaluation/rq1_rq2_generate.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Ensure the project root is importable when launched directly
# (`python evaluation/rq1_rq2_generate.py`), which otherwise puts evaluation/ —
# not the project root — on sys.path and breaks `from agents...` imports.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 console output so non-ASCII characters in theses/evidence survive
# redirection to a file on Windows (default cp1252 raises UnicodeEncodeError).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # not a reconfigurable TextIOWrapper
        pass

from agents.coordinator_agent import run_full_analysis
from llm.providers import OpenAIProvider
from config import OPENAI_MODEL

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
DAYS_BACK = 365
RESULTS_DIR = Path("evaluation/results")
WORKSHEET = RESULTS_DIR / "annotation_worksheet.txt"

E_ID_RE = re.compile(r"\bE\d+\b")
# Sentence splitter: break after . ! ? when followed by whitespace and a new
# sentence start. Approximate (decimals / "EUR 6.8 mn" stay intact); the human
# annotator can re-segment by hand if needed.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'$])')

BAR = "=" * 80
WORKSHEET_HEADER = [
    BAR,
    "RQ1 / RQ2 GROUNDING ANNOTATION WORKSHEET",
    BAR,
    "Per sentence, fill the mark slot(s) at the start of the line:",
    "  Slot 1  [ ] S/U/C           -> S=Supported, U=Unsupported, C=Contradicted",
    "  Slot 2  [ ] cite-faithful?  -> Y/N/NA  (appears only on sentences that carry an",
    "                                 inline E-id; does the cited evidence support the claim?)",
    "Two systems over the SAME evidence pack (shown per ticker block):",
    "  InvestIQ cites STRUCTURALLY (memo.citations field, listed per block);",
    "  Baseline cites INLINE in prose (E-ids embedded in sentences).",
    BAR,
    "",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_eids(text: str) -> list[str]:
    """Distinct E-ids (\\bE\\d+\\b) present in *text*, in first-seen order."""
    seen: list[str] = []
    for eid in E_ID_RE.findall(text or ""):
        if eid not in seen:
            seen.append(eid)
    return seen


def _is_markdown_title(line: str) -> bool:
    """True for a line that is purely a markdown heading / bold title, e.g.
    '# Thesis' or '**Investment Thesis for AAPL**' — not a claim to annotate."""
    s = line.strip()
    if not s:
        return False
    if s.startswith("#"):
        return True
    if s.startswith("**") and s.endswith("**") and len(s) > 4:
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """Sentence segmentation for the annotation worksheet.

    Splits on line breaks first (so a markdown title / bullet the LLM emits
    cannot absorb the following sentence's marker), drops pure markdown-title
    lines, then splits each remaining line on sentence-ending punctuation.
    Approximate by design — decimals / 'EUR 6.8 mn' stay intact.
    """
    text = (text or "").strip()
    if not text:
        return []
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _is_markdown_title(line):
            continue
        out.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
    return out


def evidence_to_dicts(evidence) -> list[dict]:
    """Serialise evidence items to {id, snippet, date}."""
    out = []
    for e in evidence:
        out.append({
            "id": e.citation_id,
            "snippet": e.snippet,
            "date": e.date.isoformat() if e.date else None,
        })
    return out


def build_baseline_thesis(ticker: str, evidence) -> str:
    """Monolithic baseline: one LLM call over the same evidence pack."""
    provider = OpenAIProvider(model=OPENAI_MODEL, max_tokens=1500, temperature=0)
    system = "You are a senior equity analyst. Write a 5-7 sentence investment thesis."
    evidence_block = "\n".join(f"{e.citation_id}: {e.snippet}" for e in evidence)
    user = (
        f"Ticker: {ticker}\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Write a complete investment thesis citing evidence by its E-id where used."
    )
    return provider.generate(system, user)


def citation_coverage(cited_ids: list[str], evidence_ids: list[str]) -> float:
    """Fraction of evidence ids exposed as citations: |cited ∩ evidence| / |evidence|."""
    if not evidence_ids:
        return 0.0
    ev = set(evidence_ids)
    return round(len(ev & set(cited_ids)) / len(ev), 4)


def sentence_lines(thesis: str) -> list[str]:
    """One worksheet line per sentence. Sentences carrying an inline E-id get a
    second 'cite-faithful?' slot so the annotator can check the cited evidence."""
    sentences = split_sentences(thesis)
    if not sentences:
        return ["[ ] S/U/C   (empty thesis)"]
    out: list[str] = []
    for s in sentences:
        if E_ID_RE.search(s):
            out.append(f"[ ] S/U/C   [ ] cite-faithful? Y/N/NA   {s}")
        else:
            out.append(f"[ ] S/U/C   {s}")
    return out


def worksheet_section_from_record(rec: dict) -> str:
    """Build one ticker's worksheet section from a saved annotation record."""
    ticker = rec["ticker"]
    evidence = rec["evidence"]
    iq = rec["investiq"]
    bl = rec["baseline"]
    structured = rec.get("investiq_structured_citations", [])

    section = [BAR, f"TICKER: {ticker}", BAR, ""]

    # InvestIQ — cites structurally (memo.citations), not inline in prose
    section.append("--- InvestIQ Pipeline Thesis ---")
    section.append(
        f"(structured citations [memo.citations]: "
        f"{', '.join(structured) or '(none)'} | coverage {iq['citation_coverage']:.2f})"
    )
    section.append(
        f"(inline-prose citations: "
        f"{', '.join(iq['inline_prose_citations']) or '(none)'} "
        f"| count {iq['inline_prose_citation_count']})"
    )
    section += sentence_lines(iq["thesis"])
    section.append("")

    # Baseline — cites inline in prose only
    section.append("--- Baseline (Monolithic) Thesis ---")
    section.append(
        f"(inline-prose citations: "
        f"{', '.join(bl['inline_prose_citations']) or '(none)'} "
        f"| coverage {bl['citation_coverage']:.2f} | count {bl['inline_prose_citation_count']})"
    )
    section += sentence_lines(bl["thesis"])
    section.append("")

    # Evidence for reference
    section.append("--- Evidence (for reference) ---")
    if evidence:
        for e in evidence:
            section.append(f"{e['id']} | {e['date'] or 'no-date'} | {e['snippet']}")
    else:
        section.append("(no evidence retrieved for this ticker)")
    section.append("")
    section.append("")
    return "\n".join(section)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== RQ1/RQ2 GENERATION (grounding/auditability material) ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    worksheet_sections: list[str] = []
    statuses: list[tuple[str, str]] = []  # (ticker, status_message)

    for ticker in TICKERS:
        print(f"\n[{ticker}] Running InvestIQ pipeline (run_full_analysis)...")
        try:
            result = run_full_analysis(
                ticker, mode="live", days_back=DAYS_BACK, run_debate_flag=True
            )
            evidence = result.research.evidence
            investiq_thesis = result.memo.thesis
            print(f"[{ticker}]   pipeline ok — {len(evidence)} evidence item(s)")

            print(f"[{ticker}] Generating monolithic baseline thesis...")
            baseline_thesis = build_baseline_thesis(ticker, evidence)
            print(f"[{ticker}]   baseline ok")

            evidence_dicts = evidence_to_dicts(evidence)
            evidence_ids = [e["id"] for e in evidence_dicts]
            # InvestIQ exposes the full retrieved evidence set as structured
            # citations (memo.citations == [e.citation_id for e in evidence]).
            investiq_structured = list(result.memo.citations)
            iq_inline = extract_eids(investiq_thesis)
            bl_inline = extract_eids(baseline_thesis)

            record = {
                "ticker": ticker,
                "evidence": evidence_dicts,
                "investiq_structured_citations": investiq_structured,
                "investiq": {
                    "thesis": investiq_thesis,
                    "inline_prose_citations": iq_inline,
                    "inline_prose_citation_count": len(iq_inline),
                    "citation_coverage": citation_coverage(investiq_structured, evidence_ids),
                },
                "baseline": {
                    "thesis": baseline_thesis,
                    "inline_prose_citations": bl_inline,
                    "inline_prose_citation_count": len(bl_inline),
                    "citation_coverage": citation_coverage(bl_inline, evidence_ids),
                },
            }
            out_json = RESULTS_DIR / f"annotation_{ticker}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

            worksheet_sections.append(worksheet_section_from_record(record))

            statuses.append((ticker, f"OK (evidence={len(evidence)}, "
                                     f"investiq_inline={len(iq_inline)}, "
                                     f"baseline_inline={len(bl_inline)})"))
        except Exception as exc:  # noqa: BLE001 — record failure, continue
            print(f"[{ticker}]   FAILED: {type(exc).__name__}: {exc}")
            statuses.append((ticker, f"FAILED: {type(exc).__name__}: {exc}"))

    # ── Write worksheet ───────────────────────────────────────────────────────
    with open(WORKSHEET, "w", encoding="utf-8") as f:
        f.write("\n".join(WORKSHEET_HEADER) + "\n".join(worksheet_sections))

    # ── Summary ───────────────────────────────────────────────────────────────
    ok = [t for t, s in statuses if s.startswith("OK")]
    failed = [t for t, s in statuses if not s.startswith("OK")]

    print("\n=== GENERATION SUMMARY ===")
    print(f"Tickers requested: {len(TICKERS)} ({', '.join(TICKERS)})")
    print(f"Theses generated (both InvestIQ + baseline): {len(ok)} ({', '.join(ok) or 'none'})")
    print(f"Failures: {len(failed)} ({', '.join(failed) or 'none'})")
    for ticker, status in statuses:
        print(f"  {ticker}: {status}")
    print()
    print(f"[saved] per-ticker JSON  → {RESULTS_DIR}/annotation_<TICKER>.json")
    print(f"[saved] worksheet        → {WORKSHEET}")


if __name__ == "__main__":
    main()
