"""
Memory Agent – persists investment memos and compares to previous runs.

Saves each memo as a compact history entry in a per-ticker .jsonl file,
then diffs the current memo against the most recent saved entry so the
pipeline can surface changes like "risk score jumped 18 points since
last analysis".
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import MEMO_HISTORY_DIR
from core.schemas import (
    InvestmentMemoSchema,
    MemoHistoryEntrySchema,
    MemoryComparisonSchema,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _memo_to_entry(memo: InvestmentMemoSchema) -> MemoHistoryEntrySchema:
    """Convert a full InvestmentMemoSchema into a compact history entry.

    Uses a SHA-1 hash of the thesis to detect thesis changes across runs.
    """
    memo_hash = hashlib.sha1(memo.thesis.encode("utf-8")).hexdigest()

    return MemoHistoryEntrySchema(
        ticker=memo.ticker,
        as_of=memo.as_of,
        risk_score=memo.risk_score,
        risk_level=memo.risk_level,
        action_signal=memo.action.signal,
        confidence=memo.action.confidence,
        thesis_snippet=memo.thesis[:200],
        memo_hash=memo_hash,
    )


# ── Persistence ──────────────────────────────────────────────────────


def save_memo(memo: InvestmentMemoSchema) -> None:
    """Append a memo history entry to the per-ticker .jsonl file.

    Creates MEMO_HISTORY_DIR if it does not exist.  Never raises —
    prints a warning on failure instead.
    """
    try:
        Path(MEMO_HISTORY_DIR).mkdir(parents=True, exist_ok=True)
        filepath = Path(MEMO_HISTORY_DIR) / f"{memo.ticker}.jsonl"
        entry = _memo_to_entry(memo)
        line = json.dumps(entry.model_dump(mode="json"), default=str)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        print(f"[WARNING] memory_agent.save_memo failed for {memo.ticker}: {exc}")


def load_history(ticker: str, n: int = 10) -> list[MemoHistoryEntrySchema]:
    """Load the last *n* memo history entries for *ticker*.

    Returns an empty list if the file does not exist or any error occurs.
    Entries are returned oldest-first, with the most recent at the end.
    """
    try:
        filepath = Path(MEMO_HISTORY_DIR) / f"{ticker}.jsonl"
        if not filepath.exists():
            return []

        entries: list[MemoHistoryEntrySchema] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Ensure as_of is a datetime object
                if isinstance(data.get("as_of"), str):
                    data["as_of"] = datetime.fromisoformat(data["as_of"])
                entries.append(MemoHistoryEntrySchema(**data))

        return entries[-n:]
    except Exception as exc:
        print(f"[WARNING] memory_agent.load_history failed for {ticker}: {exc}")
        return []


def load_last_memo(ticker: str) -> Optional[MemoHistoryEntrySchema]:
    """Return the most recent saved history entry for *ticker*, or None."""
    history = load_history(ticker, n=1)
    return history[0] if history else None


# ── Main entry-point ─────────────────────────────────────────────────


def compare_to_last(current_memo: InvestmentMemoSchema) -> MemoryComparisonSchema:
    """Compare *current_memo* to the last saved entry for the same ticker.

    Returns a MemoryComparisonSchema with a human-readable summary of
    what changed (risk delta, signal change, thesis change).
    """
    ticker = current_memo.ticker
    current_entry = _memo_to_entry(current_memo)
    previous = load_last_memo(ticker)

    # ── No history: first run ────────────────────────────────────────
    if previous is None:
        return MemoryComparisonSchema(
            ticker=ticker,
            current_as_of=current_entry.as_of,
            previous_as_of=None,
            risk_score_delta=None,
            signal_changed=False,
            thesis_changed=False,
            summary=f"First analysis recorded for {ticker}. No previous memo to compare.",
        )

    # ── Compute deltas ───────────────────────────────────────────────
    risk_score_delta = current_entry.risk_score - previous.risk_score
    signal_changed = current_entry.action_signal != previous.action_signal
    thesis_changed = current_entry.memo_hash != previous.memo_hash

    # Days between previous and current
    days_ago: Optional[int] = None
    try:
        prev_as_of = previous.as_of
        if isinstance(prev_as_of, str):
            prev_as_of = datetime.fromisoformat(prev_as_of)
        curr_as_of = current_entry.as_of
        if isinstance(curr_as_of, str):
            curr_as_of = datetime.fromisoformat(curr_as_of)
        delta = curr_as_of - prev_as_of
        days_ago = max(0, delta.days)
    except Exception:
        days_ago = None

    # ── Build summary ────────────────────────────────────────────────
    parts: list[str] = []

    if days_ago is not None:
        parts.append(f"vs analysis {days_ago} day(s) ago")

    if abs(risk_score_delta) >= 1:
        direction = "up" if risk_score_delta > 0 else "down"
        parts.append(
            f"Risk score {direction} {abs(risk_score_delta):.0f} pts "
            f"({previous.risk_score:.0f} -> {current_entry.risk_score:.0f})"
        )
    else:
        parts.append("Risk score unchanged")

    if signal_changed:
        parts.append(
            f"Signal changed: {previous.action_signal.upper()} -> "
            f"{current_entry.action_signal.upper()}"
        )
    else:
        parts.append(f"Signal unchanged ({current_entry.action_signal.upper()})")

    if thesis_changed:
        parts.append("Thesis has changed since last analysis")

    summary = ". ".join(parts) + "."

    return MemoryComparisonSchema(
        ticker=ticker,
        current_as_of=current_entry.as_of,
        previous_as_of=previous.as_of,
        risk_score_delta=risk_score_delta,
        signal_changed=signal_changed,
        thesis_changed=thesis_changed,
        summary=summary,
    )


# ── Smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from core.schemas import (
        InvestmentMemoSchema,
        ActionSignalSchema,
    )
    from datetime import datetime, timezone

    print("Testing Memory Agent...")

    def _make_test_memo(ticker, risk_score, signal, thesis):
        return InvestmentMemoSchema(
            ticker=ticker,
            as_of=datetime.now(timezone.utc),
            question="What are the risks?",
            thesis=thesis,
            catalysts=["Catalyst A"],
            risks=["Risk A"],
            action=ActionSignalSchema(
                signal=signal,
                confidence=0.65,
                rationale="Test rationale",
            ),
            citations=["E1"],
            risk_level="moderate",
            risk_score=risk_score,
            writer_mode="deterministic",
        )

    memo1 = _make_test_memo(
        "AAPL",
        34.0,
        "buy",
        "Apple shows strong momentum with positive sentiment and low risk profile.",
    )
    memo2 = _make_test_memo(
        "AAPL",
        52.0,
        "hold",
        "Apple risk has increased due to macro headwinds and elevated volatility.",
    )

    print("\n--- Saving memo 1 ---")
    save_memo(memo1)
    print("Saved memo 1")

    print("\n--- Comparing memo 2 to last ---")
    comparison = compare_to_last(memo2)
    print(f"Summary: {comparison.summary}")
    print(f"Risk delta: {comparison.risk_score_delta}")
    print(f"Signal changed: {comparison.signal_changed}")
    print(f"Thesis changed: {comparison.thesis_changed}")

    print("\n--- Saving memo 2 ---")
    save_memo(memo2)
    print("Saved memo 2")

    print("\n--- Loading history ---")
    history = load_history("AAPL", n=5)
    print(f"History entries: {len(history)}")
    for entry in history:
        print(
            f"  {entry.as_of} | "
            f"risk={entry.risk_score} | "
            f"signal={entry.action_signal}"
        )

    print("\nMemory Agent test complete.")
