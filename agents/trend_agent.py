from __future__ import annotations

from datetime import datetime
import math

import pandas as pd

from config import DEFAULT_TICKER
from core.schemas import TrendOutputSchema, TrendSignalSchema
from core.market_data_manager import MarketDataManager, get_market_data


# ── Helper functions ────────────────────────────────────────────────


def _ensure_close_series(df: pd.DataFrame) -> pd.Series:
    """Extract the 'Close' column from a price DataFrame.

    Raises ValueError if the column is missing.
    """
    if "Close" not in df.columns:
        raise ValueError(
            "DataFrame has no 'Close' column. "
            f"Available columns: {list(df.columns)}"
        )
    return df["Close"]


def _compute_return_pct(close: pd.Series, days: int) -> float:
    """Compute the percentage return over the last *days* trading days.

    If fewer rows than (days + 1) are available, falls back to the
    earliest available price so partial-window results are still useful.
    """
    if len(close) < 2:
        return 0.0

    end_price = close.iloc[-1]

    # Use the price `days` bars ago if we have enough data, else earliest
    start_idx = -(days + 1)
    if abs(start_idx) > len(close):
        start_idx = 0
    start_price = close.iloc[start_idx]

    if start_price == 0:
        return 0.0

    return ((end_price / start_price) - 1) * 100


def _compute_volatility_pct(close: pd.Series, lookback_days: int) -> float:
    """Annualised volatility (%) over the last *lookback_days* returns.

    Formula: std(daily returns) × √252 × 100
    """
    daily_returns = close.pct_change().dropna()

    if daily_returns.empty:
        return 0.0

    # Restrict to the most recent `lookback_days` returns
    if len(daily_returns) > lookback_days:
        daily_returns = daily_returns.iloc[-lookback_days:]

    std = daily_returns.std()
    if math.isnan(std):
        return 0.0

    return std * math.sqrt(252) * 100


def _compute_max_drawdown_pct(close: pd.Series, lookback_days: int) -> float:
    """Maximum drawdown (%) over the last *lookback_days* trading days.

    Returns a negative number (e.g. -12.5 means a 12.5 % peak-to-trough drop).
    """
    # Use the tail of the close series for the window
    window = close.iloc[-lookback_days:] if len(close) > lookback_days else close

    if window.empty:
        return 0.0

    rolling_peak = window.cummax()
    drawdown = window / rolling_peak - 1  # fraction (≤ 0)
    max_dd = drawdown.min()

    if math.isnan(max_dd):
        return 0.0

    return max_dd * 100


# ── Main entry-point ────────────────────────────────────────────────


HORIZONS = [7, 30, 90]


def _label_trend(return_pct: float) -> str:
    """Simple threshold-based trend label."""
    if return_pct > 3:
        return "bullish"
    if return_pct < -3:
        return "bearish"
    return "neutral"


def run_trend(
    ticker: str,
    mode: str = "live",
    filepath: str | None = None,
) -> TrendOutputSchema:
    """Compute trend + risk metrics for *ticker* and return a typed schema.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    mode : str
        ``"live"`` fetches from Yahoo Finance; ``"offline"`` reads a file.
    filepath : str | None
        Required when *mode* is ``"offline"``.

    Returns
    -------
    TrendOutputSchema
    """
    # ── Fetch price data ────────────────────────────────────────────
    mdm = MarketDataManager()

    if mode == "live":
        df = get_market_data(mdm, mode="live", ticker=ticker)
    elif mode == "offline":
        df = get_market_data(mdm, mode="offline", filepath=filepath)
    else:
        raise ValueError(f"mode must be 'live' or 'offline', got '{mode}'.")

    # Ensure Date is datetime and sorted ascending (MarketDataManager
    # already does this, but guard against raw DataFrames).
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # ── Extract Close series ────────────────────────────────────────
    close = _ensure_close_series(df)

    # ── Build signals for each horizon ──────────────────────────────
    signals: list[TrendSignalSchema] = []

    for days in HORIZONS:
        ret = _compute_return_pct(close, days)
        vol = _compute_volatility_pct(close, days)
        mdd = _compute_max_drawdown_pct(close, days)

        signals.append(
            TrendSignalSchema(
                horizon=f"{days}d",
                return_pct=round(ret, 4),
                volatility_pct=round(vol, 4),
                max_drawdown_pct=round(mdd, 4),
                trend_label=_label_trend(ret),
            )
        )

    # ── Summary (based on 30d signal) ───────────────────────────────
    sig_30 = next((s for s in signals if s.horizon == "30d"), signals[0])
    summary = (
        f"30d trend: {sig_30.trend_label} "
        f"({sig_30.return_pct:.2f}% return), "
        f"vol {sig_30.volatility_pct:.2f}%, "
        f"maxDD {sig_30.max_drawdown_pct:.2f}%."
    )

    return TrendOutputSchema(
        ticker=ticker,
        mode=mode,
        as_of=datetime.now(),
        signals=signals,
        summary=summary,
    )


# ── Smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running trend analysis for {DEFAULT_TICKER} (live) …\n")
    result = run_trend(DEFAULT_TICKER, mode="live")

    print(f"Summary : {result.summary}")
    print(f"As-of   : {result.as_of.isoformat()}")
    print(f"Mode    : {result.mode}\n")

    for sig in result.signals:
        print(
            f"  {sig.horizon:>3s}  |  return {sig.return_pct:+8.2f}%  "
            f"|  vol {sig.volatility_pct:6.2f}%  "
            f"|  maxDD {sig.max_drawdown_pct:+7.2f}%  "
            f"|  {sig.trend_label}"
        )
