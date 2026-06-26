from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yfinance as yf

# ── Per-day yfinance cache ────────────────────────────────────────────────────
# Key: (ticker, date_iso, period, interval) → DataFrame copy
_yfinance_cache: dict[tuple, pd.DataFrame] = {}


def clear_yfinance_cache() -> None:
    """Clear the in-process yfinance cache. Intended for tests."""
    _yfinance_cache.clear()


@dataclass
class MarketDataManager:
    """
    A single interface to get market data from:
      (1) Live source: Yahoo Finance (for home dev/testing)
      (2) Offline files: Bloomberg exports (for lab / final demo)
    """

    default_period: str = "6mo"
    default_interval: str = "1d"

    def _normalize_price_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a price DataFrame to standard schema:
        Date, Open, High, Low, Close, Volume

        Supports common Bloomberg and finance column variants.
        Handles MultiIndex columns (e.g., from yfinance) by flattening them.
        """
        # Handle MultiIndex columns (e.g., from yfinance download)
        if isinstance(df.columns, pd.MultiIndex):
            # For single-ticker downloads yfinance returns tuples like
            # ('Close', 'AAPL').  We only need the first level (the
            # metric name); the ticker part is redundant.
            flattened_cols = []
            for col_tuple in df.columns:
                first = str(col_tuple[0]).strip()
                if first and first.lower() != "nan":
                    flattened_cols.append(first)
                else:
                    # Fallback: join all non-empty parts
                    parts = [str(x).strip() for x in col_tuple
                             if str(x).strip() and str(x).lower() != "nan"]
                    flattened_cols.append("_".join(parts) if parts else "Unnamed")
            df.columns = flattened_cols
        else:
            # Convert columns to regular Index if not already
            df.columns = df.columns.to_list()

        # Strip whitespace from column names
        df.columns = [str(col).strip() for col in df.columns]

        # Mapping of standard column names to their variants
        column_mapping = {
            "Date": ["date", "DATE", "Trade Date", "trade date"],
            "Open": ["open", "OPEN", "PX_OPEN", "px_open"],
            "High": ["high", "HIGH", "PX_HIGH", "px_high"],
            "Low": ["low", "LOW", "PX_LOW", "px_low"],
            "Close": ["close", "CLOSE", "PX_LAST", "px_last", "Last Price", "last price", "Adj Close", "adj close"],
            "Volume": ["volume", "VOLUME", "PX_VOLUME", "px_volume", "VOL", "vol"],
        }

        # Create a reverse mapping: variant -> standard name
        reverse_mapping = {}
        for standard, variants in column_mapping.items():
            for variant in variants:
                reverse_mapping[variant] = standard

        # Rename columns
        renamed_cols = {}
        for col in df.columns:
            if col in reverse_mapping:
                renamed_cols[col] = reverse_mapping[col]
        df = df.rename(columns=renamed_cols)

        # Ensure Date column exists
        if "Date" not in df.columns:
            raise ValueError(
                "Date column not found. "
                "Provide a column named 'Date', 'date', 'Trade Date', etc."
            )

        # Parse Date to datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Sort by Date ascending
        df = df.sort_values("Date").reset_index(drop=True)

        # Keep only standard columns that exist, prioritizing:
        # Date (always) + Close (if available) + others
        standard_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        available_cols = [c for c in standard_cols if c in df.columns]
        df = df[available_cols].copy()

        return df

    def fetch_live_data_yfinance(
        self,
        ticker: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch live OHLCV data from yfinance, caching per ticker per day.

        Returns a DataFrame with standard columns:
        Date, Open, High, Low, Close, Volume
        """
        period = period or self.default_period
        interval = interval or self.default_interval

        cache_key = (ticker, date.today().isoformat(), period, interval)
        if cache_key in _yfinance_cache:
            return _yfinance_cache[cache_key].copy()

        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if df is None or df.empty:
            raise ValueError(f"No yfinance data returned for ticker='{ticker}'")

        df = df.reset_index()
        df = self._normalize_price_df(df)

        _yfinance_cache[cache_key] = df
        return df.copy()

    def load_offline_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Placeholder: load Bloomberg-exported data from file.

        For now:
          - supports CSV and Parquet cleanly
          - PDF ingestion is intentionally NOT implemented yet (needs parsing/OCR strategy)

        Expected output format (same as yfinance):
        Date, Open, High, Low, Close, Volume

        Later you will:
          - map Bloomberg column names -> this standard schema
          - attach metadata (ticker, source, timezone, etc.)
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Offline file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".pdf":
            raise NotImplementedError(
                "PDF loading not implemented yet. "
                "For MVP, export Bloomberg data as CSV. We'll add PDF parsing later if needed."
            )
        else:
            raise ValueError(f"Unsupported file type: '{suffix}'. Use .csv or .parquet for now.")

        # Basic sanity check
        if df.empty:
            raise ValueError("Offline file loaded but is empty.")

        # Apply normalization to standard schema
        df = self._normalize_price_df(df)

        return df


def get_market_data(
    manager: MarketDataManager,
    *,
    mode: str,
    ticker: Optional[str] = None,
    filepath: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Simple switch function so the rest of your system uses ONE call.

    mode:
      - "live"    -> uses yfinance
      - "offline" -> loads Bloomberg-exported file
    """
    mode = mode.lower().strip()

    if mode == "live":
        if not ticker:
            raise ValueError("mode='live' requires ticker='AAPL' (or similar).")
        return manager.fetch_live_data_yfinance(ticker)

    if mode == "offline":
        if not filepath:
            raise ValueError("mode='offline' requires filepath='path/to/bloomberg.csv'.")
        return manager.load_offline_data(filepath)

    raise ValueError("mode must be either 'live' or 'offline'.")


__all__ = ["MarketDataManager", "get_market_data"]
