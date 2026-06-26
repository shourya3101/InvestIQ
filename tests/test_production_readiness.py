"""
Tests for production-readiness fixes — written BEFORE implementation (TDD).

Covers:
  1. config.py   — dev/prod env separation (get_log_level, is_debug)
  2. core/logging_config.py — structured logging to logs/app.log
  3. core/market_data_manager.py — yfinance daily cache
  4. api/routes.py — validate_ticker(), rate-limiter attached to app
  5. agents/memory_agent.py — WARNING prints → logger.warning
"""

import logging
import logging.handlers
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── 1. config.py — dev/prod environment ──────────────────────────────────────


def test_get_log_level_dev_returns_debug():
    from config import get_log_level
    assert get_log_level("dev") == logging.DEBUG


def test_get_log_level_prod_returns_info():
    from config import get_log_level
    assert get_log_level("prod") == logging.INFO


def test_get_log_level_production_alias_returns_info():
    from config import get_log_level
    assert get_log_level("production") == logging.INFO


def test_get_log_level_unknown_defaults_to_debug():
    from config import get_log_level
    assert get_log_level("staging") == logging.DEBUG


def test_is_debug_true_in_dev():
    from config import is_debug
    assert is_debug("dev") is True


def test_is_debug_false_in_prod():
    from config import is_debug
    assert is_debug("prod") is False


def test_is_debug_false_in_production():
    from config import is_debug
    assert is_debug("production") is False


def test_config_exports_app_env():
    from config import APP_ENV
    assert isinstance(APP_ENV, str)
    assert APP_ENV in ("dev", "prod", "production", "test", "staging")


def test_config_exports_log_dir():
    from config import LOG_DIR
    assert isinstance(LOG_DIR, Path)
    assert LOG_DIR.name == "logs"


def test_config_exports_log_level():
    from config import LOG_LEVEL
    assert LOG_LEVEL in (logging.DEBUG, logging.INFO, logging.WARNING)


def test_config_exports_debug_flag():
    from config import DEBUG
    assert isinstance(DEBUG, bool)


# ── 2. core/logging_config.py — structured logging ───────────────────────────


def _close_rfh():
    """Close and remove all RotatingFileHandlers from the root logger.

    Must be called before TemporaryDirectory exits on Windows, where open
    file handles prevent directory deletion (WinError 32).
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.handlers.RotatingFileHandler):
            h.close()
            root.removeHandler(h)


def test_setup_logging_creates_log_file():
    from core.logging_config import setup_logging
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        setup_logging(log_dir, logging.DEBUG)
        result = (log_dir / "app.log").exists()
        _close_rfh()
    assert result


def test_get_logger_returns_named_logger():
    from core.logging_config import get_logger
    logger = get_logger("agents.research")
    assert logger.name == "agents.research"


def test_log_message_appears_in_file():
    from core.logging_config import setup_logging, get_logger
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        setup_logging(log_dir, logging.DEBUG)
        logger = get_logger("test.probe")
        logger.info("hello_unique_sentinel_xyz")
        for h in logging.getLogger().handlers:
            h.flush()
        log_text = (log_dir / "app.log").read_text(encoding="utf-8")
        _close_rfh()
    assert "hello_unique_sentinel_xyz" in log_text


def test_log_format_includes_logger_name():
    from core.logging_config import setup_logging, get_logger
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        setup_logging(log_dir, logging.DEBUG)
        logger = get_logger("agents.trend")
        logger.warning("probe_message_for_format_check")
        for h in logging.getLogger().handlers:
            h.flush()
        log_text = (log_dir / "app.log").read_text(encoding="utf-8")
        _close_rfh()
    assert "agents.trend" in log_text


def test_log_format_includes_level():
    from core.logging_config import setup_logging, get_logger
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        setup_logging(log_dir, logging.DEBUG)
        logger = get_logger("test.level_check")
        logger.warning("level_test_sentinel")
        for h in logging.getLogger().handlers:
            h.flush()
        log_text = (log_dir / "app.log").read_text(encoding="utf-8")
        _close_rfh()
    assert "WARNING" in log_text


def test_setup_logging_creates_parent_dirs_if_missing():
    from core.logging_config import setup_logging
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "nested" / "logs"
        setup_logging(log_dir, logging.DEBUG)
        result = (log_dir / "app.log").exists()
        _close_rfh()
    assert result


def test_setup_logging_uses_rotating_file_handler():
    from core.logging_config import setup_logging
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        setup_logging(log_dir, logging.DEBUG)
        root = logging.getLogger()
        has_rotating = any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for h in root.handlers
        )
        _close_rfh()
    assert has_rotating


# ── 3. yfinance cache ─────────────────────────────────────────────────────────


def _fake_df():
    """Minimal DataFrame that passes _normalize_price_df."""
    import pandas as pd
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "Open":  [180.0, 181.0],
        "High":  [185.0, 186.0],
        "Low":   [179.0, 180.0],
        "Close": [183.0, 184.0],
        "Volume":[1_000_000, 1_100_000],
    })


def test_yfinance_called_once_for_two_fetches_of_same_ticker():
    from core.market_data_manager import MarketDataManager, clear_yfinance_cache
    clear_yfinance_cache()
    mgr = MarketDataManager()

    raw = _fake_df().set_index("Date")

    with patch("yfinance.download", return_value=raw) as mock_dl:
        mgr.fetch_live_data_yfinance("AAPL")
        mgr.fetch_live_data_yfinance("AAPL")

    assert mock_dl.call_count == 1, "yf.download should be called once (cache hit on 2nd call)"


def test_yfinance_called_twice_for_different_tickers():
    from core.market_data_manager import MarketDataManager, clear_yfinance_cache
    clear_yfinance_cache()
    mgr = MarketDataManager()

    raw = _fake_df().set_index("Date")

    with patch("yfinance.download", return_value=raw) as mock_dl:
        mgr.fetch_live_data_yfinance("AAPL")
        mgr.fetch_live_data_yfinance("MSFT")

    assert mock_dl.call_count == 2


def test_cached_result_equals_original():
    from core.market_data_manager import MarketDataManager, clear_yfinance_cache
    clear_yfinance_cache()
    mgr = MarketDataManager()

    raw = _fake_df().set_index("Date")

    with patch("yfinance.download", return_value=raw):
        df1 = mgr.fetch_live_data_yfinance("TSLA")
        df2 = mgr.fetch_live_data_yfinance("TSLA")

    pd.testing.assert_frame_equal(df1, df2)


def test_cache_returns_copy_not_same_reference():
    from core.market_data_manager import MarketDataManager, clear_yfinance_cache
    clear_yfinance_cache()
    mgr = MarketDataManager()

    raw = _fake_df().set_index("Date")

    with patch("yfinance.download", return_value=raw):
        df1 = mgr.fetch_live_data_yfinance("NVDA")
        df2 = mgr.fetch_live_data_yfinance("NVDA")

    assert df1 is not df2


def test_clear_yfinance_cache_forces_new_download():
    from core.market_data_manager import MarketDataManager, clear_yfinance_cache
    clear_yfinance_cache()
    mgr = MarketDataManager()

    raw = _fake_df().set_index("Date")

    with patch("yfinance.download", return_value=raw) as mock_dl:
        mgr.fetch_live_data_yfinance("AAPL")
        clear_yfinance_cache()
        mgr.fetch_live_data_yfinance("AAPL")

    assert mock_dl.call_count == 2


def test_cache_key_includes_today_date():
    """Changing the mocked date forces a cache miss."""
    from core.market_data_manager import MarketDataManager, clear_yfinance_cache
    clear_yfinance_cache()
    mgr = MarketDataManager()

    raw = _fake_df().set_index("Date")

    with patch("yfinance.download", return_value=raw) as mock_dl:
        with patch("core.market_data_manager.date") as mock_date:
            mock_date.today.return_value.isoformat.return_value = "2024-01-01"
            mgr.fetch_live_data_yfinance("AAPL")

        with patch("core.market_data_manager.date") as mock_date:
            mock_date.today.return_value.isoformat.return_value = "2024-01-02"
            mgr.fetch_live_data_yfinance("AAPL")

    assert mock_dl.call_count == 2, "Different dates should produce different cache keys"


# ── 4. validate_ticker ────────────────────────────────────────────────────────


def test_validate_ticker_aapl_passes():
    from api.routes import validate_ticker
    assert validate_ticker("AAPL") == "AAPL"


def test_validate_ticker_lowercased_input_gets_uppercased():
    from api.routes import validate_ticker
    assert validate_ticker("aapl") == "AAPL"


def test_validate_ticker_numeric_only_passes():
    from api.routes import validate_ticker
    assert validate_ticker("9988") == "9988"


def test_validate_ticker_single_char_passes():
    from api.routes import validate_ticker
    assert validate_ticker("A") == "A"


def test_validate_ticker_five_chars_passes():
    from api.routes import validate_ticker
    assert validate_ticker("GOOGL") == "GOOGL"


def test_validate_ticker_empty_raises():
    from fastapi import HTTPException
    from api.routes import validate_ticker
    with pytest.raises(HTTPException) as exc_info:
        validate_ticker("")
    assert exc_info.value.status_code == 422


def test_validate_ticker_six_chars_raises():
    from fastapi import HTTPException
    from api.routes import validate_ticker
    with pytest.raises(HTTPException) as exc_info:
        validate_ticker("TOOLNG")
    assert exc_info.value.status_code == 422


def test_validate_ticker_special_char_raises():
    from fastapi import HTTPException
    from api.routes import validate_ticker
    with pytest.raises(HTTPException) as exc_info:
        validate_ticker("AA!L")
    assert exc_info.value.status_code == 422


def test_validate_ticker_dot_raises():
    from fastapi import HTTPException
    from api.routes import validate_ticker
    with pytest.raises(HTTPException) as exc_info:
        validate_ticker("BRK.B")
    assert exc_info.value.status_code == 422


def test_validate_ticker_space_raises():
    from fastapi import HTTPException
    from api.routes import validate_ticker
    with pytest.raises(HTTPException) as exc_info:
        validate_ticker("AA PL")
    assert exc_info.value.status_code == 422


# ── 4b. Rate limiting attached to app ────────────────────────────────────────


def test_rate_limiter_is_attached_to_app_state():
    from api.routes import app
    assert hasattr(app.state, "limiter"), "app.state.limiter must be set for slowapi"


def test_rate_limit_exceeded_handler_registered():
    from api.routes import app
    from slowapi.errors import RateLimitExceeded
    assert RateLimitExceeded in app.exception_handlers


def test_analyze_endpoint_returns_200_on_first_request():
    """Smoke-test that rate-limiting setup doesn't break the endpoint."""
    from fastapi.testclient import TestClient
    from api.routes import app

    async def fake_gen(*a, **kw):
        yield {"event": "complete", "data": {"ticker": "TEST", "memo": {}, "risk": {}, "trend": {}, "sentiment": {}, "research": {}, "debate": None, "memory": None, "pipeline_trace": [], "total_runtime_seconds": 0.1, "mode": "live", "as_of": "2024-01-01T00:00:00"}}

    with patch("api.routes.stream_pipeline_events", fake_gen):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/analyze/stream", json={
            "ticker": "AAPL", "question": "q", "mode": "live", "days_back": 30, "run_debate": False
        })

    assert resp.status_code == 200


# ── 5. memory_agent uses logger not print ─────────────────────────────────────


def test_memory_agent_save_memo_uses_logger_not_print(capsys):
    """save_memo failure must log via logger.warning, not print()."""
    import agents.memory_agent as mem_mod

    # Force an exception in save_memo by making _memo_to_entry raise
    with patch.object(mem_mod, "_memo_to_entry", side_effect=RuntimeError("forced")):
        mock_memo = MagicMock()
        mock_memo.ticker = "TEST"
        mem_mod.save_memo(mock_memo)  # must not raise

    captured = capsys.readouterr()
    # The WARNING must NOT appear on stdout/stderr (it should go to logger)
    assert "[WARNING]" not in captured.out
    assert "[WARNING]" not in captured.err


def test_memory_agent_load_history_uses_logger_not_print(capsys, tmp_path):
    """load_history failure must log via logger.warning, not print()."""
    import agents.memory_agent as mem_mod

    # Make open() raise inside the function
    with patch("builtins.open", side_effect=PermissionError("no access")):
        result = mem_mod.load_history("FAKE_TICKER_THAT_EXISTS", n=5)

    assert result == []
    captured = capsys.readouterr()
    assert "[WARNING]" not in captured.out
    assert "[WARNING]" not in captured.err
