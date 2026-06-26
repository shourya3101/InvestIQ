"""Structured logging configuration for InvestIQ."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5


def setup_logging(log_dir: Path, log_level: int = logging.DEBUG) -> None:
    """Configure a rotating file handler + console handler on the root logger.

    Calling this multiple times replaces the existing file handler so that
    the log file always points to *log_dir* / app.log.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any existing RotatingFileHandlers (may point to a stale path).
    for h in list(root.handlers):
        if isinstance(h, logging.handlers.RotatingFileHandler):
            h.close()
            root.removeHandler(h)

    # Add fresh rotating file handler.
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    fh.setFormatter(formatter)
    fh.setLevel(log_level)
    root.addHandler(fh)

    # Console handler — add only if none exists yet.
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(log_level)
        root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() first for file output."""
    return logging.getLogger(name)
