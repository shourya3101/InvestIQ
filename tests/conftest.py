"""
Shared pytest fixtures.

Windows holds a file lock on any open RotatingFileHandler, which prevents
TemporaryDirectory from cleaning up when logging tests finish.  The
cleanup_log_handlers fixture runs after every test to release those locks.
"""

import logging
import logging.handlers

import pytest


@pytest.fixture(autouse=True)
def cleanup_log_handlers():
    """Close and remove all RotatingFileHandlers from the root logger after each test."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.close()
            root.removeHandler(handler)
