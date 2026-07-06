"""Cross-encoder singleton: lazy load, cached failure, test injection."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

from core import singletons
from core.singletons import get_reranker, reset_reranker


def teardown_function():
    reset_reranker()  # never leak state between tests


def test_loads_once_and_caches():
    fake = MagicMock(name="cross_encoder")
    with patch.object(singletons, "_load_reranker", return_value=fake) as loader:
        assert get_reranker() is fake
        assert get_reranker() is fake
    loader.assert_called_once()


def test_load_failure_caches_none():
    with patch.object(singletons, "_load_reranker", side_effect=RuntimeError("no model")) as loader:
        assert get_reranker() is None
        assert get_reranker() is None   # failure cached, not retried
    loader.assert_called_once()


def test_reset_injects_fake():
    fake = MagicMock()
    reset_reranker(fake)
    assert get_reranker() is fake


def test_reset_no_args_forces_reload():
    reset_reranker(MagicMock())
    reset_reranker()
    with patch.object(singletons, "_load_reranker", return_value="fresh"):
        assert get_reranker() == "fresh"
