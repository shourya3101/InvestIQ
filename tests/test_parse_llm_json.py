"""
Tests for core.utils.parse_llm_json — written BEFORE implementation (TDD).

Covers: happy path, markdown fence stripping, missing keys, non-dict JSON,
invalid JSON, empty input, and whitespace tolerance.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.utils import parse_llm_json


# ── Happy path ────────────────────────────────────────────────────────────────


def test_parses_plain_json_with_all_required_keys():
    raw = '{"arguments": ["a", "b"], "confidence": 0.7, "key_evidence": ["E1"]}'
    result = parse_llm_json(raw, ["arguments", "confidence", "key_evidence"])
    assert result == {"arguments": ["a", "b"], "confidence": 0.7, "key_evidence": ["E1"]}


def test_returns_dict_when_extra_keys_present():
    raw = '{"a": 1, "b": 2, "c": 3}'
    result = parse_llm_json(raw, ["a"])
    assert result is not None
    assert result["b"] == 2


def test_handles_whitespace_around_json():
    raw = '   \n  {"x": 1}  \n  '
    result = parse_llm_json(raw, ["x"])
    assert result == {"x": 1}


# ── Markdown fence stripping ──────────────────────────────────────────────────


def test_strips_triple_backtick_json_fence():
    raw = '```json\n{"key": "value"}\n```'
    result = parse_llm_json(raw, ["key"])
    assert result == {"key": "value"}


def test_strips_plain_triple_backtick_fence():
    raw = '```\n{"key": "value"}\n```'
    result = parse_llm_json(raw, ["key"])
    assert result == {"key": "value"}


def test_strips_fence_with_surrounding_whitespace():
    raw = '  ```json\n{"score": 0.9}\n```  '
    result = parse_llm_json(raw, ["score"])
    assert result is not None
    assert result["score"] == 0.9


# ── Failure cases ─────────────────────────────────────────────────────────────


def test_returns_none_for_missing_required_key():
    raw = '{"arguments": ["a"]}'
    result = parse_llm_json(raw, ["arguments", "confidence"])
    assert result is None


def test_returns_none_for_json_array_not_dict():
    raw = '["a", "b", "c"]'
    result = parse_llm_json(raw, [])
    assert result is None


def test_returns_none_for_json_scalar():
    raw = "42"
    result = parse_llm_json(raw, [])
    assert result is None


def test_returns_none_for_invalid_json():
    raw = "{not valid json}"
    result = parse_llm_json(raw, [])
    assert result is None


def test_returns_none_for_empty_string():
    result = parse_llm_json("", ["key"])
    assert result is None


def test_returns_none_for_whitespace_only_string():
    result = parse_llm_json("   \n  ", ["key"])
    assert result is None


# ── Debate agent still works via import ──────────────────────────────────────


def test_debate_agent_imports_parse_llm_json_from_core_utils():
    """debate_agent must re-export via core.utils, not define its own copy."""
    import importlib, inspect
    import agents.debate_agent as debate_mod
    import core.utils as utils_mod

    # debate_agent must NOT define _parse_llm_json locally
    assert not hasattr(debate_mod, "_parse_llm_json"), (
        "debate_agent still defines _parse_llm_json locally; "
        "it should import from core.utils instead"
    )
    # parse_llm_json must live in core.utils
    assert hasattr(utils_mod, "parse_llm_json")
