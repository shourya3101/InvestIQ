"""Shared utilities for the URECA agent pipeline."""

from __future__ import annotations

import json


def parse_llm_json(raw: str, required_keys: list) -> dict | None:
    """Safely parse JSON from an LLM response.

    Strips markdown code fences, attempts json.loads, and verifies that
    every key in *required_keys* is present.  Returns None on any failure.
    """
    try:
        text = raw.strip()
        if not text:
            return None
        if text.startswith("```"):
            first_newline = text.index("\n")
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        for key in required_keys:
            if key not in data:
                return None
        return data
    except Exception:
        return None
