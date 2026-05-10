"""
LLM provider abstraction layer.

Supports:
  • **GroqProvider**   – primary (fast, cost-effective)
  • **ClaudeProvider** – fallback (high-quality reasoning)
  • **LLMRouter**      – try primary, fall back on failure

Dependencies are imported lazily so the module can be loaded even when
``groq`` or ``anthropic`` are not installed.
"""

from __future__ import annotations

import os
from typing import Optional


# ── Base interface ──────────────────────────────────────────────────


class LLMProvider:
    """Abstract base for all LLM providers."""

    def generate(self, system: str, user: str) -> str:
        """Send a system + user prompt and return the assistant's text.

        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement generate()")


# ── Groq ────────────────────────────────────────────────────────────


class GroqProvider(LLMProvider):
    """LLM provider backed by the Groq API (``pip install groq``).

    Requires the ``GROQ_API_KEY`` environment variable.

    Parameters
    ----------
    model : str
        Model identifier (default: ``llama-3.3-70b-versatile``).
    temperature : float
        Sampling temperature (default: 0.3).
    max_tokens : int
        Maximum tokens in the response (default: 1024).
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> None:
        try:
            from groq import Groq  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'groq' package is required for GroqProvider. "
                "Install it with: pip install groq"
            )

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get your key at https://console.groq.com and set it:\n"
                '  $env:GROQ_API_KEY = "gsk_..."   (PowerShell)\n'
                '  export GROQ_API_KEY="gsk_..."    (bash)'
            )

        self._client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


# ── Claude (Anthropic) ──────────────────────────────────────────────


class ClaudeProvider(LLMProvider):
    """LLM provider backed by the Anthropic Messages API (``pip install anthropic``).

    Requires the ``ANTHROPIC_API_KEY`` environment variable.

    Parameters
    ----------
    model : str
        Model identifier (default: ``claude-sonnet-4-20250514``).
    temperature : float
        Sampling temperature (default: 0.3).
    max_tokens : int
        Maximum tokens in the response (default: 1024).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        try:
            from anthropic import Anthropic  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for ClaudeProvider. "
                "Install it with: pip install anthropic"
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Get your key at https://console.anthropic.com and set it:\n"
                '  $env:ANTHROPIC_API_KEY = "sk-ant-..."   (PowerShell)\n'
                '  export ANTHROPIC_API_KEY="sk-ant-..."    (bash)'
            )

        self._client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.content[0].text


# ── OpenAI ──────────────────────────────────────────────────────────


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI API (``pip install openai``).

    Requires the ``OPENAI_API_KEY`` environment variable.

    Parameters
    ----------
    model : str
        Model identifier (default: ``gpt-4o-mini``).
    temperature : float
        Sampling temperature (default: 0.3).
    max_tokens : int
        Maximum tokens in the response (default: 1024).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> None:
        try:
            from openai import OpenAI  # noqa: F811
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Get your key at https://platform.openai.com/api-keys and set it:\n"
                '  $env:OPENAI_API_KEY = "sk-..."   (PowerShell)\n'
                '  export OPENAI_API_KEY="sk-..."    (bash)'
            )

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


# ── Router (primary + fallback) ─────────────────────────────────────


class LLMRouter(LLMProvider):
    """Try *primary* first; on any exception fall back to *fallback*.

    If *fallback* is ``None`` or also fails, the exception propagates.

    Parameters
    ----------
    primary : LLMProvider
        First provider to try.
    fallback : LLMProvider | None
        Provider to use when *primary* raises.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallback: Optional[LLMProvider] = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def generate(self, system: str, user: str) -> str:
        try:
            return self.primary.generate(system, user)
        except Exception as primary_err:
            if self.fallback is None:
                raise

            try:
                return self.fallback.generate(system, user)
            except Exception as fallback_err:
                raise RuntimeError(
                    f"Both LLM providers failed.\n"
                    f"  Primary ({type(self.primary).__name__}): {primary_err}\n"
                    f"  Fallback ({type(self.fallback).__name__}): {fallback_err}"
                ) from fallback_err


# ── Smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("LLM providers module loaded successfully.\n")

    groq_key = os.environ.get("GROQ_API_KEY")

    if groq_key:
        print("GROQ_API_KEY found – running a tiny live test …")
        provider = GroqProvider()
        reply = provider.generate(
            system="You are a test assistant.",
            user="Say 'ok' only.",
        )
        print(f"Groq response: {reply!r}")
    else:
        print("Skipping live test: missing GROQ_API_KEY")

    print("\nDone.")
