"""Unified LLM client: OpenRouter (cloud) or Ollama (local)."""
from __future__ import annotations

from . import config


def call_llm(
    prompt: str,
    timeout: int = 60,
    temperature: float = 0.3,
) -> str:
    """Call LLM via OpenRouter if API key is set, otherwise fall back to local Ollama.

    OpenRouter uses google/gemma-2-27b-it (13x larger than local gemma4:e2b) and
    produces dramatically better reasoning and analysis.
    """
    if config.USE_OPENROUTER:
        import json as _json
        import urllib.request

        payload = _json.dumps({
            "model": config.GEMMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2000,
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = _json.loads(r.read())
        return data["choices"][0]["message"]["content"]
    else:
        import ollama as ollama_lib

        client = ollama_lib.Client(host=config.OLLAMA_BASE_URL, timeout=timeout)
        response = client.chat(
            model=config.GEMMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        return getattr(response.message, "content", "") or ""
