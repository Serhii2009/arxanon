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
        import time as _time
        import urllib.error as _ue

        last_exc: Exception | None = None
        for _wait in (0, 5, 10):
            if _wait:
                _time.sleep(_wait)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = _json.loads(r.read())
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                elif "content" in data:
                    return next(
                        (c["text"] for c in data["content"] if c.get("type") == "text"),
                        "",
                    )
                elif "error" in data:
                    last_exc = ValueError(f"OpenRouter error: {data['error']}")
                    continue
                else:
                    raise ValueError(
                        f"Unknown OpenRouter response format: {list(data.keys())}"
                    )
            except _ue.HTTPError as exc:
                if exc.code == 429:
                    last_exc = exc
                    continue
                raise
        raise last_exc or RuntimeError("OpenRouter: all retries exhausted")
    else:
        import ollama as ollama_lib

        client = ollama_lib.Client(host=config.OLLAMA_BASE_URL, timeout=timeout)
        response = client.chat(
            model=config.GEMMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        return getattr(response.message, "content", "") or ""
