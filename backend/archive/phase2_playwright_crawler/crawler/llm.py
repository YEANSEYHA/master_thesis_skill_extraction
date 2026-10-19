"""Shared LLM backend for crawler navigation/pagination calls.

Backend is chosen by CRAWLER_LLM in .env:
  auto (default) — DeepSeek API if a key is set (~2-5s, parallel-safe),
                   else local Ollama (qwen, ~30-60s, serialized on one GPU)
  deepseek       — force the API
  ollama         — force local qwen (ACTIVE_MODEL) even if a key is set
If the API errors, calls fall back to Ollama automatically.
"""
import requests
from config import (
    ACTIVE_MODEL, OLLAMA_URL, CRAWLER_LLM,
    DEEPSEEK_API_KEY, DEEPSEEK_URL, DEEPSEEK_MODEL,
)


def _ask_deepseek(prompt: str, timeout: int) -> str:
    r = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            # v4 models think by default — pure token waste for JSON extraction
            "thinking": {"type": "disabled"},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _ask_ollama(prompt: str, timeout: int) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": ACTIVE_MODEL,
            "prompt": prompt,
            "stream": False,
            # JSON-constrained + big context (default 4096 truncates the
            # ~15k-char HTML and qwen drifts into prose without format=json)
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 16384},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def ask_llm(prompt: str, timeout: int = 120) -> str:
    """Return the model's raw JSON-text response for `prompt`.

    DeepSeek requires the word "json" somewhere in the prompt when
    response_format=json_object — all crawler prompts already ask for JSON.
    Raises only if BOTH backends fail.
    """
    use_deepseek = (CRAWLER_LLM == "deepseek"
                    or (CRAWLER_LLM == "auto" and bool(DEEPSEEK_API_KEY)))
    if use_deepseek:
        try:
            out = _ask_deepseek(prompt, timeout)
            print(f"[LLM] deepseek ({DEEPSEEK_MODEL})")
            return out
        except Exception as e:
            print(f"[LLM] DeepSeek failed ({e}) — falling back to Ollama")
    out = _ask_ollama(prompt, max(timeout, 300))
    print(f"[LLM] ollama ({ACTIVE_MODEL})")
    return out
