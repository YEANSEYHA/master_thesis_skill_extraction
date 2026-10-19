"""DeepSeek JSON calls with local-Ollama fallback (same contract as
archive/phase2_playwright_crawler/crawler/llm.py, kept standalone so this
package has no import path into the other crawler)."""
import requests
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_URL, DEEPSEEK_MODEL,
    ACTIVE_MODEL, OLLAMA_URL,
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
            # v4 models think by default — token waste for JSON extraction
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
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 16384},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def ask_llm(prompt: str, timeout: int = 120) -> str:
    """Raw JSON-text response. DeepSeek first (if key set), Ollama fallback."""
    if DEEPSEEK_API_KEY:
        try:
            out = _ask_deepseek(prompt, timeout)
            print(f"[LLM] deepseek ({DEEPSEEK_MODEL})")
            return out
        except Exception as e:
            print(f"[LLM] DeepSeek failed ({e}) — falling back to Ollama")
    out = _ask_ollama(prompt, max(timeout, 300))
    print(f"[LLM] ollama ({ACTIVE_MODEL})")
    return out
