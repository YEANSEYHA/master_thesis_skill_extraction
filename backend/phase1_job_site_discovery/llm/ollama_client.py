import requests
from config import OLLAMA_MODEL, OLLAMA_URL


def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call local Ollama API (Qwen, Llama, etc.)."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "")
