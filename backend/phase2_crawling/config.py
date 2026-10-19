import os
from pathlib import Path
from dotenv import load_dotenv

# Single source of truth for all phases: backend/.env
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE.parent / ".env")

DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY")
                    or os.getenv("deepseek_api")
                    or os.getenv("deekseek_api")  # typo'd name in .env
                    or "")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Local Ollama fallback (same envs as the main phase2 crawler)
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "qwen3:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

OUTPUT_DIR = _HERE / "outputs"
