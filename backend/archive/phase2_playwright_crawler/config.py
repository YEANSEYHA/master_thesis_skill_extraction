import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))  # single source: backend/.env

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
MODE = os.getenv("MODE", "development")  # "development" = local only, "production" = upload to R2
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "qwen3:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# DeepSeek API (optional) — when a key is present, navigation/pagination LLM
# calls go to the API (~2-5s, parallel-safe) instead of local Ollama
# (~30-60s, serialized across concurrent site crawls). Ollama stays the
# fallback if the API errors.
DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY")
                    or os.getenv("deepseek_api")
                    or os.getenv("deekseek_api")  # typo'd name in .env
                    or "")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
# deepseek-v4-flash non-thinking = the fast/cheap tier ($0.14/M in, $0.28/M out).
# ("deepseek-chat" is the same model but that alias is deprecated 2026-07-24.)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Which backend the crawler's nav/pagination LLM calls use:
#   "auto"     (default) — DeepSeek if a key is set, else Ollama
#   "deepseek" — force the API
#   "ollama"   — force local qwen (ACTIVE_MODEL) even if a key is set
CRAWLER_LLM = os.getenv("CRAWLER_LLM", "auto").lower()

# --------------------------------------------------
# Cloudflare R2
# --------------------------------------------------
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "thesis-raw-htmls")

# --------------------------------------------------
# PostgreSQL (Neon) — shared with Phase 1
# --------------------------------------------------
DB_HOST = os.getenv("PG_HOST")
DB_PORT = os.getenv("PG_PORT", 5432)
DB_USER = os.getenv("PG_USER")
DB_PASS = os.getenv("PG_PASS")
DB_NAME = os.getenv("PG_NAME")