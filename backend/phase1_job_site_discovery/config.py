import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))  # single source: backend/.env

# LLM Settings
CLAUDE_MODEL = "claude-sonnet-4-20250514"
GPT_MODEL = "gpt-5.2"
GLM_MODEL = "glm-4-plus"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# HTTP Settings
REQUEST_TIMEOUT = 15.0
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
HTML_MAX_LENGTH = 15000
HTML_PROMPT_MAX_LENGTH = 8000

# Verification Settings
MIN_CONFIDENCE = 0.7

# Database Settings
DB_HOST = os.getenv("PG_HOST")
DB_PORT = os.getenv("PG_PORT", 5432)
DB_USER = os.getenv("PG_USER")
DB_PASS = os.getenv("PG_PASS")
DB_NAME = os.getenv("PG_NAME")
DB_SSLMODE = os.getenv("PG_SSLMODE", "prefer")  # "require" for hosted DBs, "disable" for local
DB_TABLE = "job_sites"
