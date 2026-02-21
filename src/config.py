"""
Central configuration — loads secrets from .env file.
"""
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── LLM (Groq) ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Database ──
# Auto-detect: if MONGODB_URI is set → use MongoDB, otherwise SQLite
MONGODB_URI = os.getenv("MONGODB_URI", "")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "traceroot.db")
DB_BACKEND = os.getenv("DB_BACKEND", "mongodb" if MONGODB_URI else "sqlite")
