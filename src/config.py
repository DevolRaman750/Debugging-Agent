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

# ── Evaluation Feedback Loop ──
# Path to the eval_config.json written by the Evaluator.
# The IntelligencePipeline reads this at startup to apply
# feedback-adjusted weights, thresholds, and pattern overrides.
EVAL_CONFIG_PATH = os.getenv(
    "EVAL_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "intel", "eval_config.json"),
)

# How often the evaluator loop runs (seconds).
# Default: 604 800 = 1 week.  Override with env var for testing.
EVAL_SCHEDULE_SECONDS = int(os.getenv("EVAL_SCHEDULE_SECONDS", str(7 * 24 * 60 * 60)))
