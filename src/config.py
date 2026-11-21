from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "kb"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
MEMORY_DIR = PROJECT_ROOT / "memory"
INDEX_DIR = PROJECT_ROOT / "index"
DB_DIR = PROJECT_ROOT / "db"

VARIANT = "V05"
DOMAIN = "E-commerce orders and returns"
TWIST = "E - Small context window"
CONTEXT_WINDOW_TOKENS = 2048
TRIM_WINDOW_K_TURNS = 3
CHAT_MODEL = os.getenv("RECALLBOT_CHAT_MODEL", "qwen3:1.7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TOKENIZER_ID = os.getenv("RECALLBOT_TOKENIZER", "Qwen/Qwen3-1.7B")

# Values must match the project specification's mandatory logging schema.
VALID_CONFIGS = {"full", "trim", "summary", "no_rag", "hybrid", "hybrid_sem", "hybrid_mem"}
VALID_CALL_TYPES = {"nlu", "chat", "summary", "extract", "importance", "reflect", "embed", "tool", "judge"}
