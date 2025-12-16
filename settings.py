# settings.py
import os
from dotenv import load_dotenv

# Load .env once at startup
load_dotenv()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


# Core keys
OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")

# Separate models
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDINGS", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDINGS", "all-MiniLM-L6-v2")

SUMMARIZER_MODEL = os.getenv("SUMMARIZER_LLM", "gpt-4o-mini")
LLM_MODEL = os.getenv("LLM", "gpt-4o-mini")
LOCAL_MODEL_TIMEOUT = int(os.getenv("LOCAL_MODEL_TIMEOUT", "300"))

# STM / MTM defaults (you can move these later)
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
ANSWER_FRACTION = float(os.getenv("ANSWER_FRACTION", "0.25"))
MTM_TOP_K = int(os.getenv("MTM_TOP_K", "5"))
MTM_MAX_PER_USER = int(os.getenv("MTM_MAX_PER_USER", "0"))

# Live web search defaults (used by the MCP server and AgentMTM integration)
WEB_SEARCH_ENABLED = _env_flag("WEB_SEARCH_ENABLED", "0")
WEB_SEARCH_AUTO = _env_flag("WEB_SEARCH_AUTO", "1")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
WEB_SEARCH_MAX_CONTEXT_TOKENS = int(os.getenv("WEB_SEARCH_MAX_CONTEXT_TOKENS", "512"))
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "10"))
WEB_SEARCH_ALLOW_NETWORK = _env_flag("WEB_SEARCH_ALLOW_NETWORK", "1")
