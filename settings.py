# settings.py
import os
from dotenv import load_dotenv

# Load .env once at startup
load_dotenv()

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
