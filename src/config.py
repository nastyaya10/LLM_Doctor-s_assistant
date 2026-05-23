import os
from pathlib import Path

# --- Базовые параметры RAG (поиск и чанкинг) ---
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 400))
OVERLAP = int(os.getenv("RAG_OVERLAP", 100))
TOP_K = int(os.getenv("RAG_TOP_K", 10))
TOP_R = int(os.getenv("RAG_TOP_R", 5))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", 0.55))
NEIGHBOR_RADIUS = int(os.getenv("RAG_NEIGHBOR_RADIUS", 1))

# --- Модели для поиска (Embeddings & Reranker) ---
EMBEDDER_MODEL_NAME = os.getenv("RAG_EMBEDDER_MODEL", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER = os.getenv("RAG_USE_RERANKER", "False").lower() in ("1", "true", "yes", "on")

# --- Лимиты и настройки LLM (YandexGPT/OpenAI) ---
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", 30000))
LLM_BASE_URL = os.getenv("BASE_URL", "https://ai.api.cloud.yandex.net/v1")
LLM_MODEL_NAME = os.getenv("MODEL", "yandexgpt/rc")
LOG_RAG_CONTEXT = os.getenv("LOG_RAG_CONTEXT", "1").lower() not in ("0", "false", "no")
