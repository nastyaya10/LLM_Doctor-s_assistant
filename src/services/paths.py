from __future__ import annotations

import os
from pathlib import Path

# Корень проекта (поднимаемся из src/services/ до корня)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

# --- НОВАЯ СТРУКТУРА ПУТЕЙ ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
PARSED_DATA = DATA_DIR / "parsed"
CHUNKED_DATA = DATA_DIR / "chunked"

# Теперь пути к базе FAISS и JSON-чанкам ведут в data/processed/
DEFAULT_FAISS_INDEX_PATH = env_path("RAG_FAISS_INDEX_PATH", PROCESSED_DATA / "faiss_index.bin")
DEFAULT_CHUNKS_PATH = env_path("RAG_CHUNKS_PATH", PROCESSED_DATA / "all_chunks.json")
DEFAULT_METADATA_PATH = env_path("RAG_METADATA_PATH", PROCESSED_DATA / "chunks_metadata.json")