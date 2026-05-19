from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env.py", override=False)


load_dotenv_if_available()


def env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default

    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


DB_DIR = env_path("RAG_DB_DIR", PROJECT_ROOT / "db")
DEFAULT_FAISS_INDEX_PATH = env_path("RAG_FAISS_INDEX_PATH", DB_DIR / "faiss_index.bin")
DEFAULT_CHUNKS_PATH = env_path("RAG_CHUNKS_PATH", DB_DIR / "all_chunks.json")
DEFAULT_METADATA_PATH = env_path("RAG_METADATA_PATH", DB_DIR / "chunks_metadata.json")
