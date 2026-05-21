from pathlib import Path

# Базовые пути
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"

# Параметры чанкинга
CHUNK_SIZE = 400
OVERLAP = 100

# Параметры поиска
TOP_K = 3