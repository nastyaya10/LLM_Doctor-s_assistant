import sys
import json
import subprocess
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

# 1. Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Импорт путей и КОНФИГА
from src.services.paths import RAW_DATA, PARSED_DATA, CHUNKED_DATA, DEFAULT_FAISS_INDEX_PATH, DEFAULT_METADATA_PATH
from src.config import EMBEDDER_MODEL_NAME, CHUNK_SIZE, OVERLAP
from scripts.chunker import append_chunks_to_json

def run_pdf_parser():
    """Запускает парсинг PDF из data/raw в data/parsed."""
    if not list(RAW_DATA.glob("*.pdf")):
        if list(PARSED_DATA.glob("*.json")):
            print(f"PDF не найдены в {RAW_DATA}; используем готовые JSON из {PARSED_DATA}.")
            return
        raise FileNotFoundError(
            f"В {RAW_DATA} нет PDF-файлов, а в {PARSED_DATA} нет готовых JSON."
        )

    parser_script = PROJECT_ROOT / "scripts" / "start_pdf.py"
    PARSED_DATA.mkdir(parents=True, exist_ok=True)
    print(f"Запуск парсинга PDF из {RAW_DATA} в {PARSED_DATA}...")
    result = subprocess.run(
        [sys.executable, str(parser_script)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        details = "\n".join(part for part in (result.stdout, result.stderr) if part.strip())
        raise RuntimeError(f"Парсинг PDF завершился с ошибкой:\n{details}")
    print(result.stdout)


def chunk_all_json_files():
    """Применяет chunking из data/parsed в data/chunked."""
    CHUNKED_DATA.mkdir(parents=True, exist_ok=True)
    json_files = list(PARSED_DATA.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"В папке {PARSED_DATA} нет JSON-файлов.")

    # Используем файл в data/chunked как общий контейнер
    all_chunks_file = CHUNKED_DATA / "all_chunks.json"
    if all_chunks_file.exists():
        all_chunks_file.unlink()

    for input_path in json_files:
        print(f"Обработка файла: {input_path}")
        # Используем параметры из config.py
        append_chunks_to_json(
            input_file_path=str(input_path),
            output_json_path=str(all_chunks_file),
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )
    print(f"Чанки сохранены в {all_chunks_file}")


def build_faiss_index():
    """Строит FAISS-индекс и сохраняет в data/processed."""
    all_chunks_file = CHUNKED_DATA / "all_chunks.json"
    with open(all_chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Используем имя модели из config.py
    model = SentenceTransformer(EMBEDDER_MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(DEFAULT_FAISS_INDEX_PATH))
    with open(DEFAULT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=True, indent=2)
    print(f"Индекс сохранен в {DEFAULT_FAISS_INDEX_PATH}")


def main():
    run_pdf_parser()
    chunk_all_json_files()
    build_faiss_index()


if __name__ == "__main__":
    main()
