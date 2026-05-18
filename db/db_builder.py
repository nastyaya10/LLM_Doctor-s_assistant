import os
import sys
import json
import subprocess

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import faiss
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.dirname(__file__)
sys.path.insert(0, DB_DIR)

from chunking.chunker import append_chunks_to_json

# Пути
PDF_PARSER_SCRIPT = os.path.join(DB_DIR, "start_pdf.py")
PDF_OUTPUT_DIR = os.path.join(DB_DIR, "output")
ALL_CHUNKS_JSON = os.path.join(DB_DIR, "all_chunks.json")
FAISS_INDEX_PATH = os.path.join(DB_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DB_DIR, "chunks_metadata.json")

# Модель для эмбеддингов
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


def load_existing_chunks():
    if not os.path.exists(ALL_CHUNKS_JSON):
        return []
    with open(ALL_CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        return []
    return chunks


def get_json_titles(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {item.get("title") for item in data if isinstance(item, dict) and item.get("title")}
    if isinstance(data, dict) and "articles" in data:
        return {item.get("title") for item in data["articles"] if isinstance(item, dict) and item.get("title")}
    if isinstance(data, dict) and data.get("title"):
        return {data["title"]}
    return set()


def run_pdf_parser():
    """Запускает парсинг PDF-файлов."""
    if not os.path.exists(PDF_PARSER_SCRIPT):
        raise FileNotFoundError(f"Скрипт {PDF_PARSER_SCRIPT} не найден.")
    print("Запуск парсинга PDF...")
    result = subprocess.run(
        [sys.executable, PDF_PARSER_SCRIPT],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode != 0:
        print("Ошибка при парсинге PDF:")
        print(result.stderr)
        raise RuntimeError("Парсинг завершился с ошибкой.")
    print(result.stdout)


def chunk_all_json_files():
    """Применяет chunking ко всем JSON из output и сохраняет в общий файл."""
    if not os.path.isdir(PDF_OUTPUT_DIR):
        raise FileNotFoundError(f"Директория {PDF_OUTPUT_DIR} не найдена.")

    json_files = [
        os.path.join(PDF_OUTPUT_DIR, f)
        for f in os.listdir(PDF_OUTPUT_DIR)
        if f.endswith(".json")
    ]
    if not json_files:
        raise FileNotFoundError(f"В папке {PDF_OUTPUT_DIR} нет JSON-файлов.")

    existing_chunks = load_existing_chunks()
    processed_titles = {chunk.get("title") for chunk in existing_chunks if chunk.get("title")}

    added_files = 0
    for input_path in json_files:
        json_titles = get_json_titles(input_path)
        if json_titles and json_titles.issubset(processed_titles):
            print(f"Уже обработан, пропускаем: {input_path}")
            continue

        print(f"Обработка файла: {input_path}")
        append_chunks_to_json(
            input_file_path=input_path,
            output_json_path=ALL_CHUNKS_JSON,
            chunk_size=400,
            overlap=100,
            text_key="text",
            title_key="title",
            tables_key="tables",
            table_insertion_method="append"
        )
        processed_titles.update(json_titles)
        added_files += 1

    if added_files == 0:
        print("Новых JSON-файлов для чанкинга нет.")
    else:
        print(f"Все чанки сохранены в {ALL_CHUNKS_JSON}")


def build_faiss_index():
    """Строит FAISS-индекс по чанкам и сохраняет его вместе с метаданными."""
    # Загружаем чанки
    with open(ALL_CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("Нет чанков для индексации.")

    print(f"Загружено {len(chunks)} чанков. Создание эмбеддингов...")

    # Инициализируем модель эмбеддингов
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Генерируем эмбеддинги для каждого чанка
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Нормализуем векторы для косинусного сходства через inner product
    faiss.normalize_L2(embeddings)

    # Размерность векторов
    d = embeddings.shape[1]

    # Создаём индекс
    index = faiss.IndexFlatIP(d)  # Inner product (cosine после нормализации)
    index.add(embeddings)

    # Сохраняем индекс
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS индекс сохранён в {FAISS_INDEX_PATH}")

    # Сохраняем метаданные (исходные чанки) для последующего поиска
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Метаданные сохранены в {METADATA_PATH}")


def main():
    print("=== Шаг 1: Парсинг PDF ===")
    run_pdf_parser()

    print("\n=== Шаг 2: Чанкование всех JSON ===")
    chunk_all_json_files()

    print("\n=== Шаг 3: Создание FAISS базы данных ===")
    build_faiss_index()

    print("\nГотово! База данных FAISS создана.")


main()
