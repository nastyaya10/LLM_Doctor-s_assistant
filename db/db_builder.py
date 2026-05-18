import os
import sys
import json
import subprocess
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Добавляем родительскую директорию в sys.path для импорта chunking.chunker
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chunking.chunker import append_chunks_to_json

# Пути
PDF_PARSER_SCRIPT = os.path.join("pdf_parser", "start_pdf.py")
PDF_OUTPUT_DIR = os.path.join("pdf_parser", "output")
ALL_CHUNKS_JSON = "all_chunks.json"
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "chunks_metadata.json"

# Модель для эмбеддингов
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


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
    """Применяет chunking ко всем JSON из pdf_parser/output и сохраняет в общий файл."""
    if not os.path.isdir(PDF_OUTPUT_DIR):
        raise FileNotFoundError(f"Директория {PDF_OUTPUT_DIR} не найдена.")

    json_files = [
        os.path.join(PDF_OUTPUT_DIR, f)
        for f in os.listdir(PDF_OUTPUT_DIR)
        if f.endswith(".json")
    ]
    if not json_files:
        raise FileNotFoundError(f"В папке {PDF_OUTPUT_DIR} нет JSON-файлов.")

    # Удаляем старый общий файл, чтобы начать заново
    if os.path.exists(ALL_CHUNKS_JSON):
        os.remove(ALL_CHUNKS_JSON)

    for input_path in json_files:
        print(f"Обработка файла: {input_path}")
        append_chunks_to_json(
            input_file_path=input_path,
            output_json_path=ALL_CHUNKS_JSON,
            chunk_size=400,
            overlap=100,
            text_key="text",
            title_key="title",
            tables_key=None,
            table_insertion_method="append"
        )
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