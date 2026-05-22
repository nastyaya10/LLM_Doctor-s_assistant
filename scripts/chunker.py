import json
import os
from typing import Optional

# Импортируем параметры из вашего нового конфига
from src.config import CHUNK_SIZE, OVERLAP


def append_chunks_to_json(
        input_file_path: str,
        output_json_path: str,
        chunk_size: int = CHUNK_SIZE,  # Теперь берем из конфига
        overlap: int = OVERLAP,  # Теперь берем из конфига
        text_key: str = "text",
        title_key: str = "title",
        tables_key: Optional[str] = None,
        table_insertion_method: str = "append"
) -> int:
    """
    Разбивает текст на чанки с использованием параметров из src.config.
    """

    # 1. Загружаем существующий массив
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                existing_chunks = json.load(f)
            if not isinstance(existing_chunks, list):
                existing_chunks = []
        except (json.JSONDecodeError, FileNotFoundError):
            existing_chunks = []
    else:
        existing_chunks = []

    global_index_start = len(existing_chunks) + 1

    # 2. Загружаем входные данные
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    articles = data["articles"] if isinstance(data, dict) and "articles" in data else (
        data if isinstance(data, list) else [data])

    # 3. Вспомогательные функции (текст и таблицы)
    def chunk_text(text: str, size: int, ovlp: int):
        if not text or size <= 0: return []
        step = size - ovlp
        if step <= 0: raise ValueError("overlap должен быть меньше chunk_size")
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + size, text_len)
            chunks.append(text[start:end])
            if end == text_len: break
            start += step
        return chunks

    def format_table_as_markdown(table: dict) -> str:
        caption = table.get("caption", "")
        md = table.get("markdown") or ""
        if not md and "data" in table:
            rows = table["data"]
            if not rows: return ""
            header = "| " + " | ".join(str(cell) for cell in rows[0]) + " |"
            sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            body = "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows[1:])
            md = f"{header}\n{sep}\n{body}"
        return f"**{caption}**\n\n{md}" if caption else md

    # 4. Чанкинг
    new_chunks = []
    file_chunk_index = 1

    for article in articles:
        text = article.get(text_key, "") if isinstance(article, dict) else article
        title = article.get(title_key, "untitled") if isinstance(article, dict) else "untitled"

        if tables_key and isinstance(article, dict) and tables_key in article:
            table_texts = [format_table_as_markdown(tbl) for tbl in article[tables_key] if
                           format_table_as_markdown(tbl)]
            if table_texts:
                text += "\n\n[ТАБЛИЦЫ]\n" + "\n\n".join(table_texts)

        for chunk_text_val in chunk_text(text, chunk_size, overlap):
            new_chunks.append({
                "text": chunk_text_val,
                "title": title,
                "file_chunk_index": file_chunk_index,
                "global_chunk_index": global_index_start
            })
            file_chunk_index += 1
            global_index_start += 1

    # 5. Запись
    existing_chunks.extend(new_chunks)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(existing_chunks, f, ensure_ascii=True, indent=2)

    return len(existing_chunks)
