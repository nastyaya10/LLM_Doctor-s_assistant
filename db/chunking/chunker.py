import json
import os
from typing import Optional


def append_chunks_to_json(
        input_file_path: str,
        output_json_path: str,
        chunk_size: int = 400,
        overlap: int = 100,
        text_key: str = "text",
        title_key: str = "title",
        tables_key: Optional[str] = None,
        table_insertion_method: str = "append"
) -> int:
    """
    Принимает JSON-файл со статьями, разбивает текст на чанки и дописывает их
    в конец единого JSON-файла (массива объектов).

    Каждый чанк сохраняется в формате:
    {
        "text": str,                # текст чанка
        "title": str,               # название статьи
        "file_chunk_index": int,    # порядковый номер чанка внутри добавляемого файла (начиная с 1)
        "global_chunk_index": int   # сквозной номер во всём выходном файле (начиная с 1)
    }

    Параметры:
        input_file_path: путь к исходному JSON-файлу.
        output_json_path: путь к итоговому JSON-файлу (будет создан или перезаписан).
        chunk_size: размер чанка в символах.
        overlap: перекрытие между соседними чанками.
        text_key: ключ поля с текстом в статье.
        title_key: ключ поля с названием статьи.
        tables_key: ключ поля с таблицами. Если None — таблицы не обрабатываются.
        table_insertion_method: способ вставки таблиц ('append' или 'replace_markers').

    Возвращает:
        Количество добавленных чанков (общее число после дописывания).
    """
    # 1. Загружаем существующий массив (если файл есть)
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                existing_chunks = json.load(f)
            if not isinstance(existing_chunks, list):
                # Если файл повреждён или не массив, начинаем с чистого
                existing_chunks = []
        except (json.JSONDecodeError, FileNotFoundError):
            existing_chunks = []
    else:
        existing_chunks = []

    # Определяем стартовый глобальный индекс
    global_index_start = len(existing_chunks) + 1

    # 2. Загружаем и нормализуем входные данные
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        articles = data
    elif isinstance(data, dict) and "articles" in data:
        articles = data["articles"]
    else:
        articles = [data]

    # 3. Функция разбиения текста на чанки с перекрытием
    def chunk_text(text: str, size: int, ovlp: int):
        if not text or size <= 0:
            return []
        step = size - ovlp
        if step <= 0:
            raise ValueError("overlap должен быть меньше chunk_size")
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start += step
        return chunks

    # 4. Обработка таблиц (опционально)
    # Для простоты здесь предполагается, что функция format_table_as_markdown
    # определена где-то в проекте. Можно вставить её код прямо сюда,
    # если нужна полная автономность.
    def format_table_as_markdown(table: dict) -> str:
        """Простая конвертация таблицы в Markdown."""
        caption = table.get("caption", "")
        if "markdown" in table:
            md = table["markdown"]
        elif "data" in table:
            rows = table["data"]
            if not rows:
                return ""
            header = "| " + " | ".join(str(cell) for cell in rows[0]) + " |"
            sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            body = "\n".join(
                "| " + " | ".join(str(cell) for cell in row) + " |"
                for row in rows[1:]
            )
            md = f"{header}\n{sep}\n{body}"
        else:
            return ""
        if caption:
            md = f"**{caption}**\n\n{md}"
        return md.strip()

    # 5. Чанкинг и формирование записей
    new_chunks = []
    file_chunk_index = 1

    for article in articles:
        if isinstance(article, str):
            text = article
            title = "untitled"
        else:
            text = article.get(text_key, "")
            title = article.get(title_key, "untitled")

        # Обработка таблиц (если указан tables_key и есть функция)
        if tables_key and isinstance(article, dict) and tables_key in article:
            tables = article[tables_key]
            if isinstance(tables, list) and tables:
                table_texts = []
                for tbl in tables:
                    tbl_str = format_table_as_markdown(tbl)
                    if tbl_str:
                        table_texts.append(tbl_str)
                if table_texts:
                    if table_insertion_method == "replace_markers":
                        modified = text
                        for tbl in tables:
                            ref = tbl.get("reference_text")
                            tbl_str = format_table_as_markdown(tbl)
                            if ref and tbl_str and ref in modified:
                                modified = modified.replace(ref, f"\n\n{tbl_str}\n\n", 1)
                            elif tbl_str:
                                modified += f"\n\n{tbl_str}"
                        text = modified
                    else:  # append
                        text += "\n\n[ТАБЛИЦЫ]\n" + "\n\n".join(table_texts)

        # Разбиваем
        chunks = chunk_text(text, chunk_size, overlap)
        for chunk_text_val in chunks:
            new_chunks.append({
                "text": chunk_text_val,
                "title": title,
                "file_chunk_index": file_chunk_index,
                "global_chunk_index": global_index_start
            })
            file_chunk_index += 1
            global_index_start += 1

    # 6. Объединяем с существующими и записываем
    existing_chunks.extend(new_chunks)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(existing_chunks, f, ensure_ascii=False, indent=2)

    return len(existing_chunks)
