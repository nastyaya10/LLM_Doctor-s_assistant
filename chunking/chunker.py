"""
Модуль чанкинга медицинских JSON-документов с техникой small-to-large.
Поддерживает таблицы, перекрытие маленьких чанков и сохранение
контекста в формате (текст чанка, название статьи).
"""

import json
from typing import List, Dict, Tuple, Optional, Union


def format_table_as_markdown(table: dict) -> str:
    """
    Преобразует объект таблицы в читаемую Markdown-строку.
    Поддерживает два формата:
    1. 'data': список списков, первая строка — заголовки столбцов.
    2. 'markdown': готовая строка с Markdown-таблицей.
    Также может содержать 'caption' для подписи.
    """
    caption = table.get("caption", "")
    if "markdown" in table and table["markdown"]:
        md = table["markdown"]
    elif "data" in table:
        rows = table["data"]
        if not rows:
            return ""
        # Формируем заголовок и разделитель
        header = "| " + " | ".join(str(cell) for cell in rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
        body = "\n".join(
            "| " + " | ".join(str(cell) for cell in row) + " |"
            for row in rows[1:]
        )
        md = f"{header}\n{separator}\n{body}"
    else:
        return ""

    if caption:
        md = f"**{caption}**\n\n{md}"
    return md.strip()


def chunk_medical_json(
        file_path: str,
        small_chunk_size: int = 400,
        small_overlap: int = 100,
        large_chunk_size: int = 1000,
        text_key: str = "text",
        title_key: str = "title",
        tables_key: Optional[str] = "tables",
        table_insertion_method: str = "append"
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Разбивает медицинский JSON на маленькие (для поиска) и большие (для генерации) чанки.

    Техника small-to-large:
      - Маленькие чанки (small_chunk_size символов, перекрытие small_overlap)
        индексируются и используются для поиска.
      - Большие чанки (large_chunk_size символов) строятся вокруг центра каждого
        маленького и подаются в LLM для генерации ответа.

    Поддерживает таблицы:
      - Если в статье есть поле tables_key (по умолчанию 'tables'), таблицы
        преобразуются в Markdown и вставляются в текст статьи.
      - Метод вставки:
        * 'append' — все таблицы добавляются в конец текста (в секцию [ТАБЛИЦЫ]).
        * 'replace_markers' — ищет в тексте строки из поля 'reference_text'
          каждой таблицы и заменяет их на саму таблицу. Если не найдено,
          таблица добавляется в конец.

    Параметры:
        file_path: путь к JSON-файлу.
        small_chunk_size: размер маленького чанка в символах.
        small_overlap: перекрытие между соседними маленькими чанками (символов).
        large_chunk_size: размер большого контекстного чанка.
        text_key: название поля с основным текстом статьи.
        title_key: название поля с заголовком статьи.
        tables_key: название поля со списком таблиц (или None, если таблиц нет).
        table_insertion_method: 'append' или 'replace_markers'.

    Возвращает:
        small_chunks: список маленьких чанков для индексации.
            Каждый элемент — словарь:
            {
                "id": str,                # уникальный ID маленького чанка
                "text": str,              # текст чанка
                "title": str,             # название статьи
                "large_chunk_id": str     # ID соответствующего большого чанка
            }
        large_chunks: словарь больших чанков для генерации.
            Ключ — large_chunk_id, значение:
            {
                "text": str,              # текст большого чанка (контекст)
                "title": str,             # название статьи
                "small_chunk_ids": [str]  # список ID маленьких чанков,
                                          # ссылающихся на этот большой чанк
            }
    """
    # 1. Загрузка JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Нормализация до списка статей
    if isinstance(data, list):
        articles = data
    elif isinstance(data, dict) and "articles" in data:
        articles = data["articles"]
    else:
        # Одиночный документ
        articles = [data]

    small_chunks: List[Dict] = []
    large_chunks: Dict[str, Dict] = {}
    article_idx = 0

    for article in articles:
        # 3. Извлечение текста и названия
        if isinstance(article, str):
            text = article
            title = f"article_{article_idx}"
        else:
            text = article.get(text_key, "")
            title = article.get(title_key, f"article_{article_idx}")

        # 4. Обработка таблиц (если есть)
        if tables_key and isinstance(article, dict) and tables_key in article:
            tables = article[tables_key]
            if isinstance(tables, list) and len(tables) > 0:
                table_texts = []
                for tbl in tables:
                    tbl_str = format_table_as_markdown(tbl)
                    if tbl_str:
                        table_texts.append(tbl_str)

                if table_texts:
                    if table_insertion_method == "replace_markers":
                        # Пытаемся заменить упоминания таблиц в тексте
                        modified_text = text
                        for tbl in tables:
                            ref = tbl.get("reference_text")
                            tbl_str = format_table_as_markdown(tbl)
                            if ref and tbl_str and ref in modified_text:
                                modified_text = modified_text.replace(
                                    ref, f"\n\n{tbl_str}\n\n", 1
                                )
                            elif tbl_str:
                                # Если не найдено — добавляем в конец
                                modified_text += f"\n\n{tbl_str}"
                        text = modified_text
                    else:  # "append" (по умолчанию)
                        text += "\n\n[ТАБЛИЦЫ]\n" + "\n\n".join(table_texts)

        if not text:
            article_idx += 1
            continue

        # 5. Разбиение на маленькие чанки с перекрытием
        total_len = len(text)
        step = small_chunk_size - small_overlap
        if step <= 0:
            raise ValueError("small_overlap должен быть меньше small_chunk_size")

        small_spans = []  # (start, end) для каждого маленького чанка
        start = 0
        while start < total_len:
            end = min(start + small_chunk_size, total_len)
            small_spans.append((start, end))
            if end == total_len:
                break
            start += step

        # 6. Для каждого маленького чанка строим большой
        for s_start, s_end in small_spans:
            small_text = text[s_start:s_end]

            # Центр маленького чанка -> центр большого окна
            center = (s_start + s_end) // 2
            large_start = max(0, center - large_chunk_size // 2)
            large_end = min(total_len, large_start + large_chunk_size)

            # Если окно меньше желаемого из-за границы документа, сдвигаем начало
            if large_end - large_start < large_chunk_size and large_start > 0:
                large_start = max(0, large_end - large_chunk_size)

            large_text = text[large_start:large_end]

            # ID большого чанка (уникальный по статье и границам)
            large_id = f"{title}_{large_start}_{large_end}"

            # Сохраняем большой чанк, если он ещё не встречался
            if large_id not in large_chunks:
                large_chunks[large_id] = {
                    "text": large_text,
                    "title": title,
                    "small_chunk_ids": []
                }
            large_chunks[large_id]["small_chunk_ids"].append(
                f"{title}_small_{s_start}_{s_end}"
            )

            # Добавляем маленький чанк
            small_chunks.append({
                "id": f"{title}_small_{s_start}_{s_end}",
                "text": small_text,
                "title": title,
                "large_chunk_id": large_id
            })

        article_idx += 1

    return small_chunks, large_chunks


# ========== Пример использования ==========
if __name__ == "__main__":
    # Пример JSON-файла (можно создать для теста)
    sample_data = [
        {
            "title": "Эффективность препарата X при гипертонии",
            "text": "В двойное слепое плацебо-контролируемое исследование были включены 120 пациентов с артериальной гипертензией 1–2 степени. Основной конечной точкой было снижение систолического артериального давления через 12 недель терапии.",
            "tables": [
                {
                    "table_id": "1",
                    "caption": "Таблица 1 — Исходные характеристики групп",
                    "data": [
                        ["Параметр", "Группа препарата X", "Группа плацебо"],
                        ["Возраст, лет", "52.3 ± 9.1", "51.8 ± 8.7"],
                        ["ИМТ, кг/м²", "29.1 ± 3.4", "28.7 ± 3.2"],
                        ["САД исходно, мм рт. ст.", "152.4 ± 10.2", "151.9 ± 9.8"]
                    ]
                },
                {
                    "table_id": "2",
                    "caption": "Таблица 2 — Динамика АД через 12 недель",
                    "data": [
                        ["Показатель", "Группа X", "Плацебо", "p"],
                        ["Δ САД, мм рт. ст.", "-18.5 ± 12.1", "-6.2 ± 11.5", "<0.001"],
                        ["Δ ДАД, мм рт. ст.", "-10.2 ± 8.3", "-3.1 ± 7.9", "<0.001"]
                    ]
                }
            ]
        },
        {
            "title": "Анализ побочных эффектов",
            "text": "Наиболее частыми нежелательными явлениями были головная боль (4.2%), головокружение (2.8%) и сухость во рту (1.9%). Серьёзных нежелательных явлений не зарегистрировано.",
            "tables": []
        }
    ]

    # Запись тестового файла
    with open("sample_medical.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    # Чанкинг
    small, large = chunk_medical_json(
        "sample_medical.json",
        small_chunk_size=300,
        small_overlap=50,
        large_chunk_size=700,
        table_insertion_method="append"
    )

    print("=== МАЛЕНЬКИЕ ЧАНКИ (пример первых трёх) ===")
    for chunk in small[:3]:
        print(f"ID: {chunk['id']}")
        print(f"Title: {chunk['title']}")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Large ID: {chunk['large_chunk_id']}")
        print("------")

    print("\n=== БОЛЬШИЕ ЧАНКИ (пример) ===")
    for lid, ldata in list(large.items())[:1]:
        print(f"ID: {lid}")
        print(f"Title: {ldata['title']}")
        print(f"Text (truncated): {ldata['text'][:150]}...")
        print(f"Linked small IDs: {ldata['small_chunk_ids']}")
