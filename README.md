# LLM Doctor Assistant

Проект состоит из двух основных частей:

- веб-чат с LLM-ассистентом;
- локальная сборка базы документов из PDF: парсинг, чанкинг, FAISS-индекс.

## Установка

```bash
pip install -r requirements.txt
```

## Переменные окружения

Создайте `.env` в корне проекта:

```env
API_KEY=your_api_key
BASE_URL=https://ai.api.cloud.yandex.net/v1
MODEL=yandexgpt/rc
FOLDER_ID=your_folder_id
MD_FILE=line_notes.md
```

Для другого OpenAI-compatible API поменяйте `BASE_URL` и `MODEL`.

## Запуск сайта

```bash
python main.py
```

Сайт откроется на:

```text
http://localhost:8000
```

## Сборка локальной базы документов

Положите PDF-файлы в:

```text
db/assets/
```

Запустите:

```bash
python db/db_builder.py
```

Pipeline:

```text
db/assets/*.pdf
  -> db/parse_pdf.py
  -> db/output/*.json
  -> db/chunking/chunker.py
  -> db/all_chunks.json
  -> db/faiss_index.bin + db/chunks_metadata.json
```

## Актуальная структура

```text
AI/md_agent.py          # LLM-агент
main.py                 # веб-сервер
frontend/               # интерфейс чата
db/assets/              # исходные PDF
db/output/              # parsed JSON
db/parse_pdf.py         # PDF -> JSON
db/chunking/chunker.py  # JSON -> chunks
db/db_builder.py        # запуск сборки базы
```
