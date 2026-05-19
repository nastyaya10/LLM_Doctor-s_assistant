# LLM Doctor Assistant

Проект состоит из двух основных частей:

- веб-чат с LLM-ассистентом;
- готовая локальная RAG-база документов: чанки JSON + FAISS-индекс.

В репозитории хранятся только файлы, нужные для запуска сайта:

```text
db/all_chunks.json
db/faiss_index.bin
db/chunks_metadata.json
```

PDF-файлы и промежуточные JSON из `db/assets/` и `db/output/` считаются локальными
файлами сборки базы и не нужны для деплоя.

## Установка

```bash
pip install -r requirements.txt
```

Зависимости для парсинга PDF и пересборки базы вынесены отдельно:

```bash
pip install -r requirements-db-build.txt
```

## Переменные окружения

Создайте `.env` в корне проекта:

```env
API_KEY=your_api_key
BASE_URL=https://ai.api.cloud.yandex.net/v1
MODEL=yandexgpt/rc
FOLDER_ID=your_folder_id
```

Для другого OpenAI-compatible API поменяйте `BASE_URL` и `MODEL`.

Настройки путей для RAG-ретривера, если база лежит не в стандартной папке `db/`:

```env
RAG_DB_DIR=db
RAG_FAISS_INDEX_PATH=db/faiss_index.bin
RAG_CHUNKS_PATH=db/all_chunks.json
RAG_METADATA_PATH=db/chunks_metadata.json
RAG_NEIGHBOR_RADIUS=1
RAG_MAX_CONTEXT_CHARS=30000
RAG_USE_RERANKER=1
LOG_RERANKER_IO=1
LOG_RAG_CONTEXT=1
HOST=0.0.0.0
PORT=8000
OPEN_BROWSER=0
```

Ретривер использует те же настройки Yandex/OpenAI-compatible LLM, что и агент:
`API_KEY`, `BASE_URL`, `MODEL`, `FOLDER_ID`. Если `API_KEY` не задан, Rewrite/HyDE
автоматически работают через локальный fallback.

Для Docker обычно достаточно примонтировать серверную папку с базой в `/app/db`
и задать:

```env
RAG_DB_DIR=/app/db
```

## Запуск сайта

```bash
python main.py
```

Сайт откроется на:

```text
http://localhost:8000
```

На сервере приложение слушает `HOST` и `PORT` из `.env`.

## Сборка локальной базы документов

Положите PDF-файлы в:

```text
db/assets/
```

Запустите:

```bash
python db/db_builder.py
```

Перед этим установите зависимости для сборки:

```bash
pip install -r requirements-db-build.txt
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
