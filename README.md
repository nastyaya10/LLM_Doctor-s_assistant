# LLM Doctor Assistant

Runtime-ветка для запуска сайта с уже готовой RAG-базой.

Эта ветка не содержит исходные PDF, parsed JSON, промежуточные индексы и
скрипты пересборки базы с нуля. Готовые файлы базы нужно передать на сервер
отдельно.

## Что есть в ветке

```text
src/              # backend и RAG runtime
webui/            # интерфейс сайта
requirements.txt  # зависимости для запуска сайта
```

## Файлы базы

Минимально нужны:

```text
data/chunked/all_chunks.json
data/chunked/faiss_index.bin
```

`data/chunked/chunks_metadata.json` необязателен: если файла нет, приложение
использует `all_chunks.json` как metadata.

Файлы базы не хранятся в Git в этой ветке.

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

RAG_FAISS_INDEX_PATH=data/chunked/faiss_index.bin
RAG_CHUNKS_PATH=data/chunked/all_chunks.json
RAG_METADATA_PATH=data/chunked/chunks_metadata.json
RAG_NEIGHBOR_RADIUS=1
RAG_MAX_CONTEXT_CHARS=30000
RAG_USE_RERANKER=1

HOST=0.0.0.0
PORT=8000
OPEN_BROWSER=0
```

## Запуск сайта

```bash
python src/api/main.py
```

Сайт будет доступен на:

```text
http://localhost:8000
```

Если база лежит не в `data/chunked/`, задайте абсолютные пути:

```env
RAG_FAISS_INDEX_PATH=/path/to/faiss_index.bin
RAG_CHUNKS_PATH=/path/to/all_chunks.json
RAG_METADATA_PATH=/path/to/chunks_metadata.json
```
