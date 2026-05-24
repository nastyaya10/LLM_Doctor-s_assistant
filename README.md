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

STT_API_KEY=your_openai_or_compatible_stt_key
STT_BASE_URL=https://api.openai.com/v1
STT_MODEL=whisper-1
# STT_PROVIDER=openai
# STT_PROVIDER=yandex
# STT_LANG=ru-RU
# STT_SAMPLE_RATE=16000

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

Для другого OpenAI-compatible API поменяйте `BASE_URL` и `MODEL`.

Голосовой ввод использует отдельный OpenAI-compatible endpoint для speech-to-text.
Если `STT_API_KEY` не задан, сервер попробует использовать `OPENAI_API_KEY` или
общий `API_KEY`; если `STT_BASE_URL` не задан, будет использован
`OPENAI_BASE_URL`.

При конфигурации Yandex (`BASE_URL=https://ai.api.cloud.yandex.net/v1` и заданные
`API_KEY`/`FOLDER_ID`) голосовой ввод автоматически переключается на Yandex
SpeechKit. Для принудительного выбора провайдера используйте
`STT_PROVIDER=openai` или `STT_PROVIDER=yandex`.

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
