# Embedding Retrieval Experiment for Russian Clinical RAG

Минимальный экспериментальный проект для сравнения embedding-моделей в задаче semantic retrieval на искусственном toy-наборе русскоязычных клинических chunks.

Важно: данные в `data/chunks.json` и `data/queries.json` полностью искусственные. В проекте нет настоящих медицинских данных пациентов.

## Что сравнивается

- `BAAI/bge-m3`
- `microsoft/biogpt`
- `intfloat/multilingual-e5-large`
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

Для `microsoft/biogpt` используется простой mean pooling через `transformers`, потому что BioGPT не является sentence-transformers embedding-моделью. Это полезно как экспериментальный baseline, но результаты стоит интерпретировать осторожно.

Опционально можно включить reranking:

- retriever получает `top-10` по cosine similarity;
- `BAAI/bge-reranker-v2-m3` пересортировывает кандидатов;
- метрики считаются до и после reranking.

## Структура

```text
.
├── data/
│   ├── chunks.json
│   └── queries.json
├── run_experiment.py
├── requirements.txt
└── README.md
```

## Установка

Рекомендуется использовать виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Первый запуск скачает модели с Hugging Face. Для крупных моделей может понадобиться несколько гигабайт свободного места и достаточно оперативной памяти.

Если `faiss-cpu` не устанавливается на вашей системе, его можно удалить из `requirements.txt`: текущая реализация использует cosine similarity через `numpy`.

## Запуск

Запустить все embedding-модели без reranking:

```bash
python run_experiment.py
```

Запустить с reranking:

```bash
python run_experiment.py --rerank
```

Запустить одну модель:

```bash
python run_experiment.py --model BAAI/bge-m3
```

Запустить одну модель с reranking:

```bash
python run_experiment.py --model BAAI/bge-m3 --rerank
```

## Результаты

После запуска создаются файлы:

- `results.csv` — метрики по каждой модели, стадии (`embedding`, `reranked`) и типу запроса;
- `errors.csv` — запросы, где правильный chunk не попал в `top-3`.

Метрики:

- `recall_at_1`
- `recall_at_3`
- `mrr`
- `avg_query_time_sec`

В строке `query_type=all` показаны агрегированные метрики по всем запросам. Остальные строки помогают сравнить устойчивость retriever на коротких, длинных, разговорных, терминологических и перефразированных запросах.

## Формат данных

`data/chunks.json`:

```json
[
  {
    "id": "hypertension_1",
    "title": "Артериальная гипертензия",
    "text": "..."
  }
]
```

`data/queries.json`:

```json
[
  {
    "query": "что делать при высоком давлении",
    "expected_chunk_id": "hypertension_1",
    "query_type": "conversational"
  }
]
```

## Как заменить toy chunks на реальные chunks из PDF

1. Извлеките текст из PDF.
2. Разбейте документ на chunks удобного размера.
3. Сохраните chunks в `data/chunks.json` в том же формате: `id`, `title`, `text`.
4. Подготовьте `data/queries.json` с тестовыми запросами и ожидаемым `expected_chunk_id`.
5. Запустите `python run_experiment.py --rerank`.

Для честной оценки лучше делать запросы разных типов: короткие, длинные, разговорные, с аббревиатурами, с медицинскими терминами, с синонимами и перефразированием.
