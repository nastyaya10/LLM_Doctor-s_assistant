# Medical RAG Retriever

## Что делает компонент

`MedicalRAGRetriever` отвечает за retrieval-часть RAG-системы: принимает медицинский запрос пользователя, находит релевантные чанки в FAISS-индексе и возвращает их с текстом, score и metadata.

FAISS используется только для vector search. Эмбеддинги запросов строятся отдельно через `SentenceTransformer` и модель `BAAI/bge-m3`.

## Pipeline работы

1. Пользователь передает текстовый запрос.
2. Запрос очищается от лишних пробелов.
3. Создаются три query variants:
   - `original` — исходный запрос;
   - `rewritten` — переписанный запрос с медицинскими терминами;
   - `hyde` — гипотетический медицинский фрагмент документа.
4. Каждый вариант запроса эмбеддится через `BAAI/bge-m3`.
5. Для каждого embedding выполняется поиск в FAISS:

```python
scores, indices = self.faiss_index.search(query_emb, limit)
```

6. По индексам `indices[0]` достаются чанки из `all_chunks.json`.
7. Metadata берется из `chunks_metadata.json`, если этот файл отличается от `all_chunks.json`.
8. Результаты от разных вариантов запроса объединяются.
9. Дубликаты удаляются по FAISS index.
10. Оставляются только chunks со score выше `SIMILARITY_THRESHOLD`.
11. Если включен reranker, найденные chunks пересортировываются через CrossEncoder.
12. Итоговые результаты возвращаются пользователю.

## Входные данные

Основной вход:

```python
retriever.retrieve(query: str)
```

Также компонент использует файлы:

```text
faiss_index.bin
all_chunks.json
chunks_metadata.json
```

## Выходные данные

Метод `retrieve()` возвращает `RetrievalResult`:

```python
status
original_query
rewritten_query
hyde_query
results
debug_info
```

Каждый элемент `results` — это `RetrievedChunk`:

```python
score
rerank_score
text
title
global_chunk_index
file_chunk_index
metadata
faiss_index
query_variants
```

## Основные классы и методы

### `MedicalRAGRetriever`

Главный класс retriever-а. Загружает FAISS index, chunks, metadata, embedding model и выполняет поиск.

Основной метод:

```python
retrieve(query: str) -> RetrievalResult
```

### `QueryTransformer`

Создает дополнительные варианты запроса:

- Rewrite;
- HyDE.

По умолчанию пытается использовать Gemini API через `GEMINI_API_KEY`. Если ключ не задан или Gemini недоступен, автоматически используется локальный rule-based fallback.

### `RetrievedChunk`

Структура одного найденного чанка. Хранит текст, score, title, индексы чанка и metadata.

### `RetrievalResult`

Структура полного результата поиска.

### `_embed_query()`

Строит embedding запроса через `SentenceTransformer("BAAI/bge-m3")`.

### `_search_variant()`

Выполняет FAISS search для одного варианта запроса.

### `_rerank_results()`

Пересортировывает найденные chunks через `CrossEncoder`. На вход подаются пары:

```python
[query, chunk_text]
```

Reranker возвращает отдельный `rerank_score`, который используется для финальной сортировки.

### `_load_faiss_index()`

Загружает FAISS index:

```python
faiss.read_index(str(path))
```

## Rewrite и HyDE

### Rewrite

Rewrite — это переформулировка пользовательского запроса в более медицинский и поисково-полезный вид.

Пример:

```text
часто хочу пить и сахар высокий
```

превращается в:

```text
часто хочу пить и сахар высокий.
Медицинские термины: сахарный диабет, гипергликемия, глюкоза крови, HbA1c, полидипсия, полиурия
```

Rewrite нужен, потому что пользователь может писать бытовым языком, а документы часто используют формальные медицинские термины.

### HyDE

HyDE означает `Hypothetical Document Embeddings`.

Вместо поиска только по вопросу создается короткий гипотетический медицинский фрагмент, похожий на текст документа, который мог бы содержать ответ. Этот фрагмент тоже эмбеддится и ищется в FAISS.

HyDE помогает находить документы по смыслу предполагаемого ответа, особенно если исходный запрос короткий или разговорный.

### Где используются Rewrite и HyDE

В методе `retrieve()`:

```python
rewritten_query = self.query_transformer.rewrite_query(normalized_query)
hyde_query = self.query_transformer.generate_hyde(normalized_query)
```

Затем все варианты идут в retrieval:

```python
query_variants = {
    "original": normalized_query,
    "rewritten": rewritten_query,
    "hyde": hyde_query,
}
```

### Чем отличаются

```text
Rewrite = улучшить сам вопрос
HyDE = создать псевдо-документ, похожий на возможный ответ
```

В текущем коде оба метода сначала пробуют Gemini API (`gemini-2.5-flash`). Если Gemini недоступен, используется rule-based реализация через регулярные выражения и словарь медицинских терминов.

## Embedding, Retrieval, Vector DB и Reranking

### Embedding

Embedding — числовое представление смысла текста. В коде используется модель:

```text
BAAI/bge-m3
```

### Vector DB / FAISS

FAISS хранит векторы чанков и быстро ищет ближайшие к query embedding. В этом компоненте FAISS не создает embeddings, а только выполняет поиск.

### Retrieval

Retrieval — это поиск релевантных чанков по пользовательскому запросу. В этом коде он включает Rewrite, HyDE, embedding, FAISS search, фильтрацию по threshold и сортировку.

### Reranking

Reranking в этом компоненте реализован через:

```text
BAAI/bge-reranker-v2-m3
```

FAISS сначала быстро находит кандидатов по embedding similarity. Затем reranker точнее оценивает пары `query + chunk` и пересортировывает найденные chunks.

В результате у chunk есть два score:

```text
score         = FAISS/vector-search score
rerank_score = CrossEncoder/reranker score
```

Итоговая сортировка выполняется по `rerank_score`, если reranker включен.

## Как встроить в проект

Создать retriever один раз при старте приложения:

```python
from medical_rag_retriever import MedicalRAGRetriever

retriever = MedicalRAGRetriever()
```

Reranker включен по умолчанию в консольном запуске. В коде его можно включить или отключить явно:

```python
retriever = MedicalRAGRetriever(use_reranker=True)
```

или:

```python
retriever = MedicalRAGRetriever(use_reranker=False)
```

На каждый пользовательский вопрос вызывать:

```python
result = retriever.retrieve(user_query)
```

Собрать context для LLM:

```python
context = "\n\n".join(
    f"[{i}] {chunk.title}\n{chunk.text}"
    for i, chunk in enumerate(result.results, start=1)
)
```

Использовать context в prompt:

```python
prompt = f"""
Ответь на вопрос пользователя только на основе контекста.

Контекст:
{context}

Вопрос:
{user_query}
"""
```

Важно: `MedicalRAGRetriever()` не нужно создавать на каждый запрос, потому что он загружает FAISS index и embedding model. Лучше держать один экземпляр retriever-а как singleton или глобальный объект приложения.

## Консольный запуск

Для работы Gemini Rewrite/HyDE нужно указать ключ Google AI Studio:

```bash
export GEMINI_API_KEY="your_api_key"
```

Можно также положить ключ в `.env`:

```text
GEMINI_API_KEY=your_api_key
```

Файл `.env` должен лежать в корне проекта рядом с `medical_rag_retriever.py`.
Пример формата есть в `.env.example`. Сам `.env` добавлен в `.gitignore`.

Код также поддерживает файл `.env.py`, если ключ уже был сохранен под таким именем. Но предпочтительный вариант — обычный `.env`.

Запуск интерактивного режима:

```bash
python medical_rag_retriever.py
```

После запуска программа один раз загрузит FAISS index и `BAAI/bge-m3`, затем будет ждать запросы из консоли:

```text
query> часто хочу пить и сахар высокий что это может быть
```

Для выхода:

```text
exit
```

В debug-строке видно, что реально сработало для Rewrite и HyDE:

```text
rewrite=gemini, hyde=gemini, gemini_model=gemini-2.5-flash
```

Если Gemini недоступен, будет:

```text
rewrite=fallback, hyde=fallback, gemini_error=...
```

Если `gemini-2.5-flash` временно перегружен и возвращает `503 UNAVAILABLE`, код делает retry и пробует fallback-модель `gemini-2.0-flash`. Модель можно переопределить через:

```text
GEMINI_MODEL=gemini-2.0-flash
```
