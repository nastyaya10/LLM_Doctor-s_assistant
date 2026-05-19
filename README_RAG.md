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
11. Результаты сортируются по `score` по убыванию.

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

В текущей версии работает локально, без OpenAI и без API-ключей.

### `RetrievedChunk`

Структура одного найденного чанка. Хранит текст, score, title, индексы чанка и metadata.

### `RetrievalResult`

Структура полного результата поиска.

### `_embed_query()`

Строит embedding запроса через `SentenceTransformer("BAAI/bge-m3")`.

### `_search_variant()`

Выполняет FAISS search для одного варианта запроса.

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

В текущем коде оба метода реализованы rule-based: через регулярные выражения и словарь медицинских терминов. Внешние API-ключи не используются.

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

Reranking в этом компоненте не реализован. Сейчас результаты сортируются по score из FAISS. При необходимости reranker можно добавить отдельным этапом после `retrieve()`.

## Как встроить в проект

Создать retriever один раз при старте приложения:

```python
from medical_rag_retriever import MedicalRAGRetriever

retriever = MedicalRAGRetriever()
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

