from __future__ import annotations

import json
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from retriever.paths import (
        PROJECT_ROOT,
        DEFAULT_CHUNKS_PATH,
        DEFAULT_FAISS_INDEX_PATH,
        DEFAULT_METADATA_PATH,
    )
except ImportError:
    from paths import (
        PROJECT_ROOT,
        DEFAULT_CHUNKS_PATH,
        DEFAULT_FAISS_INDEX_PATH,
        DEFAULT_METADATA_PATH,
    )

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_K = 5

# Explicit retrieval cutoff. With a normalized BGE-M3 embedding + IndexFlatIP
# this is cosine-like similarity. Tune it on your validation queries.
SIMILARITY_THRESHOLD = 0.55
DEFAULT_SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD

DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"
LEGACY_DOTENV_PATH = PROJECT_ROOT / ".env.py"


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized_value = value.strip().strip("\"'").lower()
    return normalized_value in ("1", "true", "yes", "on")


def load_project_dotenv() -> bool:
    """Load local settings from the project env file."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    for dotenv_path in (DEFAULT_DOTENV_PATH, LEGACY_DOTENV_PATH):
        if dotenv_path.exists():
            return bool(load_dotenv(dotenv_path=dotenv_path, override=False))

    return bool(load_dotenv())


@dataclass
class QueryTransformer:
    """
    Query rewrite and HyDE generator.

    By default it uses the same Yandex/OpenAI-compatible LLM settings as the
    agent. If the LLM is unavailable, it falls back to deterministic medical
    term expansion.
    """

    rewrite_fn: Callable[[str], str] | None = None
    hyde_fn: Callable[[str], str] | None = None
    use_llm: bool = True
    api_key_env: str = "API_KEY"
    folder_id_env: str = "FOLDER_ID"
    base_url_env: str = "BASE_URL"
    model_env: str = "MODEL"
    default_base_url: str = "https://ai.api.cloud.yandex.net/v1"
    default_model: str = "yandexgpt/rc"
    max_retries: int = 2
    last_rewrite_source: str = field(default="not_run", init=False)
    last_hyde_source: str = field(default="not_run", init=False)
    last_llm_error: str | None = field(default=None, init=False)
    last_llm_model: str | None = field(default=None, init=False)

    def rewrite_query(self, query: str) -> str:
        if self.rewrite_fn is not None:
            self.last_rewrite_source = "custom"
            return self.rewrite_fn(query).strip()

        if self.use_llm:
            rewritten_query = self._try_llm_rewrite(query)
            if rewritten_query:
                self.last_rewrite_source = "yandex_llm"
                return rewritten_query

        self.last_rewrite_source = "fallback"
        return self._rule_based_rewrite(query)

    def generate_hyde(self, query: str) -> str:
        if self.hyde_fn is not None:
            self.last_hyde_source = "custom"
            return self.hyde_fn(query).strip()

        if self.use_llm:
            hyde_query = self._try_llm_hyde(query)
            if hyde_query:
                self.last_hyde_source = "yandex_llm"
                return hyde_query

        self.last_hyde_source = "fallback"
        rewritten_query = self._rule_based_rewrite(query)
        key_terms = self._expanded_terms(query)
        return (
            "Гипотетический фрагмент медицинского документа. "
            f"Запрос пациента: {query.strip()}. "
            f"Нормализованная формулировка: {rewritten_query}. "
            f"Ключевые клинические понятия: {', '.join(key_terms) or 'симптомы, диагноз, обследование, лечение'}. "
            "В документе могут описываться жалобы, анамнез, факторы риска, "
            "дифференциальная диагностика, лабораторные и инструментальные "
            "исследования, показания к консультации специалиста, лечение, "
            "профилактика и маршрутизация пациента."
        )

    def _try_llm_rewrite(self, query: str) -> str | None:
        prompt = f"""
        Ты медицинский query-rewriter для RAG по русскоязычным клиническим
        документам.

        Задача:
        - перепиши запрос пользователя в одну короткую поисковую формулировку;
        - добавь медицинские синонимы, расшифровки аббревиатур и ключевые термины;
        - не отвечай на вопрос пользователя;
        - не добавляй markdown, списки, кавычки и пояснения.

        Запрос пользователя:
        {query.strip()}
        """
        return self._call_llm(prompt)

    def _try_llm_hyde(self, query: str) -> str | None:
        prompt = f"""
        Ты генерируешь HyDE pseudo-document для medical RAG.

        Напиши один короткий абзац на русском языке, похожий на фрагмент
        клинических рекомендаций или медицинского документа, который мог бы
        содержать ответ на запрос пользователя.

        Требования:
        - упомяни релевантные симптомы, диагнозы, обследования, лечение и
          маршрутизацию пациента;
        - не давай окончательный медицинский совет;
        - не добавляй markdown, списки и дисклеймеры;
        - текст должен быть полезен именно для embedding search.

        Запрос пользователя:
        {query.strip()}
        """
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str | None:
        self._load_dotenv_if_available()
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            self.last_llm_error = f"{self.api_key_env} is not set"
            return None

        try:
            from openai import OpenAI
        except Exception as error:
            self.last_llm_error = f"{type(error).__name__}: {error}"
            return None

        prompt_text = textwrap.dedent(prompt).strip()
        folder_id = os.getenv(self.folder_id_env)
        base_url = os.getenv(self.base_url_env, self.default_base_url)
        model_name = self._model_name(folder_id)
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            project=folder_id,
            timeout=60.0,
            max_retries=0,
        )
        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.0,
                    max_tokens=350,
                )
                text = response.choices[0].message.content
                if not text:
                    last_error = f"{model_name}: LLM returned empty text"
                    break

                self.last_llm_error = None
                self.last_llm_model = model_name
                return self._clean_generated_text(text)
            except Exception as error:
                last_error = f"{model_name}: {type(error).__name__}: {error}"
                if attempt < self.max_retries:
                    time.sleep(0.7 * (attempt + 1))

        self.last_llm_error = last_error
        self.last_llm_model = None
        return None

    def _model_name(self, folder_id: str | None) -> str:
        model_name = os.getenv(self.model_env, self.default_model)
        if model_name.startswith("gpt://"):
            return model_name
        if folder_id:
            return f"gpt://{folder_id}/{model_name}"
        return model_name

    @staticmethod
    def _load_dotenv_if_available() -> None:
        load_project_dotenv()

    @classmethod
    def _rule_based_rewrite(cls, query: str) -> str:
        normalized_query = cls._normalize_text(query)
        expanded_terms = cls._expanded_terms(normalized_query)

        if not expanded_terms:
            return normalized_query

        return f"{normalized_query}. Медицинские термины: {', '.join(expanded_terms)}"

    @classmethod
    def _expanded_terms(cls, query: str) -> list[str]:
        query_lower = query.lower()
        expansions: list[str] = []

        rules = {
            r"\bад\b|давлен|гипертенз|гипертони": [
                "артериальная гипертензия",
                "повышение артериального давления",
                "поражение органов-мишеней",
            ],
            r"сахар|глюкоз|диабет|пить|жажд|мочеиспуск": [
                "сахарный диабет",
                "гипергликемия",
                "глюкоза крови",
                "HbA1c",
                "полидипсия",
                "полиурия",
            ],
            r"одыш|каш|мокрот|хобл|кури|спирометр": [
                "хроническая обструктивная болезнь легких",
                "ХОБЛ",
                "бронходилататоры",
                "спирометрия",
                "отказ от курения",
            ],
            r"астм|свист|удуш|ингаляц|бронхоспаз": [
                "бронхиальная астма",
                "бронхиальная обструкция",
                "ингаляционные глюкокортикостероиды",
                "бронхолитики",
            ],
            r"инсульт|реч|лиц|слабост|онемен|парез|тиа": [
                "острое нарушение мозгового кровообращения",
                "инсульт",
                "транзиторная ишемическая атака",
                "неврологический дефицит",
            ],
            r"грудин|стенокард|ишеми|инфаркт|нагрузк|покой": [
                "ишемическая болезнь сердца",
                "стенокардия",
                "боль за грудиной",
                "электрокардиография",
            ],
            r"скф|альбуминур|креатинин|почеч|почек|хбп": [
                "хроническая болезнь почек",
                "снижение скорости клубочковой фильтрации",
                "альбуминурия",
                "креатинин",
            ],
            r"лихорад|температур|пневмон|рентген|боль в груди": [
                "пневмония",
                "внебольничная пневмония",
                "рентгенография органов грудной клетки",
                "антибактериальная терапия",
            ],
            r"головн|мигрен|аур|светобояз|тошнот": [
                "мигрень",
                "аура",
                "фотофобия",
                "профилактическая терапия",
            ],
            r"желез|анеми|ферритин|гемоглобин": [
                "железодефицитная анемия",
                "ферритин",
                "гемоглобин",
                "препараты железа",
            ],
        }

        for pattern, terms in rules.items():
            if re.search(pattern, query_lower):
                expansions.extend(terms)

        return list(dict.fromkeys(expansions))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _clean_generated_text(cls, text: str) -> str:
        cleaned = cls._normalize_text(text)
        cleaned = cleaned.strip("\"'` ")
        return cleaned[:1200]



@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    text: str
    title: str
    global_chunk_index: int | None
    file_chunk_index: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    faiss_index: int | None = None
    rerank_score: float | None = None
    query_variants: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalDebugInfo:
    raw_results_by_variant: dict[str, int]
    merged_results_count: int
    results_after_threshold: int
    top_k: int
    similarity_threshold: float
    model_name: str
    faiss_index_path: str
    chunks_path: str
    metadata_path: str | None
    metadata_source: str
    reranker_model_name: str | None
    reranking_enabled: bool
    rewrite_source: str
    hyde_source: str
    llm_error: str | None
    llm_model: str | None


@dataclass(frozen=True)
class RetrievalResult:
    status: str
    original_query: str
    rewritten_query: str
    hyde_query: str
    results: list[RetrievedChunk]
    debug_info: RetrievalDebugInfo


class MedicalRAGRetriever:
    """
    FAISS-backed retriever for the medical RAG system.

    Responsibilities are deliberately separated:
    - SentenceTransformer(BAAI/bge-m3) embeds queries.
    - FAISS is used only for vector search via faiss.search(...).
    - all_chunks.json stores chunk text and primary chunk fields.
    - chunks_metadata.json is loaded as metadata when it differs from chunks.
    """

    ORIGINAL_VARIANT = "original"
    REWRITTEN_VARIANT = "rewritten"
    HYDE_VARIANT = "hyde"

    def __init__(
        self,
        faiss_index_path: str | Path = DEFAULT_FAISS_INDEX_PATH,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
        metadata_path: str | Path | None = DEFAULT_METADATA_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        query_transformer: QueryTransformer | None = None,
        normalize_query_embeddings: bool = True,
        use_reranker: bool | None = None,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL_NAME,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        load_project_dotenv()

        self.faiss_index_path = Path(faiss_index_path)
        self.chunks_path = Path(chunks_path)
        self.metadata_path = Path(metadata_path) if metadata_path is not None else None
        self.model_name = model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.query_transformer = query_transformer or QueryTransformer()
        self.normalize_query_embeddings = normalize_query_embeddings
        self.use_reranker = (
            env_bool("RAG_USE_RERANKER", False) if use_reranker is None else use_reranker
        )
        self.reranker_model_name = reranker_model_name
        self.log_reranker_io = env_bool("LOG_RERANKER_IO", True)

        self._log_reranker_config()

        self.chunks = self._load_json_list(self.chunks_path, "chunks")
        self.metadata_items, self.metadata_source = self._load_metadata_items()
        self.faiss_index = self._load_faiss_index(self.faiss_index_path)
        self.embedder = None
        print(
            f"[RAG] Embedding-модель будет загружена лениво перед первым поиском: "
            f"{self.model_name}"
        )
        self.reranker = None
        if self.use_reranker and self.log_reranker_io:
            print(
                "[Reranker] Реранкер включен, но модель будет загружена лениво "
                "только перед первым реальным реранкингом."
            )

        if self.faiss_index.ntotal > len(self.chunks):
            raise ValueError(
                "FAISS index contains more vectors than all_chunks.json contains chunks: "
                f"{self.faiss_index.ntotal} > {len(self.chunks)}."
            )

    def retrieve(self, query: str) -> RetrievalResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty.")

        rewritten_query = self.query_transformer.rewrite_query(normalized_query)
        hyde_query = self.query_transformer.generate_hyde(normalized_query)

        query_variants = {
            self.ORIGINAL_VARIANT: normalized_query,
            self.REWRITTEN_VARIANT: rewritten_query,
            self.HYDE_VARIANT: hyde_query,
        }

        raw_results_by_variant: dict[str, int] = {}
        merged_chunks: dict[int, RetrievedChunk] = {}

        for variant_name, variant_query in query_variants.items():
            raw_results = self._search_variant(variant_query, self.top_k, variant_name)
            raw_results_by_variant[variant_name] = len(raw_results)

            for chunk in raw_results:
                if chunk.faiss_index is None:
                    continue

                existing_chunk = merged_chunks.get(chunk.faiss_index)
                if existing_chunk is None:
                    merged_chunks[chunk.faiss_index] = chunk
                else:
                    merged_chunks[chunk.faiss_index] = self._merge_duplicate_chunks(
                        existing_chunk,
                        chunk,
                    )

        filtered_results = [
            chunk
            for chunk in merged_chunks.values()
            if chunk.score >= self.similarity_threshold
        ]
        filtered_results.sort(key=lambda chunk: chunk.score, reverse=True)
        self._log_reranker_decision(len(merged_chunks), len(filtered_results))
        if self.use_reranker and filtered_results:
            reranker = self._get_reranker()
            filtered_results = self._rerank_results(
                normalized_query,
                filtered_results,
                reranker,
            )

        debug_info = RetrievalDebugInfo(
            raw_results_by_variant=raw_results_by_variant,
            merged_results_count=len(merged_chunks),
            results_after_threshold=len(filtered_results),
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            model_name=self.model_name,
            faiss_index_path=str(self.faiss_index_path),
            chunks_path=str(self.chunks_path),
            metadata_path=str(self.metadata_path) if self.metadata_path else None,
            metadata_source=self.metadata_source,
            reranker_model_name=self.reranker_model_name if self.use_reranker else None,
            reranking_enabled=self.use_reranker,
            rewrite_source=self.query_transformer.last_rewrite_source,
            hyde_source=self.query_transformer.last_hyde_source,
            llm_error=self.query_transformer.last_llm_error,
            llm_model=self.query_transformer.last_llm_model,
        )

        return RetrievalResult(
            status="ok" if filtered_results else "no_relevant_chunks",
            original_query=normalized_query,
            rewritten_query=rewritten_query,
            hyde_query=hyde_query,
            results=filtered_results,
            debug_info=debug_info,
        )

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Compatibility helper returning plain dicts for callers that expect
        index.search(query, top_k). It searches only the original query.
        """
        limit = top_k if top_k is not None else self.top_k
        chunks = self._search_variant(query.strip(), limit, self.ORIGINAL_VARIANT)
        return [
            {
                "score": chunk.score,
                "text": chunk.text,
                "title": chunk.title,
                "global_chunk_index": chunk.global_chunk_index,
                "file_chunk_index": chunk.file_chunk_index,
                "rerank_score": chunk.rerank_score,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
            if chunk.score >= self.similarity_threshold
        ]

    def _search_variant(
        self,
        query: str,
        top_k: int,
        variant_name: str,
    ) -> list[RetrievedChunk]:
        if not query:
            return []

        query_emb = self._embed_query(query)
        limit = min(top_k, self.faiss_index.ntotal, len(self.chunks))

        # FAISS is used only here, for vector search.
        scores, indices = self.faiss_index.search(query_emb, limit)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue

            chunk = self.chunks[idx]
            metadata = self._metadata_for_index(idx)
            merged_metadata = {**chunk, **metadata}

            results.append(
                RetrievedChunk(
                    score=float(score),
                    text=str(chunk.get("text", "")),
                    title=str(merged_metadata.get("title", "")),
                    global_chunk_index=self._optional_int(
                        merged_metadata.get("global_chunk_index")
                    ),
                    file_chunk_index=self._optional_int(
                        merged_metadata.get("file_chunk_index")
                    ),
                    metadata=merged_metadata,
                    faiss_index=int(idx),
                    query_variants=[variant_name],
                )
            )

        return results

    def _embed_query(self, query: str) -> Any:
        try:
            import numpy as np
        except ImportError as error:
            raise ImportError(
                "numpy is not installed. Install dependencies from `requirements.txt` "
                "before running retrieval."
            ) from error

        embedder = self._get_embedder()
        embedding = embedder.encode(
            [query],
            normalize_embeddings=self.normalize_query_embeddings,
        )
        return np.asarray(embedding, dtype=np.float32)

    @staticmethod
    def _load_faiss_index(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {path}")

        try:
            import faiss
        except ImportError as error:
            raise ImportError(
                "faiss-cpu is not installed. Install it with `pip install faiss-cpu`."
            ) from error

        return faiss.read_index(str(path))

    @staticmethod
    def _load_embedder(model_name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "sentence-transformers is not installed. Install dependencies from "
                "`requirements.txt` before running retrieval."
            ) from error

        return SentenceTransformer(model_name)

    def _get_embedder(self) -> Any:
        if self.embedder is None:
            print(f"[RAG] Загружаем embedding-модель: {self.model_name}")
            start_time = time.time()
            self.embedder = self._load_embedder(self.model_name)
            elapsed = time.time() - start_time
            print(f"[RAG] Embedding-модель загружена за {elapsed:.2f} сек: {self.model_name}")

        return self.embedder

    @staticmethod
    def _load_reranker(model_name: str) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise ImportError(
                "sentence-transformers is not installed. Install dependencies from "
                "`requirements.txt` before running reranking."
            ) from error

        return CrossEncoder(model_name)

    def _get_reranker(self) -> Any:
        if self.reranker is None:
            if self.log_reranker_io:
                print(f"[Reranker] Загружаем модель реранкера: {self.reranker_model_name}")
            start_time = time.time()
            self.reranker = self._load_reranker(self.reranker_model_name)
            if self.log_reranker_io:
                elapsed = time.time() - start_time
                print(
                    f"[Reranker] Модель реранкера загружена за {elapsed:.2f} сек: "
                    f"{self.reranker_model_name}"
                )

        return self.reranker

    def _log_reranker_config(self) -> None:
        if not self.log_reranker_io:
            return

        print(
            "[Reranker] Конфигурация: "
            f"RAG_USE_RERANKER={os.getenv('RAG_USE_RERANKER')!r}, "
            f"enabled={self.use_reranker}, "
            f"LOG_RERANKER_IO={os.getenv('LOG_RERANKER_IO')!r}, "
            f"model={self.reranker_model_name}"
        )

    def _log_reranker_decision(
        self,
        merged_results_count: int,
        filtered_results_count: int,
    ) -> None:
        if not self.log_reranker_io:
            return

        print("\n[Reranker] Проверка перед реранкингом")
        print(
            "[Reranker] Пояснение: до модели реранкера доходят только кандидаты, "
            "которые прошли similarity_threshold после FAISS-поиска."
        )
        print(f"[Reranker] Кандидатов после объединения вариантов запроса: {merged_results_count}")
        print(f"[Reranker] Кандидатов после similarity_threshold: {filtered_results_count}")

        if not self.use_reranker:
            print("[Reranker] Реранкер НЕ будет вызван: RAG_USE_RERANKER выключен или не задан.")
        elif not filtered_results_count:
            print("[Reranker] Реранкер НЕ будет вызван: нет кандидатов после порога релевантности.")
        else:
            if self.reranker is None:
                print("[Reranker] Реранкер будет вызван сейчас; модель сначала загрузится лениво.")
            else:
                print("[Reranker] Реранкер будет вызван сейчас.")

        print("[Reranker] Конец проверки перед реранкингом\n")

    def _rerank_results(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        reranker: Any,
    ) -> list[RetrievedChunk]:
        pairs = [[query, self._chunk_to_reranker_text(chunk)] for chunk in chunks]
        self._log_reranker_input(query, chunks, pairs)
        scores = reranker.predict(pairs)
        scores_list = [float(score) for score in scores]
        self._log_reranker_output(chunks, scores_list)

        reranked_chunks = [
            self._replace_rerank_score(chunk, score)
            for chunk, score in zip(chunks, scores_list)
        ]
        reranked_chunks.sort(
            key=lambda chunk: (
                chunk.rerank_score if chunk.rerank_score is not None else float("-inf"),
                chunk.score,
            ),
            reverse=True,
        )
        return reranked_chunks

    def _log_reranker_input(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        pairs: list[list[str]],
    ) -> None:
        if not self.log_reranker_io:
            return

        print("\n[Reranker] Что отправляем в модель реранкера")
        print(
            "[Reranker] Пояснение: CrossEncoder получает список пар "
            "[запрос пользователя, текст найденного кандидата]. "
            "Для каждой пары модель вернет числовой rerank_score."
        )
        print(f"[Reranker] Модель: {self.reranker_model_name}")
        print(f"[Reranker] Количество пар: {len(pairs)}")
        print(f"[Reranker] Общий запрос для всех пар: {query}")

        for index, (chunk, pair) in enumerate(zip(chunks, pairs), start=1):
            print(f"\n[Reranker] Пара #{index}, отправленная в модель")
            print(
                "[Reranker] Метаданные кандидата: "
                f"faiss_score={chunk.score:.4f}, "
                f"global_chunk_index={chunk.global_chunk_index}, "
                f"file_chunk_index={chunk.file_chunk_index}, "
                f"title={chunk.title}, "
                f"variants={','.join(chunk.query_variants)}"
            )
            print("[Reranker] pair[0] = запрос:")
            print(pair[0])
            print("[Reranker] pair[1] = текст кандидата:")
            print(pair[1])

        print("[Reranker] Конец входных данных реранкера\n")

    def _log_reranker_output(
        self,
        chunks: list[RetrievedChunk],
        scores: list[float],
    ) -> None:
        if not self.log_reranker_io:
            return

        print("\n[Reranker] Что вернула модель реранкера")
        print(
            "[Reranker] Пояснение: это сырые оценки релевантности от CrossEncoder. "
            "Чем выше rerank_score, тем выше кандидат будет после сортировки."
        )

        for index, (chunk, score) in enumerate(zip(chunks, scores), start=1):
            print(
                f"[Reranker] Пара #{index}: "
                f"rerank_score={score:.6f}, "
                f"faiss_score={chunk.score:.4f}, "
                f"global_chunk_index={chunk.global_chunk_index}, "
                f"title={chunk.title}"
            )

        print("[Reranker] Конец ответа реранкера\n")

    @staticmethod
    def _chunk_to_reranker_text(chunk: RetrievedChunk) -> str:
        return f"{chunk.title}. {chunk.text}".strip()

    @staticmethod
    def _replace_rerank_score(
        chunk: RetrievedChunk,
        rerank_score: float | None,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            score=chunk.score,
            text=chunk.text,
            title=chunk.title,
            global_chunk_index=chunk.global_chunk_index,
            file_chunk_index=chunk.file_chunk_index,
            metadata=chunk.metadata,
            faiss_index=chunk.faiss_index,
            rerank_score=rerank_score,
            query_variants=chunk.query_variants,
        )

    @staticmethod
    def _load_json_list(path: Path, label: str) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(f"{label} must be a JSON list: {path}")

        return [dict(item) for item in data]

    def _load_metadata_items(self) -> tuple[list[dict[str, Any]], str]:
        if self.metadata_path is None or not self.metadata_path.exists():
            return self.chunks, "all_chunks.json"

        metadata_items = self._load_json_list(self.metadata_path, "metadata")
        if metadata_items == self.chunks:
            return self.chunks, "all_chunks.json"

        if len(metadata_items) != len(self.chunks):
            raise ValueError(
                "chunks_metadata.json must have the same length as all_chunks.json: "
                f"{len(metadata_items)} != {len(self.chunks)}."
            )

        return metadata_items, str(self.metadata_path)

    def _metadata_for_index(self, idx: int) -> dict[str, Any]:
        if idx >= len(self.metadata_items):
            return {}

        return dict(self.metadata_items[idx])

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None

        return int(value)

    @staticmethod
    def _merge_duplicate_chunks(
        existing_chunk: RetrievedChunk,
        new_chunk: RetrievedChunk,
    ) -> RetrievedChunk:
        query_variants = sorted(
            set(existing_chunk.query_variants) | set(new_chunk.query_variants)
        )
        best_chunk = new_chunk if new_chunk.score > existing_chunk.score else existing_chunk

        return RetrievedChunk(
            score=max(existing_chunk.score, new_chunk.score),
            text=best_chunk.text,
            title=best_chunk.title,
            global_chunk_index=best_chunk.global_chunk_index,
            file_chunk_index=best_chunk.file_chunk_index,
            metadata=best_chunk.metadata,
            faiss_index=best_chunk.faiss_index,
            rerank_score=best_chunk.rerank_score,
            query_variants=query_variants,
        )


def print_retrieval_result(result: RetrievalResult) -> None:
    print()
    print(f"Status: {result.status}")
    print(f"Original query: {result.original_query}")
    print(f"Rewritten query: {result.rewritten_query}")
    print(f"HyDE query: {result.hyde_query}")
    print(
        "Debug: "
        f"raw={result.debug_info.raw_results_by_variant}, "
        f"merged={result.debug_info.merged_results_count}, "
        f"after_threshold={result.debug_info.results_after_threshold}, "
        f"threshold={result.debug_info.similarity_threshold}, "
        f"reranking={result.debug_info.reranking_enabled}, "
        f"rewrite={result.debug_info.rewrite_source}, "
        f"hyde={result.debug_info.hyde_source}, "
        f"llm_model={result.debug_info.llm_model or 'n/a'}, "
        f"llm_error={result.debug_info.llm_error or 'none'}"
    )
    print()

    if not result.results:
        print("No chunks passed the similarity threshold.")
        return

    for rank, chunk in enumerate(result.results, start=1):
        text_preview = textwrap.shorten(
            chunk.text.replace("\n", " "),
            width=650,
            placeholder="...",
        )
        print(
            f"{rank}. faiss_score={chunk.score:.4f}, "
            f"rerank_score={chunk.rerank_score if chunk.rerank_score is not None else 'n/a'}, "
            f"global_chunk_index={chunk.global_chunk_index}, "
            f"file_chunk_index={chunk.file_chunk_index}, "
            f"variants={','.join(chunk.query_variants)}"
        )
        print(f"   title={chunk.title}")
        print(f"   text={text_preview}")
        print()


def main() -> None:
    load_project_dotenv()

    retriever = MedicalRAGRetriever(
        top_k=5,
        similarity_threshold=SIMILARITY_THRESHOLD,
        query_transformer=QueryTransformer(use_llm=True),
    )

    has_api_key = bool(os.getenv("API_KEY"))

    print("Medical RAG retriever is ready.")
    print(
        "Yandex LLM Rewrite/HyDE: "
        + ("enabled via API_KEY" if has_api_key else "no key found, using local fallback")
    )
    print("Type a medical query and press Enter. Type 'exit' or 'quit' to stop.")
    print()

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue

        try:
            result = retriever.retrieve(query)
        except Exception as error:
            print(f"Retrieval error: {error}")
            continue

        print_retrieval_result(result)


if __name__ == "__main__":
    main()
