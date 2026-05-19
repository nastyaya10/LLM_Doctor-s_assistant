from __future__ import annotations

import json
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_K = 5

# Explicit retrieval cutoff. With a normalized BGE-M3 embedding + IndexFlatIP
# this is cosine-like similarity. Tune it on your validation queries.
SIMILARITY_THRESHOLD = 0.55
DEFAULT_SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DOTENV_PATH = PROJECT_DIR / ".env"
LEGACY_DOTENV_PATH = PROJECT_DIR / ".env.py"
DEFAULT_FAISS_INDEX_PATH = PROJECT_DIR / "faiss_index.bin"
DEFAULT_CHUNKS_PATH = PROJECT_DIR / "all_chunks.json"
DEFAULT_METADATA_PATH = PROJECT_DIR / "chunks_metadata.json"


def load_project_dotenv() -> bool:
    """Load GEMINI_API_KEY and other local settings from the project env file."""
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

    By default it tries Gemini API for Rewrite and HyDE when GEMINI_API_KEY is
    configured. If Gemini is unavailable, it falls back to deterministic medical
    term expansion.
    """

    rewrite_fn: Callable[[str], str] | None = None
    hyde_fn: Callable[[str], str] | None = None
    use_gemini: bool = True
    gemini_model: str = "gemini-2.5-flash"
    gemini_fallback_models: tuple[str, ...] = ("gemini-2.0-flash",)
    gemini_api_key_env: str = "GEMINI_API_KEY"
    gemini_max_retries: int = 2
    last_rewrite_source: str = field(default="not_run", init=False)
    last_hyde_source: str = field(default="not_run", init=False)
    last_gemini_error: str | None = field(default=None, init=False)
    last_gemini_model: str | None = field(default=None, init=False)

    def rewrite_query(self, query: str) -> str:
        if self.rewrite_fn is not None:
            self.last_rewrite_source = "custom"
            return self.rewrite_fn(query).strip()

        if self.use_gemini:
            rewritten_query = self._try_gemini_rewrite(query)
            if rewritten_query:
                self.last_rewrite_source = "gemini"
                return rewritten_query

        self.last_rewrite_source = "fallback"
        return self._rule_based_rewrite(query)

    def generate_hyde(self, query: str) -> str:
        if self.hyde_fn is not None:
            self.last_hyde_source = "custom"
            return self.hyde_fn(query).strip()

        if self.use_gemini:
            hyde_query = self._try_gemini_hyde(query)
            if hyde_query:
                self.last_hyde_source = "gemini"
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

    def _try_gemini_rewrite(self, query: str) -> str | None:
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
        return self._call_gemini(prompt)

    def _try_gemini_hyde(self, query: str) -> str | None:
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
        return self._call_gemini(prompt)

    def _call_gemini(self, prompt: str) -> str | None:
        self._load_dotenv_if_available()
        api_key = os.getenv(self.gemini_api_key_env)
        if not api_key:
            self.last_gemini_error = f"{self.gemini_api_key_env} is not set"
            return None

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
        except Exception as error:
            self.last_gemini_error = f"{type(error).__name__}: {error}"
            return None

        prompt_text = textwrap.dedent(prompt).strip()
        model_names = self._gemini_model_names()
        last_error: str | None = None

        for model_name in model_names:
            for attempt in range(self.gemini_max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt_text,
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=350,
                        ),
                    )
                    text = getattr(response, "text", None)
                    if not text:
                        last_error = f"{model_name}: Gemini returned empty text"
                        break

                    self.last_gemini_error = None
                    self.last_gemini_model = model_name
                    return self._clean_generated_text(text)
                except Exception as error:
                    last_error = f"{model_name}: {type(error).__name__}: {error}"
                    if attempt < self.gemini_max_retries:
                        time.sleep(0.7 * (attempt + 1))

        self.last_gemini_error = last_error
        self.last_gemini_model = None
        return None

    def _gemini_model_names(self) -> tuple[str, ...]:
        env_model = os.getenv("GEMINI_MODEL")
        preferred_model = env_model or self.gemini_model
        return tuple(dict.fromkeys((preferred_model, *self.gemini_fallback_models)))

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
    gemini_error: str | None
    gemini_model: str | None


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
        use_reranker: bool = True,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL_NAME,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        self.faiss_index_path = Path(faiss_index_path)
        self.chunks_path = Path(chunks_path)
        self.metadata_path = Path(metadata_path) if metadata_path is not None else None
        self.model_name = model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.query_transformer = query_transformer or QueryTransformer()
        self.normalize_query_embeddings = normalize_query_embeddings
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model_name

        self.chunks = self._load_json_list(self.chunks_path, "chunks")
        self.metadata_items, self.metadata_source = self._load_metadata_items()
        self.faiss_index = self._load_faiss_index(self.faiss_index_path)
        self.embedder = self._load_embedder(self.model_name)
        self.reranker = (
            self._load_reranker(self.reranker_model_name) if self.use_reranker else None
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
        if self.reranker is not None and filtered_results:
            filtered_results = self._rerank_results(
                normalized_query,
                filtered_results,
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
            reranker_model_name=self.reranker_model_name if self.reranker else None,
            reranking_enabled=self.reranker is not None,
            rewrite_source=self.query_transformer.last_rewrite_source,
            hyde_source=self.query_transformer.last_hyde_source,
            gemini_error=self.query_transformer.last_gemini_error,
            gemini_model=self.query_transformer.last_gemini_model,
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

        embedding = self.embedder.encode(
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

    def _rerank_results(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        pairs = [[query, self._chunk_to_reranker_text(chunk)] for chunk in chunks]
        scores = self.reranker.predict(pairs)

        reranked_chunks = [
            self._replace_rerank_score(chunk, float(score))
            for chunk, score in zip(chunks, scores)
        ]
        reranked_chunks.sort(
            key=lambda chunk: (
                chunk.rerank_score if chunk.rerank_score is not None else float("-inf"),
                chunk.score,
            ),
            reverse=True,
        )
        return reranked_chunks

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
        f"gemini_model={result.debug_info.gemini_model or 'n/a'}, "
        f"gemini_error={result.debug_info.gemini_error or 'none'}"
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
        query_transformer=QueryTransformer(use_gemini=True),
        use_reranker=True,
    )

    has_gemini_key = bool(os.getenv("GEMINI_API_KEY"))

    print("Medical RAG retriever is ready.")
    print(
        "Gemini Rewrite/HyDE: "
        + ("enabled via GEMINI_API_KEY" if has_gemini_key else "no key found, using local fallback")
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
