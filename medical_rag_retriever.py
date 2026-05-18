from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Protocol


DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.55


class MedicalChunkIndexProtocol(Protocol):
    """Minimal protocol expected from VectorDB/MedicalChunkIndex."""

    def search(self, query: str, top_k: int) -> list[Any]:
        """Return top-k chunks for the query with score and metadata."""
        ...


@dataclass
class QueryTransformer:
    """
    Placeholder for query rewrite and HyDE generation.

    Later this class can call a real LLM. For now, it either uses optional
    functions passed from outside or returns transparent stub values.
    """

    rewrite_fn: Callable[[str], str] | None = None
    hyde_fn: Callable[[str], str] | None = None

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite the user query in a more formal medical style.

        A production implementation may fix typos, expand abbreviations,
        normalize medical terms, and translate colloquial wording into
        clinical terminology.
        """
        if self.rewrite_fn is not None:
            return self.rewrite_fn(query)

        return f"Формальный медицинский запрос: {query.strip()}"

    def generate_hyde(self, query: str) -> str:
        """
        Generate a pseudo-answer for HyDE retrieval.

        HyDE means that a hypothetical answer is embedded and searched as an
        additional semantic query. This is only a stub until an LLM is attached.
        """
        if self.hyde_fn is not None:
            return self.hyde_fn(query)

        return (
            "Псевдоответ для медицинского поиска: вероятные симптомы, диагноз, "
            f"факторы риска, обследование и лечение, связанные с запросом: {query.strip()}"
        )


@dataclass(frozen=True)
class RetrievedChunk:
    """A normalized chunk returned by the retriever."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None
    query_variants: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalDebugInfo:
    """Debug counters for retrieval and filtering."""

    raw_results_by_variant: dict[str, int]
    merged_results_count: int
    results_after_threshold: int
    top_k: int
    similarity_threshold: float


@dataclass(frozen=True)
class RetrievalResult:
    """Structured output of the medical RAG retriever."""

    status: str
    original_query: str
    rewritten_query: str
    hyde_query: str
    results: list[RetrievedChunk]
    debug_info: RetrievalDebugInfo


class MedicalRAGRetriever:
    """
    Retriever component for a medical RAG system.

    This class intentionally knows nothing about FAISS internals. It only calls
    index.search(query, top_k). The VectorDB/MedicalChunkIndex dependency is
    responsible for query embedding, vector normalization, FAISS lookup, and
    returning chunks with scores.

    Important architecture notes:
    - BAAI/bge-m3 is an embedding model.
    - FAISS is a vector index / storage for nearest-neighbor search, not an
      embedding model.
    - Use the same embedder for indexing chunks and embedding user queries.
    """

    ORIGINAL_VARIANT = "original"
    REWRITTEN_VARIANT = "rewritten"
    HYDE_VARIANT = "hyde"

    def __init__(
        self,
        index: MedicalChunkIndexProtocol,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        query_transformer: QueryTransformer | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1.")

        self.index = index
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.query_transformer = query_transformer or QueryTransformer()

    def retrieve(self, query: str) -> RetrievalResult:
        """Run multi-query semantic retrieval and return filtered chunks."""
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
        merged_chunks: dict[str, RetrievedChunk] = {}

        for variant_name, variant_query in query_variants.items():
            raw_results = self.index.search(variant_query, self.top_k)
            raw_results_by_variant[variant_name] = len(raw_results)

            for raw_result in raw_results:
                chunk = self._normalize_search_result(raw_result, variant_name)
                dedup_key = self._dedup_key(chunk)

                existing_chunk = merged_chunks.get(dedup_key)
                if existing_chunk is None:
                    merged_chunks[dedup_key] = chunk
                    continue

                merged_chunks[dedup_key] = self._merge_duplicate_chunks(
                    existing_chunk,
                    chunk,
                )

        filtered_results = [
            chunk
            for chunk in merged_chunks.values()
            if chunk.score >= self.similarity_threshold
        ]
        filtered_results.sort(key=lambda chunk: chunk.score, reverse=True)

        status = "ok" if filtered_results else "no_relevant_chunks"
        debug_info = RetrievalDebugInfo(
            raw_results_by_variant=raw_results_by_variant,
            merged_results_count=len(merged_chunks),
            results_after_threshold=len(filtered_results),
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
        )

        return RetrievalResult(
            status=status,
            original_query=normalized_query,
            rewritten_query=rewritten_query,
            hyde_query=hyde_query,
            results=filtered_results,
            debug_info=debug_info,
        )

    def _normalize_search_result(
        self,
        raw_result: Any,
        variant_name: str,
    ) -> RetrievedChunk:
        """Convert dict/dataclass/object search result into RetrievedChunk."""
        result = self._to_mapping(raw_result)

        metadata = dict(result.get("metadata") or {})
        chunk_id = (
            result.get("chunk_id")
            or result.get("id")
            or metadata.get("chunk_id")
            or metadata.get("id")
        )
        text = (
            result.get("text")
            or result.get("content")
            or result.get("chunk_text")
            or metadata.get("text")
            or metadata.get("content")
            or ""
        )
        score = result.get("score", result.get("similarity", 0.0))

        return RetrievedChunk(
            text=str(text),
            score=float(score),
            metadata=metadata,
            chunk_id=str(chunk_id) if chunk_id is not None else None,
            query_variants=[variant_name],
        )

    @staticmethod
    def _to_mapping(raw_result: Any) -> dict[str, Any]:
        """Best-effort conversion of common search result shapes to a dict."""
        if isinstance(raw_result, dict):
            return raw_result
        if is_dataclass(raw_result):
            return asdict(raw_result)

        fields = ("chunk_id", "id", "text", "content", "chunk_text", "score", "similarity", "metadata")
        return {
            field_name: getattr(raw_result, field_name)
            for field_name in fields
            if hasattr(raw_result, field_name)
        }

    @staticmethod
    def _dedup_key(chunk: RetrievedChunk) -> str:
        """Prefer stable chunk_id, fallback to exact chunk text."""
        if chunk.chunk_id:
            return f"id:{chunk.chunk_id}"
        return f"text:{chunk.text.strip()}"

    @staticmethod
    def _merge_duplicate_chunks(
        existing_chunk: RetrievedChunk,
        new_chunk: RetrievedChunk,
    ) -> RetrievedChunk:
        """Merge duplicate chunks, keeping max score and all query variants."""
        query_variants = sorted(
            set(existing_chunk.query_variants) | set(new_chunk.query_variants)
        )
        best_chunk = new_chunk if new_chunk.score > existing_chunk.score else existing_chunk

        metadata = dict(existing_chunk.metadata)
        metadata.update(new_chunk.metadata)

        return RetrievedChunk(
            text=best_chunk.text,
            score=max(existing_chunk.score, new_chunk.score),
            metadata=metadata,
            chunk_id=best_chunk.chunk_id,
            query_variants=query_variants,
        )


class MockMedicalChunkIndex:
    """
    Fallback stub so this file can be run without a real FAISS index.

    In the real project, replace this with MedicalChunkIndex. That index should
    use one embedding model, for example BAAI/bge-m3, both for chunk embeddings
    and query embeddings. FAISS should only store/search vectors.
    """

    def __init__(self) -> None:
        self.chunks = [
            {
                "chunk_id": "diabetes_1",
                "text": (
                    "Сахарный диабет проявляется повышением глюкозы крови, "
                    "жаждой, частым мочеиспусканием и требует контроля HbA1c."
                ),
                "metadata": {"title": "Сахарный диабет"},
            },
            {
                "chunk_id": "copd_1",
                "text": (
                    "ХОБЛ связана с хронической одышкой, кашлем, мокротой и "
                    "часто развивается у курящих пациентов."
                ),
                "metadata": {"title": "Хроническая обструктивная болезнь легких"},
            },
            {
                "chunk_id": "stroke_1",
                "text": (
                    "При инсульте важны внезапная слабость в конечностях, "
                    "асимметрия лица, нарушение речи и срочная госпитализация."
                ),
                "metadata": {"title": "Инсульт"},
            },
        ]

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Tiny keyword-based mock. Real MedicalChunkIndex.search should embed the
        query with the same embedder used for chunks and search in FAISS.
        """
        query_lower = query.lower()
        scored_chunks = []

        for chunk in self.chunks:
            text_lower = chunk["text"].lower()
            title_lower = chunk["metadata"]["title"].lower()
            score = 0.25

            if any(word in query_lower for word in ("диабет", "глюкоз", "сахар", "hba1c")):
                score = 0.82 if chunk["chunk_id"] == "diabetes_1" else 0.38
            elif any(word in query_lower for word in ("хобл", "одыш", "каш", "мокрот")):
                score = 0.80 if chunk["chunk_id"] == "copd_1" else 0.35
            elif any(word in query_lower for word in ("инсульт", "реч", "лиц", "слабост")):
                score = 0.84 if chunk["chunk_id"] == "stroke_1" else 0.36
            elif any(token in text_lower or token in title_lower for token in query_lower.split()):
                score = 0.58

            scored_chunks.append({**chunk, "score": score})

        return sorted(scored_chunks, key=lambda item: item["score"], reverse=True)[:top_k]


def main() -> None:
    """Example usage with a mock index."""
    index = MockMedicalChunkIndex()
    retriever = MedicalRAGRetriever(
        index=index,
        top_k=3,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
    )

    query = "часто хочу пить и сахар высокий что это может быть"
    result = retriever.retrieve(query)

    print(f"Status: {result.status}")
    print(f"Original query: {result.original_query}")
    print(f"Rewritten query: {result.rewritten_query}")
    print(f"HyDE query: {result.hyde_query}")
    print(f"Debug info: {result.debug_info}")
    print()

    for index, chunk in enumerate(result.results, start=1):
        print(f"{index}. score={chunk.score:.3f}, chunk_id={chunk.chunk_id}")
        print(f"   query_variants={chunk.query_variants}")
        print(f"   title={chunk.metadata.get('title')}")
        print(f"   text={chunk.text}")


if __name__ == "__main__":
    main()
