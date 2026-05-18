from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_CHUNKS_PATH = Path("data/chunks.json")


@dataclass(frozen=True)
class MedicalChunk:
    """One text chunk stored in the medical vector index."""

    chunk_id: str
    title: str
    text: str
    metadata: dict[str, Any]


class MedicalChunkIndex:
    """
    Minimal VectorDB/MedicalChunkIndex implementation for testing retrieval.

    Important:
    - BAAI/bge-m3 is the embedding model. It converts text into vectors.
    - FAISS is the vector index / storage. It searches over already built vectors.
    - The same embedding model must be used for chunk indexing and query embedding.

    This class exposes the method expected by MedicalRAGRetriever:

        search(query: str, top_k: int) -> list[dict[str, Any]]
    """

    def __init__(
        self,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        use_faiss: bool = True,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.model_name = model_name
        self.use_faiss = use_faiss

        self.embedding_model = SentenceTransformer(model_name)
        self.chunks = self._load_chunks(self.chunks_path)
        self.chunk_texts = [self._chunk_to_embedding_text(chunk) for chunk in self.chunks]

        self.chunk_embeddings = self._embed_texts(self.chunk_texts)
        self.index = self._build_faiss_index(self.chunk_embeddings) if use_faiss else None

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Embed query and return the most similar chunks.

        The retriever calls this method for original, rewritten, and HyDE queries.
        Each of those query variants is embedded here with the same BAAI/bge-m3
        model that was used to embed document chunks.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not query.strip():
            raise ValueError("query must not be empty.")

        query_embedding = self._embed_texts([query.strip()])
        limit = min(top_k, len(self.chunks))

        if self.index is not None:
            scores, indices = self.index.search(query_embedding, limit)
            ranked_pairs = zip(indices[0].tolist(), scores[0].tolist())
        else:
            scores = self.chunk_embeddings @ query_embedding[0]
            top_indices = np.argsort(-scores)[:limit]
            ranked_pairs = ((int(index), float(scores[index])) for index in top_indices)

        results = []
        for chunk_index, score in ranked_pairs:
            if chunk_index < 0:
                continue

            chunk = self.chunks[chunk_index]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "score": float(score),
                    "metadata": {
                        **chunk.metadata,
                        "title": chunk.title,
                        "model_name": self.model_name,
                    },
                }
            )

        return results

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts with BAAI/bge-m3 and L2-normalize embeddings.

        With normalized vectors, inner product in FAISS IndexFlatIP is equivalent
        to cosine similarity.
        """
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _build_faiss_index(embeddings: np.ndarray) -> Any:
        """
        Build a FAISS vector index over chunk embeddings.

        FAISS does not create embeddings. It only stores vectors and performs
        nearest-neighbor search over them.
        """
        try:
            import faiss
        except ImportError as error:
            raise ImportError(
                "faiss-cpu is not installed. Install it with `pip install faiss-cpu` "
                "or create MedicalChunkIndex(..., use_faiss=False) for numpy fallback."
            ) from error

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    @staticmethod
    def _load_chunks(path: Path) -> list[MedicalChunk]:
        """Load chunks from a JSON file with id/title/text fields."""
        with path.open("r", encoding="utf-8") as file:
            raw_chunks = json.load(file)

        chunks = []
        for raw_chunk in raw_chunks:
            chunk_id = str(raw_chunk["id"])
            title = str(raw_chunk.get("title", ""))
            text = str(raw_chunk["text"])
            metadata = {
                key: value
                for key, value in raw_chunk.items()
                if key not in {"id", "title", "text"}
            }
            metadata["source_path"] = str(path)

            chunks.append(
                MedicalChunk(
                    chunk_id=chunk_id,
                    title=title,
                    text=text,
                    metadata=metadata,
                )
            )

        return chunks

    @staticmethod
    def _chunk_to_embedding_text(chunk: MedicalChunk) -> str:
        """Combine title and body so semantic search sees the medical topic."""
        return f"{chunk.title}. {chunk.text}".strip()


def main() -> None:
    """Small manual test for MedicalChunkIndex.search."""
    index = MedicalChunkIndex(
        chunks_path=DEFAULT_CHUNKS_PATH,
        model_name=DEFAULT_MODEL_NAME,
        use_faiss=True,
    )

    query = "часто хочу пить, высокий сахар, что проверить"
    results = index.search(query=query, top_k=3)

    print(f"Model: {DEFAULT_MODEL_NAME}")
    print(f"Query: {query}")
    print()

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. score={result['score']:.4f}, chunk_id={result['chunk_id']}")
        print(f"   title={result['metadata'].get('title')}")
        print(f"   text={result['text']}")


if __name__ == "__main__":
    main()
