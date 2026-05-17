import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer


DATA_DIR = Path("data")
DEFAULT_CHUNKS_PATH = DATA_DIR / "chunks.json"
DEFAULT_QUERIES_PATH = DATA_DIR / "queries.json"
DEFAULT_RESULTS_PATH = Path("results.csv")
DEFAULT_ERRORS_PATH = Path("errors.csv")


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    backend: str = "sentence_transformers"
    query_prefix: str = ""
    passage_prefix: str = ""


MODEL_SPECS = [
    EmbeddingModelSpec("BAAI/bge-m3"),
    EmbeddingModelSpec("microsoft/biogpt", backend="hf_mean_pooling"),
    EmbeddingModelSpec(
        "intfloat/multilingual-e5-large",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    EmbeddingModelSpec("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def chunk_to_text(chunk: dict[str, str]) -> str:
    return f"{chunk['title']}. {chunk['text']}"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, a_min=1e-12, a_max=None)


class HFMeanPoolingEncoder:
    """Small wrapper for models that are not native sentence-transformers models."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 8,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        vectors = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with self.torch.no_grad():
                outputs = self.model(**inputs)

            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_embeddings = token_embeddings * attention_mask
            summed = masked_embeddings.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            batch_vectors = summed / counts
            vectors.append(batch_vectors.cpu().numpy())

        result = np.vstack(vectors)
        if normalize_embeddings:
            result = normalize_rows(result)
        return result.astype(np.float32)


def load_encoder(spec: EmbeddingModelSpec) -> Any:
    if spec.backend == "sentence_transformers":
        return SentenceTransformer(spec.name)
    if spec.backend == "hf_mean_pooling":
        return HFMeanPoolingEncoder(spec.name)
    raise ValueError(f"Unknown backend: {spec.backend}")


def encode_texts(
    encoder: Any,
    texts: list[str],
    prefix: str = "",
    batch_size: int = 8,
) -> np.ndarray:
    prefixed_texts = [prefix + text for text in texts]
    embeddings = encoder.encode(
        prefixed_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def top_k_by_cosine(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_ids: list[str],
    k: int,
) -> list[tuple[str, float]]:
    scores = chunk_embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:k]
    return [(chunk_ids[index], float(scores[index])) for index in top_indices]


def reciprocal_rank(ranked_ids: list[str], expected_id: str) -> float:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id == expected_id:
            return 1.0 / rank
    return 0.0


def calculate_metrics(query_results: list[dict[str, Any]], time_key: str) -> dict[str, float]:
    total = len(query_results)
    recall_at_1 = sum(row["expected_chunk_id"] in row["top_ids"][:1] for row in query_results) / total
    recall_at_3 = sum(row["expected_chunk_id"] in row["top_ids"][:3] for row in query_results) / total
    mrr = sum(reciprocal_rank(row["top_ids"], row["expected_chunk_id"]) for row in query_results) / total
    avg_time = sum(row[time_key] for row in query_results) / total

    return {
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "mrr": mrr,
        "avg_query_time_sec": avg_time,
    }


def build_metric_rows(
    model_name: str,
    stage: str,
    query_results: list[dict[str, Any]],
    time_key: str,
) -> list[dict[str, Any]]:
    rows = [
        {
            "model": model_name,
            "stage": stage,
            "query_type": "all",
            **calculate_metrics(query_results, time_key),
        }
    ]

    query_types = sorted({row["query_type"] for row in query_results})
    for query_type in query_types:
        typed_results = [
            row for row in query_results if row["query_type"] == query_type
        ]
        rows.append(
            {
                "model": model_name,
                "stage": stage,
                "query_type": query_type,
                **calculate_metrics(typed_results, time_key),
            }
        )

    return rows


def format_top_ids(top_results: list[tuple[str, float]]) -> str:
    return " | ".join(f"{chunk_id}:{score:.4f}" for chunk_id, score in top_results)


def evaluate_embedding_model(
    spec: EmbeddingModelSpec,
    chunks: list[dict[str, str]],
    queries: list[dict[str, str]],
    top_k: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    print(f"\nLoading embedding model: {spec.name}")
    encoder = load_encoder(spec)

    chunk_ids = [chunk["id"] for chunk in chunks]
    chunk_texts = [chunk_to_text(chunk) for chunk in chunks]

    print("Encoding chunks...")
    chunk_embeddings = encode_texts(
        encoder,
        chunk_texts,
        prefix=spec.passage_prefix,
        batch_size=batch_size,
    )

    query_results = []
    print("Running retrieval...")
    for query in queries:
        started_at = time.perf_counter()
        query_embedding = encode_texts(
            encoder,
            [query["query"]],
            prefix=spec.query_prefix,
            batch_size=batch_size,
        )[0]
        top_results = top_k_by_cosine(query_embedding, chunk_embeddings, chunk_ids, top_k)
        elapsed = time.perf_counter() - started_at

        top_ids = [chunk_id for chunk_id, _ in top_results]
        query_results.append(
            {
                "model": spec.name,
                "query": query["query"],
                "query_type": query["query_type"],
                "expected_chunk_id": query["expected_chunk_id"],
                "top_ids": top_ids,
                "top_results": top_results,
                "query_time_sec": elapsed,
            }
        )

    rows = build_metric_rows(
        model_name=spec.name,
        stage="embedding",
        query_results=query_results,
        time_key="query_time_sec",
    )
    return rows, query_results, chunk_embeddings


def rerank_results(
    model_name: str,
    query_results: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, str]],
    reranker_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(f"Loading reranker: {reranker_name}")
    reranker = CrossEncoder(reranker_name)

    reranked_results = []
    print("Running reranking...")
    for row in query_results:
        started_at = time.perf_counter()
        pairs = [
            [row["query"], chunk_to_text(chunks_by_id[chunk_id])]
            for chunk_id in row["top_ids"]
        ]
        scores = reranker.predict(pairs)
        scored_ids = sorted(
            zip(row["top_ids"], scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        elapsed = time.perf_counter() - started_at

        reranked_results.append(
            {
                **row,
                "top_ids": [chunk_id for chunk_id, _ in scored_ids],
                "reranked_top_results": [
                    (chunk_id, float(score)) for chunk_id, score in scored_ids
                ],
                "rerank_time_sec": elapsed,
            }
        )

    rows = build_metric_rows(
        model_name=model_name,
        stage="reranked",
        query_results=reranked_results,
        time_key="rerank_time_sec",
    )
    return rows, reranked_results


def build_error_rows(query_results: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    errors = []
    for row in query_results:
        if row["expected_chunk_id"] not in row["top_ids"][:3]:
            errors.append(
                {
                    "model": row["model"],
                    "stage": stage,
                    "query": row["query"],
                    "query_type": row["query_type"],
                    "expected_chunk_id": row["expected_chunk_id"],
                    "top_3": " | ".join(row["top_ids"][:3]),
                    "top_10_with_scores": format_top_ids(
                        row.get("reranked_top_results", row["top_results"])
                    ),
                }
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare embedding models for toy Russian clinical semantic retrieval."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", action="append", help="Run only this model. Can be repeated.")
    parser.add_argument("--rerank", action="store_true", help="Apply BAAI/bge-reranker-v2-m3 to top-k.")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_json(args.chunks)
    queries = load_json(args.queries)
    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}

    if args.top_k < 3:
        raise ValueError("--top-k must be at least 3 to compute Recall@3.")

    selected_specs = MODEL_SPECS
    if args.model:
        requested = set(args.model)
        selected_specs = [spec for spec in MODEL_SPECS if spec.name in requested]
        unknown = requested - {spec.name for spec in MODEL_SPECS}
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown))}")

    result_rows = []
    error_rows = []

    for spec in selected_specs:
        metric_rows, query_results, _ = evaluate_embedding_model(
            spec=spec,
            chunks=chunks,
            queries=queries,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        result_rows.extend(metric_rows)
        error_rows.extend(build_error_rows(query_results, stage="embedding"))

        if args.rerank:
            reranked_rows, reranked_results = rerank_results(
                model_name=spec.name,
                query_results=query_results,
                chunks_by_id=chunks_by_id,
                reranker_name=args.reranker_model,
            )
            result_rows.extend(reranked_rows)
            error_rows.extend(build_error_rows(reranked_results, stage="reranked"))

    results_df = pd.DataFrame(result_rows)
    errors_df = pd.DataFrame(error_rows)

    results_df.to_csv(args.results, index=False)
    errors_df.to_csv(args.errors, index=False)

    print("\nResults")
    print(results_df.to_string(index=False))

    if errors_df.empty:
        print("\nNo errors: every expected chunk was found in top-3.")
    else:
        print("\nErrors: expected chunk was not found in top-3")
        print(errors_df[["model", "stage", "query_type", "query", "expected_chunk_id", "top_3"]].to_string(index=False))
        print(f"\nFull error table saved to {args.errors}")

    print(f"\nMetrics saved to {args.results}")


if __name__ == "__main__":
    main()
