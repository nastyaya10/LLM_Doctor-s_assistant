from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from retriever.paths import DEFAULT_METADATA_PATH


DEFAULT_NEIGHBOR_RADIUS = int(os.getenv("RAG_NEIGHBOR_RADIUS", "1"))
DEFAULT_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "30000"))


class RetrievalContextBuilder:
    """Builds an LLM context from retrieved chunks and their nearby chunks."""

    def __init__(
        self,
        chunks_path: str | Path = DEFAULT_METADATA_PATH,
        neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        if neighbor_radius < 0:
            raise ValueError("neighbor_radius must be non-negative.")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive.")

        self.chunks_path = Path(chunks_path)
        self.neighbor_radius = neighbor_radius
        self.max_context_chars = max_context_chars
        self.chunks = self._load_chunks(self.chunks_path)
        self.by_global_index = {
            int(chunk["global_chunk_index"]): chunk
            for chunk in self.chunks
            if chunk.get("global_chunk_index") is not None
        }

    def build_context(self, retrieval_result: Any) -> str:
        if getattr(retrieval_result, "status", None) != "ok":
            return ""

        context_parts: list[str] = []
        seen: set[int] = set()

        for result in getattr(retrieval_result, "results", []):
            for chunk in self.get_chunk_with_neighbors(result):
                global_index = self._optional_int(chunk.get("global_chunk_index"))
                if global_index is None or global_index in seen:
                    continue

                text = str(chunk.get("text", "")).strip()
                if not text:
                    continue

                seen.add(global_index)
                context_parts.append(self._format_chunk(chunk, text))

        return self._join_with_limit(context_parts)

    def get_chunk_with_neighbors(self, retrieved_chunk: Any) -> list[dict[str, Any]]:
        center_index = self._chunk_id_to_int(retrieved_chunk)
        if center_index is None:
            text = str(getattr(retrieved_chunk, "text", "")).strip()
            if not text:
                return []
            metadata = dict(getattr(retrieved_chunk, "metadata", {}) or {})
            metadata.setdefault("text", text)
            return [metadata]

        center_chunk = self.by_global_index.get(center_index)
        center_title = self._title_for_retrieved_chunk(retrieved_chunk, center_chunk)

        chunks: list[dict[str, Any]] = []
        for index in range(
            center_index - self.neighbor_radius,
            center_index + self.neighbor_radius + 1,
        ):
            chunk = self.by_global_index.get(index)
            if chunk is None:
                continue
            if center_title and chunk.get("title") != center_title:
                continue
            chunks.append(chunk)

        return chunks

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"chunks file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(f"chunks file must be a JSON list: {path}")

        return [dict(item) for item in data]

    @staticmethod
    def _chunk_id_to_int(retrieved_chunk: Any) -> int | None:
        chunk_id = getattr(retrieved_chunk, "chunk_id", None)
        if chunk_id is None:
            chunk_id = getattr(retrieved_chunk, "global_chunk_index", None)
        return RetrievalContextBuilder._optional_int(chunk_id)

    @staticmethod
    def _title_for_retrieved_chunk(
        retrieved_chunk: Any,
        center_chunk: dict[str, Any] | None,
    ) -> str:
        metadata = getattr(retrieved_chunk, "metadata", {}) or {}
        return str(
            metadata.get("title")
            or getattr(retrieved_chunk, "title", "")
            or (center_chunk or {}).get("title", "")
        )

    @staticmethod
    def _format_chunk(chunk: dict[str, Any], text: str) -> str:
        title = str(chunk.get("title") or "Без названия")
        global_index = chunk.get("global_chunk_index", "?")
        file_index = chunk.get("file_chunk_index", "?")
        return (
            f"[Источник: {title}; global_chunk_index={global_index}; "
            f"file_chunk_index={file_index}]\n{text}"
        )

    def _join_with_limit(self, context_parts: list[str]) -> str:
        selected: list[str] = []
        total_chars = 0

        for part in context_parts:
            separator_chars = 6 if selected else 0
            next_total = total_chars + separator_chars + len(part)
            if next_total > self.max_context_chars:
                break
            selected.append(part)
            total_chars = next_total

        return "\n\n---\n\n".join(selected)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None


def build_retrieval_context(
    retrieval_result: Any,
    chunks_path: str | Path = DEFAULT_METADATA_PATH,
    neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    builder = RetrievalContextBuilder(
        chunks_path=chunks_path,
        neighbor_radius=neighbor_radius,
        max_context_chars=max_context_chars,
    )
    return builder.build_context(retrieval_result)
