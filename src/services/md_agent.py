import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 1. Определяем корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.context_builder import RetrievalContextBuilder
from src.services.medical_rag_retriever import MedicalRAGRetriever
from src.services.prompt_loader import load_prompt
from src.services.router import need_rag
from src.config import CHAT_HISTORY_DEPTH, LLM_BASE_URL, LLM_MODEL_NAME

# Загрузка переменных окружения
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("API_KEY")
folder_id = os.getenv("FOLDER_ID")
base_url = os.getenv("BASE_URL", LLM_BASE_URL)
model_name = os.getenv("MODEL", LLM_MODEL_NAME)

if model_name.startswith("gpt://"):
    model = model_name
elif folder_id:
    model = f"gpt://{folder_id}/{model_name}"
else:
    model = model_name

if not api_key:
    print("Ошибка: переменная API_KEY не найдена.")
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    project=folder_id,
    timeout=60.0,
    max_retries=0
)

_retriever = None
_context_builder = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = MedicalRAGRetriever()
    return _retriever


def get_context_builder():
    global _context_builder
    if _context_builder is None:
        _context_builder = RetrievalContextBuilder()
    return _context_builder


class MedicalAgent:
    def __init__(self):
        print("[Agent] Инициализация RAG-агента.")
        self.messages = []

    def generate_response(self, user_message: str) -> str:
        if not user_message.strip():
            return json.dumps({"answer": "Пожалуйста, введите текст вопроса.", "sources": []})

        use_rag = need_rag(user_message)
        print(f"[Agent] Запрос: {user_message[:50]}... | Режим: {'RAG' if use_rag else 'CHAT'}")

        try:
            available_sources = {}
            sources = []
            debug_chunks = []
            if use_rag:
                retrieval_result = get_retriever().retrieve(user_message)
                
                # top_chunks уже отсортированы реранкером и ограничены в retrieve() до TOP_R
                top_chunks = retrieval_result.results
                
                if not top_chunks:
                    return json.dumps({"answer": "Информации по вопросу в базе не найдено.", "sources": [], "debug_chunks": []})

                context = get_context_builder().build_context(retrieval_result)
                
                seen_sources = set()
                fallback_source = None
                sources = []
                debug_chunks = []
                for index, chunk in enumerate(top_chunks, start=1):
                    if chunk.title not in seen_sources:
                        available_sources[chunk.title] = {"source": chunk.title}
                        seen_sources.add(chunk.title)
                        if fallback_source is None:
                            fallback_source = chunk.title

                    debug_chunks.append(
                        {
                            "rank": index,
                            "source": chunk.title,
                            "global_chunk_index": chunk.global_chunk_index,
                            "file_chunk_index": chunk.file_chunk_index,
                            "score": round(float(chunk.score), 4) if chunk.score is not None else None,
                            "rerank_score": round(float(chunk.rerank_score), 4) if chunk.rerank_score is not None else None,
                            "text": chunk.text,
                        }
                    )
                
                system_prompt = load_prompt("system_prompt")
                user_prompt = load_prompt("user_prompt_rag").format(
                    context=context, 
                    question=user_message
                )
            else:
                system_prompt = load_prompt("system_prompt_chat")
                user_prompt = user_message

            history_messages = self.messages[-CHAT_HISTORY_DEPTH:] if CHAT_HISTORY_DEPTH > 0 else []
            messages = [
                {"role": "system", "content": system_prompt},
                *history_messages,
                {"role": "user", "content": user_prompt},
            ]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            raw_answer = response.choices[0].message.content or ""
            answer = raw_answer
            if use_rag:
                parsed_answer = self._parse_structured_rag_answer(raw_answer)
                answer = parsed_answer["answer"]
                sources_by_normalized_title = {
                    self._normalize_source_title(source): source
                    for source in available_sources
                }
                sources = []
                seen_used_sources = set()
                for source in parsed_answer["used_sources"]:
                    matched_source = sources_by_normalized_title.get(
                        self._normalize_source_title(source)
                    )
                    if matched_source in available_sources and matched_source not in seen_used_sources:
                        sources.append(available_sources[matched_source])
                        seen_used_sources.add(matched_source)
                if not sources and fallback_source in available_sources:
                    sources = [available_sources[fallback_source]]

            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": answer})

            # Возвращаем JSON вместо чистого текста
            return json.dumps(
                {"answer": answer, "sources": sources, "debug_chunks": debug_chunks},
                ensure_ascii=False,
            )

        except Exception as e:
            return json.dumps({"answer": f"Ошибка при обработке запроса: {e}", "sources": [], "debug_chunks": []})

    @staticmethod
    def _parse_structured_rag_answer(raw_answer: str) -> dict[str, list[str] | str]:
        cleaned = raw_answer.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"answer": raw_answer, "used_sources": []}

        if not isinstance(payload, dict):
            return {"answer": raw_answer, "used_sources": []}

        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = raw_answer

        used_sources = payload.get("used_sources", [])
        if not isinstance(used_sources, list):
            used_sources = []

        normalized_sources = [
            source.strip()
            for source in used_sources
            if isinstance(source, str) and source.strip()
        ]

        return {"answer": answer, "used_sources": normalized_sources}

    @staticmethod
    def _normalize_source_title(title: str) -> str:
        return " ".join(title.replace("«", "\"").replace("»", "\"").lower().split())


if __name__ == "__main__":
    agent = MedicalAgent()
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ("exit", "quit"): break
        # Фронтенд должен уметь парсить этот JSON
        print(agent.generate_response(user_input))
