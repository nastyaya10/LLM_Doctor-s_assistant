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
            sources = []
            if use_rag:
                retrieval_result = get_retriever().retrieve(user_message)
                
                # top_chunks уже отсортированы реранкером и ограничены в retrieve() до TOP_R
                top_chunks = retrieval_result.results
                
                if not top_chunks:
                    return json.dumps({"answer": "Информации по вопросу в базе не найдено.", "sources": []})

                context = get_context_builder().build_context(top_chunks)
                
                # Собираем источники в JSON-структуру для фронтенда
                sources = [{"value": c.text[:200] + "...", "source": c.title} for c in top_chunks]
                
                system_prompt = load_prompt("system_prompt")
                user_prompt = load_prompt("user_prompt_rag").format(
                    context=context, 
                    question=user_message
                )
            else:
                system_prompt = load_prompt("system_prompt_chat")
                user_prompt = user_message

            messages = [
                {"role": "system", "content": system_prompt},
                *self.messages[-CHAT_HISTORY_DEPTH:] if CHAT_HISTORY_DEPTH > 0 else [],
                {"role": "user", "content": user_prompt},
            ]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            answer = response.choices[0].message.content

            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": answer})

            # Возвращаем JSON вместо чистого текста
            return json.dumps({"answer": answer, "sources": sources}, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"answer": f"Ошибка при обработке запроса: {e}", "sources": []})


if __name__ == "__main__":
    agent = MedicalAgent()
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ("exit", "quit"): break
        # Фронтенд должен уметь парсить этот JSON
        print(agent.generate_response(user_input))
