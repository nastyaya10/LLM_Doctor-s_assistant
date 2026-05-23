import os
import sys
import time
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
from src.services.router import need_rag  # Импортируем наш маршрутизатор

# Загрузка переменных окружения
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("API_KEY")
folder_id = os.getenv("FOLDER_ID")
base_url = os.getenv("BASE_URL", "https://ai.api.cloud.yandex.net/v1")
model_name = os.getenv("MODEL", "yandexgpt/rc")
log_rag_context = os.getenv("LOG_RAG_CONTEXT", "1").lower() not in ("0", "false", "no")

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
        print("[Agent] Инициализация RAG-агента с маршрутизатором.")
        self.messages = []

    def generate_response(self, user_message: str) -> str:
        if not user_message.strip():
            return "Пожалуйста, введите текст вопроса."

        # Решаем, нужен ли RAG
        use_rag = need_rag(user_message)
        print(f"[Agent] Запрос: {user_message[:50]}... | Режим: {'RAG' if use_rag else 'CHAT'}")

        try:
            if use_rag:
                # Пайплайн RAG
                retrieval_result = get_retriever().retrieve(user_message)
                context = get_context_builder().build_context(retrieval_result)

                if not context:
                    return "Информации по вопросу в базе не найдено."

                # Загружаем системный промпт для RAG
                system_prompt = load_prompt("system_prompt")
                
                # Загружаем шаблон юзер-промпта и форматируем его с контекстом
                user_prompt_template = load_prompt("user_prompt_rag")
                user_prompt = user_prompt_template.format(
                    context=context,
                    question=user_message
                )
            else:
                # Прямой ответ LLM (режим обычного чата)
                
                # Загружаем отдельный системный промпт для обычного общения
                system_prompt = load_prompt("system_prompt_chat")
                user_prompt = user_message

            messages = [
                {"role": "system", "content": system_prompt},
                *self.messages[-CHAT_HISTORY_DEPTH:] if CHAT_HISTORY_DEPTH > 0 else [],  # Берем только последние сообщения для контекста сессии
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
            return answer

        except Exception as e:
            return f"Ошибка при обработке запроса: {e}"


if __name__ == "__main__":
    agent = MedicalAgent()
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ("exit", "quit"): break
        print(agent.generate_response(user_input))
