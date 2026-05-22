import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from src.services import load_prompt

# 1. Определяем корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    LLM_BASE_URL,
    LLM_MODEL_NAME,
    LOG_RAG_CONTEXT
)
from src.services.context_builder import RetrievalContextBuilder
from src.services.medical_rag_retriever import MedicalRAGRetriever

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("API_KEY")
folder_id = os.getenv("FOLDER_ID")

model = f"gpt://{folder_id}/{LLM_MODEL_NAME}" if folder_id else LLM_MODEL_NAME

if not api_key:
    print("Ошибка: переменная API_KEY не найдена.")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL, project=folder_id, timeout=60.0, max_retries=0)

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
        # Загружаем промпты из файлов
        self.system_prompt = load_prompt("system_prompt")
        self.rag_template = load_prompt("user_rag_template")
        self.messages = []

    def generate_response(self, user_message: str) -> str:
        if not user_message.strip():
            return "Пожалуйста, введите текст вопроса."

        start_time = time.time()
        try:
            retrieval_result = get_retriever().retrieve(user_message)
            context = get_context_builder().build_context(retrieval_result)

            if not context:
                return "В базе знаний не найдено релевантных фрагментов."

            # Формируем промпт через шаблон
            user_prompt = self.rag_template.format(context=context, user_message=user_message)

            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.messages,
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
            return f"Ошибка при обращении к LLM: {e}"

# Оставляем консольный режим, если запустить md_agent.py напрямую
if __name__ == "__main__":
    agent = MedicalAgent()
    print("\nЗадавайте вопросы (exit для выхода).\n")
    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit"):
            print("Завершение работы.")
            break
        if not user_input:
            continue
        answer = agent.generate_response(user_input)
        print(answer)
