import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 1. Определяем корень проекта (поднимаемся на 3 уровня вверх из src/services/md_agent.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.context_builder import RetrievalContextBuilder
from src.services.medical_rag_retriever import MedicalRAGRetriever
from src.services.prompt_loader import load_prompt

# Загрузка переменных окружения
load_dotenv(PROJECT_ROOT / ".env")  # Явно указываем путь к .env в корне проекта

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
    print("Ошибка: переменная API_KEY не найдена. Создайте файл .env в корне проекта.")
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    project=folder_id,
    timeout=60.0,
    max_retries=0
)


_retriever: MedicalRAGRetriever | None = None
_context_builder: RetrievalContextBuilder | None = None


def get_retriever() -> MedicalRAGRetriever:
    global _retriever
    if _retriever is None:
        print("[RAG] Инициализация retriever...")
        _retriever = MedicalRAGRetriever()
        print("[RAG] Retriever готов.")
    return _retriever


def get_context_builder() -> RetrievalContextBuilder:
    global _context_builder
    if _context_builder is None:
        print("[RAG] Загрузка чанков для сборки контекста...")
        _context_builder = RetrievalContextBuilder()
        print("[RAG] Сборщик контекста готов.")
    return _context_builder


def build_system_prompt() -> str:
    return load_prompt("system_prompt")


def log_context(context: str) -> None:
    if not log_rag_context:
        return

    print("\n[RAG] Контекст, отправленный в LLM:")
    print(context)
    print("[RAG] Конец контекста\n")


class MedicalAgent:
    """RAG-агент, отвечающий только по найденному контексту."""

    def __init__(self):
        print("[Agent] Инициализация RAG-агента.")
        self.messages = []
        print("[Agent] Агент готов к работе.")

    def generate_response(self, user_message: str) -> str:
        """Принимает вопрос, отправляет в LLM и возвращает ответ."""
        if not user_message.strip():
            return "Пожалуйста, введите текст вопроса."

        print(f"[Agent] Запрос: {user_message[:80]}...")

        start_time = time.time()
        try:
            retrieval_result = get_retriever().retrieve(user_message)
            context = get_context_builder().build_context(retrieval_result)

            if not context:
                return "В базе знаний не найдено релевантных фрагментов для ответа."

            log_context(context)

            user_prompt = (
                "Следующая информация:\n"
                f"{context}\n\n"
                "Ответь на вопрос пользователя исключительно на основе информации выше.\n"
                "Вопрос пользователя:\n"
                f"{user_message}"
            )

            messages = [
                {"role": "system", "content": build_system_prompt()},
                *self.messages,
                {"role": "user", "content": user_prompt},
            ]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            elapsed = time.time() - start_time
            print(f"[Agent] Ответ получен за {elapsed:.2f} сек.")
            answer = response.choices[0].message.content
            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[Agent] Ошибка через {elapsed:.2f} сек: {e}")
            return f"Ошибка при обращении к языковой модели: {e}"


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
