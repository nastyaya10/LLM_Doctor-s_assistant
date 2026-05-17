import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
model = "gpt-4o-mini"

if not api_key:
    print("Ошибка: переменная OPENAI_API_KEY не найдена.")
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60.0,
    max_retries=0
)


def load_markdown(file_path: str) -> str:
    """Загружает Markdown-файл и возвращает его содержимое."""
    print(f"[DEBUG] Загрузка файла: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"[DEBUG] Файл загружен, длина: {len(content)} символов")
        return content
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        sys.exit(1)


def build_system_prompt(md_content: str) -> str:
    if len(md_content) > 30_000:
        print("[WARN] Файл очень большой, возможна медленная работа или ошибка контекста.")
    return (
        "Ты – ИИ-ассистент, который отвечает на вопросы пользователя "
        "ИСКЛЮЧИТЕЛЬНО на основе содержимого предоставленного Markdown-файла.\n"
        "Ты не имеешь права использовать свои внешние знания или додумывать что-либо.\n"
        "Если ответа нет в файле, честно скажи, что в файле такой информации нет.\n\n"
        f"Содержимое файла:\n{md_content}"
    )


class MedicalAgent:
    """RAG-агент, отвечающий только по содержимому MD-файла."""

    def __init__(self, file_path: str = None):
        if file_path is None:
            file_path = os.getenv("MD_FILE", "sample.md")
        self.file_path = file_path
        print(f"[Agent] Инициализация с файлом: {file_path}")
        md_content = load_markdown(file_path)
        self.system_prompt = build_system_prompt(md_content)
        self.messages = [{"role": "system", "content": self.system_prompt}]
        print("[Agent] Агент готов к работе.")

    def generate_response(self, user_message: str) -> str:
        """Принимает вопрос, отправляет в LLM и возвращает ответ."""
        if not user_message.strip():
            return "Пожалуйста, введите текст вопроса."

        self.messages.append({"role": "user", "content": user_message})
        print(f"[Agent] Запрос: {user_message[:80]}...")

        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=self.messages,
                temperature=0.0,
            )
            elapsed = time.time() - start_time
            print(f"[Agent] Ответ получен за {elapsed:.2f} сек.")
            answer = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[Agent] Ошибка через {elapsed:.2f} сек: {e}")
            # Убираем последний вопрос, чтобы не засорять историю
            self.messages.pop()
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
