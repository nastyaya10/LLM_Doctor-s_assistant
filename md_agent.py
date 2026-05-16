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

# Добавляем таймаут для всех HTTP-запросов (в секундах)
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60.0,  # <-- общий таймаут на запрос
    max_retries=0  # без повторных попыток, чтобы не ждать лишнего
)


def load_markdown(file_path: str) -> str:
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
    # Если файл слишком большой, предупреждаем
    if len(md_content) > 30_000:
        print("[WARN] Файл очень большой, возможна медленная работа или ошибка контекста.")
    return (
        "Ты – ИИ-ассистент, который отвечает на вопросы пользователя "
        "ИСКЛЮЧИТЕЛЬНО на основе содержимого предоставленного Markdown-файла.\n"
        "Ты не имеешь права использовать свои внешние знания или додумывать что-либо.\n"
        "Если ответа нет в файле, честно скажи, что в файле такой информации нет.\n\n"
        f"Содержимое файла:\n{md_content}"
    )


def main():
    file_path = os.getenv("MD_FILE", "sample.md")
    print(f"Агент запущен, читает: {file_path}\n")

    md_content = load_markdown(file_path)
    system_prompt = build_system_prompt(md_content)
    messages = [{"role": "system", "content": system_prompt}]

    print("Задавайте вопросы (exit для выхода).\n")

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

        messages.append({"role": "user", "content": user_input})

        print("[DEBUG] Отправка запроса к LLM...", end="", flush=True)
        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            elapsed = time.time() - start_time
            print(f" ответ получен за {elapsed:.2f} сек.")

            answer = response.choices[0].message.content
            print(answer)
            messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            elapsed = time.time() - start_time
            print(f" ошибка через {elapsed:.2f} сек.")
            print(f"Ошибка при обращении к LLM: {e}")
            messages.pop()  # убираем последний вопрос


if __name__ == "__main__":
    main()
