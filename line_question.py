import argparse
import json
import os
import ssl
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import certifi
except ImportError:
    certifi = None


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEXT_PATH = PROJECT_DIR / "line_notes.md"
DEFAULT_ENV_PATH = PROJECT_DIR / ".env"
DEFAULT_LLM_BASE_URL = "https://api.groq.com/openai"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"


SYSTEM_PROMPT = (
    "Ты ИИ-ассистент, который отвечает на вопросы пользователя "
    "исключительно на основе содержимого предоставленного Markdown-файла. "
    "Ты не имеешь права использовать внешние знания или додумывать что-либо. "
    "Если ответа нет в файле, честно скажи, что в файле такой информации нет."
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_payload(question: str, markdown_text: str, model: str) -> dict:
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Markdown-файл:\n"
                    f"{markdown_text}\n\n"
                    f"Вопрос пользователя: {question}"
                ),
            },
        ],
    }


def send_llm_request(base_url: str, api_key: str, payload: dict) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "project-1-course-markdown-qa/0.1",
        },
    )

    ssl_context = None
    if certifi is not None:
        ssl_context = ssl.create_default_context(cafile=certifi.where())

    with urlopen(request, timeout=60, context=ssl_context) as response:
        response_body = response.read().decode("utf-8")
        return json.loads(response_body)


def extract_llm_content(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask LLM questions using only a local Markdown file."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_TEXT_PATH,
        help="Markdown file used as the only answer source.",
    )
    parser.add_argument(
        "--print-payload",
        action="store_true",
        help="Print request payload without sending it.",
    )
    return parser.parse_args()


def answer_question(
    question: str,
    markdown_text: str,
    base_url: str,
    api_key: str,
    model: str,
    print_payload: bool,
) -> bool:
    payload = build_payload(question, markdown_text, model)

    if print_payload:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return True

    try:
        response = send_llm_request(base_url, api_key, payload)
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print(f"HTTP error {error.code}: {error_body}", file=sys.stderr)
        return False
    except URLError as error:
        print(f"Network error: {error.reason}", file=sys.stderr)
        return False

    print(extract_llm_content(response))
    return True


def main() -> int:
    args = parse_args()
    load_env_file(DEFAULT_ENV_PATH)

    markdown_text = read_markdown(args.file)
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)

    if not args.print_payload and (
        not api_key or api_key == "PASTE_YOUR_GROQ_KEY_HERE"
    ):
        print("Error: add GROQ_API_KEY to .env before sending the request.", file=sys.stderr)
        return 1

    print("Задайте вопрос по Markdown-файлу. Для выхода: exit или quit.")

    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            print()
            break

        if question.lower() in {"exit", "quit", "выход"}:
            break

        if not question:
            continue

        answer_question(
            question,
            markdown_text,
            base_url,
            api_key,
            model,
            args.print_payload,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
