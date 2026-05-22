import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_KEY = os.getenv("API_KEY")
FOLDER_ID = os.getenv("FOLDER_ID")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")  # yandexgpt/rc

PROMPT_PATH = Path("../prompts/router.txt")


def _load_prompt_template() -> str:
    """Загружает шаблон промпта из файла или возвращает fallback."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def need_rag(query: str, pseudo_answer: str = None) -> bool:
    """
    True — нужно делать retrieval по медицинским документам,
    False — достаточно ответа LLM без поиска.
    Решение принимает YandexGPT (промпт из файла + быстрый rule-based отсев).
    """

    # Загружаем и форматируем промпт
    template = _load_prompt_template()
    prompt = template.format(
        query=query,
        pseudo_answer=pseudo_answer if pseudo_answer else "отсутствует"
    )

    # Вызов YandexGPT
    url = f"{BASE_URL.rstrip('/')}/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Api-Key {API_KEY}",
        "x-folder-id": FOLDER_ID,
        "Content-Type": "application/json"
    }
    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/{MODEL}",
        "completionOptions": {
            "stream": False,
            "temperature": 0.0,
            "maxTokens": 10
        },
        "messages": [{"role": "user", "text": prompt}]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        answer = resp.json()["result"]["alternatives"][0]["message"]["text"].strip().lower()
        return answer == "yes"
    except Exception:
        # При любой ошибке безопаснее включить поиск
        return True
