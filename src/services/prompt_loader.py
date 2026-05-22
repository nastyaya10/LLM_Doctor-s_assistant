from pathlib import Path
from src.services.paths import PROJECT_ROOT

def load_prompt(prompt_name: str) -> str:
    path = PROJECT_ROOT / "src" / "prompts" / f"{prompt_name}.txt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        # Fallback на случай отсутствия файла
        return "Отсутствует промпт для {prompt_name}"
