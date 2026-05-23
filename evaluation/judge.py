import json
import os
from openai import OpenAI
from src.config import LLM_BASE_URL
from src.evaluation.eval_prompts import JUDGE_CORRECTNESS_PROMPT


class LLMJudge:
    def __init__(self):
        # Используем те же доступы, что и в основном приложении
        api_key = os.getenv("API_KEY")
        folder_id = os.getenv("FOLDER_ID")
        model_name = os.getenv("MODEL", "yandexgpt/rc")

        self.model = f"gpt://{folder_id}/{model_name}" if folder_id else model_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=LLM_BASE_URL,
            project=folder_id,
            timeout=60.0
        )

    def evaluate_correctness(self, question: str, generated_answer: str, ground_truth: str) -> dict:
        user_content = (
            f"ВОПРОС: {question}\n\n"
            f"ЭТАЛОН (МПО): {ground_truth}\n\n"
            f"СГЕНЕРИРОВАННЫЙ ОТВЕТ: {generated_answer}"
        )

        messages = [
            {"role": "system", "content": JUDGE_CORRECTNESS_PROMPT},
            {"role": "user", "content": user_content}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,  # Судья должен быть детерминированным (temperature=0)
            )

            raw_result = response.choices[0].message.content

            # Парсим JSON-ответ от судьи. Убираем markdown-разметку, если модель ее добавила
            cleaned_result = raw_result.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_result)

        except Exception as e:
            print(f"Ошибка при оценке: {e}")
            return {"reasoning": f"Error: {str(e)}", "score": 0}
