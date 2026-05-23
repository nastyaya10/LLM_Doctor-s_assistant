import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Импортируем вашего агента и судью
from src.services import MedicalAgent, MedicalRAGRetriever, load_prompt
from src.evaluation import LLMJudge


def run_evaluation(csv_path: str, output_path: str, limit: int = None):
    print(f"Загрузка датасета: {csv_path}")
    df = pd.read_csv(csv_path)

    # Очищаем данные, если есть пустые МПО
    df = df.dropna(subset=['Запрос врача', 'Минимально приемлемый ответ'])

    if limit:
        df = df.head(limit)

    agent = MedicalAgent()
    judge = LLMJudge()

    results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Оценка RAG"):
        question = row['Запрос врача']
        ground_truth = row['Минимально приемлемый ответ']

        # 1. Генерируем ответ нашим текущим RAG-пайплайном
        # (Предполагается, что generate_response возвращает JSON строку, как мы обсуждали ранее)
        response_json = agent.generate_response(question)

        try:
            parsed_response = json.loads(response_json)
            generated_answer = parsed_response.get("answer", "")
        except json.JSONDecodeError:
            generated_answer = response_json  # fallback если вернулся просто текст

        # 2. Судья оценивает результат
        eval_result = judge.evaluate_correctness(question, generated_answer, ground_truth)

        # 3. Сохраняем метрику
        results.append({
            "id": row.get('ID', index),
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "score": eval_result.get("score", 0),
            "reasoning": eval_result.get("reasoning", "")
        })

    # Сохраняем результаты в новый CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # Считаем среднюю оценку
    mean_score = results_df[results_df['score'] > 0]['score'].mean()
    print(f"\nЭвалюация завершена! Средний балл (Correctness): {mean_score:.2f} / 5.0")
    print(f"Результаты сохранены в: {output_path}")


if __name__ == "__main__":
    DATASET_PATH = str(PROJECT_ROOT / "data" / "eval" / "Error_Types_Yandex_Dataset.xlsx - заполн.csv")
    OUTPUT_PATH = str(PROJECT_ROOT / "data" / "eval" / "evaluation_results.csv")

    run_evaluation(DATASET_PATH, OUTPUT_PATH, limit=5)