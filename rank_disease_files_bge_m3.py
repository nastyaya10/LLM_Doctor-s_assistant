from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DISEASE_FILES = [
    "01_ишемическая_болезнь_сердца.txt",
    "02_инсульт.txt",
    "03_хроническая_обструктивная_болезнь_легких.txt",
    "04_инфекции_нижних_дыхательных_путей.txt",
    "05_грипп.txt",
    "06_болезнь_альцгеймера_и_деменции.txt",
    "07_рак_легких.txt",
    "08_диабет.txt",
    "09_хронические_болезни_почек.txt",
    "10_диарея_кишечные_инфекции.txt",
]


def main():
    model_name = "BAAI/bge-m3"
    project_dir = Path(__file__).resolve().parent

    documents = load_documents(project_dir)
    questions = read_questions_from_console(count=3)

    print()
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding disease files...")
    document_embeddings = model.encode(
        [document["search_text"] for document in documents],
        normalize_embeddings=True,
    )

    print("Encoding questions...")
    question_embeddings = model.encode(
        questions,
        normalize_embeddings=True,
    )

    similarities = cosine_similarity(question_embeddings, document_embeddings)

    print()
    print_rankings(questions, documents, similarities)


def load_documents(project_dir):
    documents = []

    for file_name in DISEASE_FILES:
        file_path = project_dir / file_name
        text = file_path.read_text(encoding="utf-8").strip()

        documents.append(
            {
                "file_name": file_name,
                "title": make_title(file_name),
                "text": text,
                "search_text": f"Название болезни: {make_title(file_name)}.\n\n{text}",
            }
        )

    return documents


def make_title(file_name):
    name_without_number = file_name.removesuffix(".txt").split("_", maxsplit=1)[1]
    return name_without_number.replace("_", " ")


def read_questions_from_console(count):
    print(f"Введите {count} вопроса про болезни.")
    print()

    questions = []
    for index in range(1, count + 1):
        while True:
            question = input(f"Вопрос {index}: ").strip()
            if question:
                questions.append(question)
                break
            print("Вопрос не должен быть пустым. Попробуйте еще раз.")

    return questions


def print_rankings(questions, documents, similarities):
    for question_index, question in enumerate(questions):
        ranked_document_indexes = sorted(
            range(len(documents)),
            key=lambda document_index: similarities[question_index][document_index],
            reverse=True,
        )

        print("=" * 80)
        print(f"Question {question_index + 1}: {question}")
        print("Ranked files:")
        print()

        for rank, document_index in enumerate(ranked_document_indexes, start=1):
            document = documents[document_index]
            similarity = similarities[question_index][document_index]
            distance = 1 - similarity

            print(
                f"{rank:>2}. {document['file_name']} "
                f"({document['title']}) | "
                f"similarity = {similarity:.6f}, "
                f"distance = {distance:.6f}"
            )

        print()


if __name__ == "__main__":
    main()
