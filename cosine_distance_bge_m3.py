from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


def main():
    model_name = "BAAI/bge-m3"
    model = SentenceTransformer(model_name)

    texts = read_texts_from_console()

    if len(texts) < 2:
        print("Нужно ввести минимум две строки для сравнения.")
        return

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    distances = cosine_distances(embeddings)
    similarities = cosine_similarity(embeddings)

    print(f"Model: {model_name}")
    print()
    print("Input strings:")
    for index, text in enumerate(texts, start=1):
        print(f"{index}. {text}")

    print()
    print("Pairwise cosine distance:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            print(
                f'"{texts[i]}" <-> "{texts[j]}": '
                f"distance = {distances[i][j]:.6f}, "
                f"similarity = {similarities[i][j]:.6f}"
            )


def read_texts_from_console():
    print("Введите строки для сравнения.")
    print("Когда закончите, нажмите Enter на пустой строке.")
    print()

    texts = []
    while True:
        text = input(f"Строка {len(texts) + 1}: ").strip()
        if not text:
            break
        texts.append(text)

    print()
    return texts



if __name__ == "__main__":
    main()
