import os
import argparse
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def load_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Загружает датасет для обучения.

    Ожидаемый формат файла data.csv:
        text,label
        "текст отзыва",positive
        "другой текст",negative

    Если путь не указан или файл не найден, используется небольшой встроенный демо-набор.
    """
    if csv_path is not None and os.path.isfile(csv_path):
        print(f"[INFO] Загружаю данные из файла: {csv_path}")
        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV-файл должен содержать колонки 'text' и 'label'")
        return df

    print("[WARN] Файл с данными не найден. Использую встроенный небольшой демо-набор.")
    data = {
        "text": [
            "Очень понравился сервис, все быстро и вежливо",
            "Ужасный опыт, больше никогда не буду пользоваться",
            "Все было нормально, но могли бы сделать быстрее",
            "Отличное качество и быстрая доставка",
            "Полная ерунда, трата времени и денег",
            "Хороший продукт за свою цену",
            "Поддержка не отвечает, ничего не работает",
            "Все супер, остался доволен",
        ],
        "label": [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "positive",
            "negative",
            "positive",
        ],
    }
    return pd.DataFrame(data)


def train_model(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.25,
    random_state: int = 42,
):
    """
    Обучает baseline-модель для классификации тональности.
    Возвращает обученный векторизатор, модель и метрики.
    """

    # Для маленького демо-дата лучше не использовать stratify,
    # чтобы избежать ошибок "least populated class..."
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
    )

    # TF-IDF векторизация текста
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True,
    )

    print("[INFO] Строю TF-IDF представление...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Логистическая регрессия как простой и сильный baseline
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )

    print("[INFO] Обучаю модель LogisticRegression...")
    model.fit(X_train_vec, y_train)

    print("[INFO] Оцениваю качество на тестовой выборке...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return vectorizer, model, acc, report


def predict_example(vectorizer, model, examples: list[str]):
    """
    Делает предсказания для списка текстов и печатает результат.
    """
    if not examples:
        return

    print("\n[INFO] Примеры предсказаний:")
    X = vectorizer.transform(examples)
    preds = model.predict(X)
    for text, label in zip(examples, preds):
        print(f"  Текст: {text!r}")
        print(f"  Предсказанный класс: {label}")
        print("-" * 40)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline sentiment classifier on text data (Python + scikit-learn)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Путь к CSV-файлу с данными (колонки 'text' и 'label'). "
             "Если не указан или файл не найден, используется встроенный демо-набор.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Доля тестовой выборки (по умолчанию 0.25).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Загружаем данные
    df = load_data(args.data)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    print(f"[INFO] Количество объектов: {len(df)}")
    print(f"[INFO] Уникальные классы: {sorted(set(labels))}")

    # 2. Обучаем baseline-модель
    vectorizer, model, acc, report = train_model(
        texts,
        labels,
        test_size=args.test_size,
    )

    # 3. Печатаем метрики
    print("\n[RESULT] Accuracy на тестовой выборке: {:.4f}".format(acc))
    print("\n[RESULT] Classification report:")
    print(report)

    # 4. Делаем несколько демонстрационных предсказаний
    demo_examples = [
        "Отличный сервис, буду заказывать ещё",
        "Все было очень плохо, я разочарован",
        "Нормально, но без восторга",
    ]
    predict_example(vectorizer, model, demo_examples)


if __name__ == "__main__":
    main()
