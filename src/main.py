import os
import argparse
from typing import Optional, List

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt


DEFAULT_DATA_PATH = "data/reviews.csv"
MODEL_PATH = "sentiment_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
CONF_MATRIX_PATH = "confusion_matrix.png"


def load_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Загружает датасет для обучения.

    Ожидаемый формат файла:
        text,label
        "текст отзыва",positive

    Если путь не указан или файл не найден, используется небольшой встроенный демо-набор.
    """
    target_path = csv_path or DEFAULT_DATA_PATH

    if target_path is not None and os.path.isfile(target_path):
        print(f"[INFO] Загружаю данные из файла: {target_path}")
        df = pd.read_csv(target_path)
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
    texts: List[str],
    labels: List[str],
    test_size: float = 0.25,
    random_state: int = 42,
):
    """
    Обучает baseline-модель для классификации тональности.
    Возвращает обученный векторизатор, модель, метрики и разбиение.
    """

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

    # Логистическая регрессия как сильный baseline
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

    # Сохраняем confusion matrix в файл
    print(f"[INFO] Сохраняю confusion matrix в {CONF_MATRIX_PATH}...")
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion matrix")
    plt.savefig(CONF_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    return vectorizer, model, acc, report


def save_artifacts(vectorizer, model) -> None:
    """
    Сохраняет модель и векторизатор на диск.
    """
    print(f"[INFO] Сохраняю модель в {MODEL_PATH} и векторизатор в {VECTORIZER_PATH}...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)


def load_artifacts():
    """
    Загружает модель и векторизатор с диска.
    """
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Модель или векторизатор не найдены. Сначала запусти обучение без --predict."
        )
    print("[INFO] Загружаю модель и векторизатор с диска...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return vectorizer, model


def predict_text(vectorizer, model, text: str) -> str:
    """
    Делает предсказание для одного текста.
    """
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline sentiment classifier on text data (Python + scikit-learn)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Путь к CSV-файлу с данными (колонки 'text' и 'label'). "
             "Если не указан, по умолчанию используется data/reviews.csv."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Доля тестовой выборки (по умолчанию 0.25).",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Если указан текст, модель загрузится с диска и сделает предсказание только для него.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Режим предсказания для одного текста
    if args.predict is not None:
        vectorizer, model = load_artifacts()
        label = predict_text(vectorizer, model, args.predict)
        print(f"[PREDICT] Текст: {args.predict!r}")
        print(f"[PREDICT] Предсказанный класс: {label}")
        return

    # Обычный режим: обучение модели
    df = load_data(args.data)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    print(f"[INFO] Количество объектов: {len(df)}")
    print(f"[INFO] Уникальные классы: {sorted(set(labels))}")

    vectorizer, model, acc, report = train_model(
        texts,
        labels,
        test_size=args.test_size,
    )

    print("\n[RESULT] Accuracy на тестовой выборке: {:.4f}".format(acc))
    print("\n[RESULT] Classification report:")
    print(report)

    save_artifacts(vectorizer, model)

    # Демонстрация предсказаний
    demo_examples = [
        "Отличный сервис, буду заказывать ещё",
        "Все было очень плохо, я разочарован",
        "Нормально, но без восторга",
    ]

    print("\n[INFO] Примеры предсказаний на демо-фразах:")
    for text in demo_examples:
        label = predict_text(vectorizer, model, text)
        print(f"  Текст: {text!r}")
        print(f"  Предсказанный класс: {label}")
        print("-" * 40)


if __name__ == "__main__":
    main()
