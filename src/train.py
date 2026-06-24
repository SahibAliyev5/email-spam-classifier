import os

import joblib
from input import get_count_vectorizer, get_data, get_tfidf_vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 420
TRAIN_PERCENT = 0.80

X_train, X_test, y_train, y_test = get_data(
    train_percent=TRAIN_PERCENT,
    seed=SEED,
)

models = {
    "Multinomial_NB": Pipeline(
        [
            ("vectorizer", get_tfidf_vectorizer()),
            ("classifier", MultinomialNB()),
        ]
    ),
    "Complement_NB": Pipeline(
        [
            ("vectorizer", get_tfidf_vectorizer()),
            ("classifier", ComplementNB()),
        ]
    ),
    "Bernoulli_NB": Pipeline(
        [
            ("vectorizer", get_count_vectorizer()),
            ("classifier", BernoulliNB()),
        ]
    ),
    "Logistic_Regression": Pipeline(
        [
            ("vectorizer", get_tfidf_vectorizer()),
            ("classifier", LogisticRegression(random_state=SEED, max_iter=1000)),
        ]
    ),
    "Linear_SVM": Pipeline(
        [
            ("vectorizer", get_tfidf_vectorizer()),
            ("classifier", LinearSVC(random_state=SEED)),
        ]
    ),
}


def train_models() -> None:
    os.makedirs("models", exist_ok=True)

    for name, pipeline in models.items():
        print(f"\n{name}:")

        pipeline.fit(X_train, y_train)
        y_predict = pipeline.predict(X_test)

        print(confusion_matrix(y_test, y_predict))
        print(f"Accuracy: {accuracy_score(y_test, y_predict):.4f}")
        print(classification_report(y_test, y_predict))

        model_path = os.path.join("models", f"{name}.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Saved model to {model_path}")


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()
