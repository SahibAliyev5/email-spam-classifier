from input import get_data, get_count_vectorizer, get_tfidf_vectorizer

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib
import os


SEED = 420
TRAIN_PERCENT = 0.85
VALIDATION_PERCENT = 0.01

X_train, X_val, X_test, y_train, y_val, y_test = get_data(
    train_percent=TRAIN_PERCENT,
    validation_percent=VALIDATION_PERCENT,
    seed=SEED
)


models = {
    "Multinomial_NB": Pipeline([
        ("vectorizer", get_tfidf_vectorizer()),
        ("classifier", MultinomialNB())
    ]),

    "Complement_NB": Pipeline([
        ("vectorizer", get_tfidf_vectorizer()),
        ("classifier", ComplementNB())
    ]),

    "Bernoulli_NB": Pipeline([
        ("vectorizer", get_count_vectorizer()),
        ("classifier", BernoulliNB())
    ]),

    "Logistic_Regression": Pipeline([
        ("vectorizer", get_tfidf_vectorizer()),
        ("classifier", LogisticRegression(random_state=SEED, max_iter=1000))
    ]),

    "Linear_SVM": Pipeline([
        ("vectorizer", get_tfidf_vectorizer()),
        ("classifier", LinearSVC(random_state=SEED))
    ])
}

def train_models():
    os.makedirs("models", exist_ok=True)

    for name, pipeline in models.items():
        print(name + ":")

        pipeline.fit(X_train, y_train)
        
        y_predict = pipeline.predict(X_val)

        print(confusion_matrix(y_val, y_predict))
        print(accuracy_score(y_val, y_predict))
        print(classification_report(y_val, y_predict))

        joblib.dump(pipeline, os.path.join("models", name + ".pkl"))


def main():
    train_models()

if __name__ == '__main__':
    main()