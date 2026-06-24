import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib
import os

SEED = 420
TRAIN_PERCENT = 0.85
VALIDATION_PERCENT = 0.01

def get_data() -> tuple:
    data = pd.read_csv(os.path.join('data', 'combined_data.csv'))

    data = data.dropna(subset=["text", "label"])
    data = data.drop_duplicates(subset=["text"])

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_PERCENT, stratify=y, random_state=SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_PERCENT,
        stratify=y_train, random_state=SEED
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = get_data()

Tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_features=50000,
    min_df=2,
    max_df=0.95
)
Count_vectorizer = CountVectorizer(
    stop_words="english", 
    binary=True, 
    max_features=50000
)



bayes_models = {
    ("Multinomial NB", Tfidf_vectorizer, MultinomialNB()),
    ("Complement NB", Tfidf_vectorizer, ComplementNB()),
    ("Bernoulli NB", Count_vectorizer, BernoulliNB())
}

for name, vectorizer, model in bayes_models:
    X_train_vector = vectorizer.fit_transform(X_train)
    X_val_vector = vectorizer.transform(X_val)
    X_test_vector = vectorizer.transform(X_test)
    print(name + ":")

    model.fit(X_train_vector, y_train)
    y_predict = model.predict(X_test_vector)

    print(confusion_matrix(y_test, y_predict))
    print(accuracy_score(y_test, y_predict))
    print(classification_report(y_test, y_predict))

    joblib.dump(model, os.path.join('models', name + '.pkl'))
    joblib.dump(vectorizer, os.path.join('models', name + ' vectorizer.pkl'))
