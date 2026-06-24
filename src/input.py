import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_tfidf_vectorizer() -> TfidfVectorizer:
    Tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=50000,
        min_df=2,
        max_df=0.95
    )
    return Tfidf_vectorizer

def get_count_vectorizer() -> CountVectorizer:
    Count_vectorizer = CountVectorizer(
        stop_words="english", 
        binary=True, 
        max_features=50000
    )
    return Count_vectorizer


def get_data(train_percent : float, validation_percent : float, seed : int) -> tuple:
    data = pd.read_csv(os.path.join('data', 'combined_data.csv'))

    data = data.dropna(subset=["text", "label"])
    data = data.drop_duplicates(subset=["text"])

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_percent, stratify=y, random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_percent,
        stratify=y_train, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
