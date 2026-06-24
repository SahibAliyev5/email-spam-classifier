import os
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


def get_tfidf_vectorizer() -> TfidfVectorizer:
    """Create the TF-IDF vectorizer used by most models."""
    return TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=50000,
        min_df=2,
        max_df=0.95,
    )


def get_count_vectorizer() -> CountVectorizer:
    """Create a binary count vectorizer for Bernoulli Naive Bayes."""
    return CountVectorizer(
        stop_words="english",
        lowercase=True,
        binary=True,
        max_features=50000,
    )


def get_data(train_percent: float, seed: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load, clean, and split the email spam dataset."""
    data_path = os.path.join("data", "combined_data.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Download the dataset and place it there."
        )

    data = pd.read_csv(data_path)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = data.dropna(subset=["text", "label"])
    data = data.drop_duplicates(subset=["text"])

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_percent,
        stratify=y,
        random_state=seed,
    )

    return X_train, X_test, y_train, y_test
