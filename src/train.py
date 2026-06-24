from input import get_data, Tfidf_vectorizer, Count_vectorizer

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
    'Multinomial_NB': (Tfidf_vectorizer, MultinomialNB()),
    'Complement_NB': (Tfidf_vectorizer, ComplementNB()),
    'Bernoulli_NB': (Count_vectorizer, BernoulliNB()),
    'Logistic_Regression': (Tfidf_vectorizer, LogisticRegression(
        random_state=SEED, max_iter=1000
    )),
}

def train_models():
    for name in models.keys():
        vectorizer, model = models[name]

        X_train_vector = vectorizer.fit_transform(X_train)
        X_test_vector = vectorizer.transform(X_test)

        print(name + ":")
        model.fit(X_train_vector, y_train)
        y_predict = model.predict(X_test_vector)

        print(confusion_matrix(y_test, y_predict))
        print(accuracy_score(y_test, y_predict))
        print(classification_report(y_test, y_predict))

        joblib.dump(model, os.path.join('models', name + '.pkl'))



def main():
    train_models()

if __name__ == '__main__':
    main()