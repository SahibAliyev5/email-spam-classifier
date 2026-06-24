# Email Spam Classifier

Email Spam Classifier is a beginner-friendly Natural Language Processing (NLP) project that classifies email messages as either **spam** or **not spam** using classical machine learning models.

The project does not use LLM APIs or deep learning. Instead, it uses traditional NLP techniques such as **TF-IDF**, **binary bag-of-words**, and supervised machine learning classifiers.

## Project Overview

The goal of this project is to build a practical spam detection pipeline using Python and scikit-learn.

The pipeline follows this process:

```text
Raw email text
→ Text vectorization
→ Machine learning classifier
→ Spam / not-spam prediction
```

The trained models are saved as scikit-learn pipelines, so each saved model contains both the vectorizer and the classifier.

## Dataset

This project uses the [Email Spam Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset/data) from Kaggle.

The dataset contains **83,446 labeled email records** classified as spam or not spam. It was created by combining the **2007 TREC Public Spam Corpus** and the **Enron-Spam Dataset**.

After downloading the dataset, place the CSV file at:

```text
data/combined_data.csv
```

The dataset is not included in this repository because of file size and licensing considerations.

## Models Used

This project compares five classical machine learning models:

| Model | Vectorizer | Description |
|---|---|---|
| Multinomial Naive Bayes | TF-IDF | Strong simple baseline for text classification |
| Complement Naive Bayes | TF-IDF | Naive Bayes variant useful for imbalanced text data |
| Bernoulli Naive Bayes | Binary CountVectorizer | Uses word presence/absence instead of TF-IDF weights |
| Logistic Regression | TF-IDF | Strong linear classifier that supports probability estimates |
| Linear SVM | TF-IDF | High-performing linear classifier for sparse text data |

## Results

The dataset was split using a stratified **80/20 train-test split**.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Multinomial Naive Bayes | 97.48% | 0.97 | 0.97 |
| Complement Naive Bayes | 97.23% | 0.97 | 0.97 |
| Bernoulli Naive Bayes | 93.30% | 0.93 | 0.93 |
| Logistic Regression | 98.64% | 0.99 | 0.99 |
| Linear SVM | 99.02% | 0.99 | 0.99 |

The best-performing model was **Linear SVM**, achieving approximately **99.02% accuracy** on the test set.

Logistic Regression also performed strongly and has the additional advantage of supporting probability estimates through `predict_proba()`.

## Project Structure

```text
email-spam-classifier/
├── models/
│   ├── Bernoulli_NB.pkl
│   ├── Complement_NB.pkl
│   ├── Linear_SVM.pkl
│   ├── Logistic_Regression.pkl
│   └── Multinomial_NB.pkl
├── src/
│   ├── input.py
│   ├── train.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

The `models/` directory contains pre-trained scikit-learn pipelines. Each pipeline includes both the vectorizer and the classifier.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Add the dataset

Download the dataset from Kaggle and place it here:

```text
data/combined_data.csv
```

### 2. Train all models

```bash
python src/train.py
```

This trains all models, prints evaluation metrics, and saves the trained pipelines into the `models/` folder.

### 3. Predict a custom email

```bash
python src/predict.py
```

Example input:

```text
Congratulations! You have won a free prize. Click here now.
```

Example output:

```text
Logistic_Regression:
Prediction: 1
0: 17.90%
1: 82.10%

Linear_SVM:
Prediction: 1
Decision score: 0.8421
Positive score means: 1
Negative score means: 0
```

In this dataset:

```text
0 = not spam
1 = spam
```

Models such as Naive Bayes and Logistic Regression support probability estimates. Linear SVM does not directly provide probabilities, so the script prints a decision score instead.

## Key Concepts

### TF-IDF

TF-IDF converts text into numerical features. It gives higher values to words that are important in a specific email but not too common across all emails.

### Binary Bag-of-Words

Binary bag-of-words records whether a word appears in an email or not. This representation is used with Bernoulli Naive Bayes.

### Naive Bayes

Naive Bayes models estimate the probability of an email being spam or not spam based on the words it contains.

### Logistic Regression

Logistic Regression learns weights for words and uses them to estimate the probability that an email belongs to a class.

### Linear SVM

Linear SVM finds a separating boundary between spam and not-spam emails and tries to maximize the margin between the classes.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib
- TF-IDF
- Naive Bayes
- Logistic Regression
- Linear SVM
