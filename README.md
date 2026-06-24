# Email Spam Classifier

Email Spam Classifier is a beginner-friendly Natural Language Processing (NLP) project that classifies email messages as either **spam** or **not spam** using classical machine learning models.

The project uses text vectorization techniques such as **TF-IDF** and **binary bag-of-words**, then compares several supervised learning algorithms for spam detection.

## Project Overview

The goal of this project is to build a practical machine learning pipeline for email spam classification without using LLM APIs or deep learning models.

The pipeline follows this process:

```text
Raw email text
→ Text vectorization
→ Machine learning classifier
→ Spam / not-spam prediction
```

## Dataset

This project uses the [Email Spam Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset/data) from Kaggle. The dataset contains 83,446 labeled email records classified as spam or not spam, and was created by combining the 2007 TREC Public Spam Corpus and the Enron-Spam Dataset.

After downloading the dataset, place the CSV file at:

```text
data/combined_data.csv
```

## Models Used

This project compares the following models:

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

Logistic Regression also performed very strongly and has the advantage of supporting probability estimates through `predict_proba()`.

## Project Structure

```text
email-spam-classifier/
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── src/
│   ├── input.py
│   ├── train.py
│   └── predict.py
├── README.md
├── requirements.txt
└── .gitignore
```

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

Place the dataset file here:

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
Linear_SVM:
Prediction: 0
Decision score: -0.2628
Positive score means: 1
Negative score means: 0

Logistic_Regression:
Prediction: 1
0: 17.90%
1: 82.10%

Multinomial_NB:
Prediction: 1
0: 15.23%
1: 84.77%

Bernoulli_NB:
Prediction: 1
0: 0.00%
1: 100.00%

Complement_NB:
Prediction: 1
0: 16.63%
1: 83.37%
```

For models such as Linear SVM, probability estimates are not directly available, so the script prints a decision score instead.

## Key Concepts

### TF-IDF

TF-IDF converts text into numerical features by giving higher scores to words that are important in a specific email but not too common across all emails.

### Naive Bayes

Naive Bayes models estimate the probability of an email being spam based on the words it contains.

### Logistic Regression

Logistic Regression learns weights for words and predicts the probability that an email belongs to the spam class.

### Linear SVM

Linear SVM finds a separating boundary between spam and not-spam emails with the largest possible margin.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib
