# Urdu News Article Classifier

Multi-class classification of Urdu news articles into 5 categories, comparing three machine learning models — all core algorithms implemented from scratch.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-93%25-brightgreen?style=flat-square)
![Models](https://img.shields.io/badge/Models-3-orange?style=flat-square)
![Articles](https://img.shields.io/badge/Dataset-1140%20Articles-lightgrey?style=flat-square)

---

## Overview

This project tackles the task of automatic Urdu news categorization — a challenging NLP problem due to Urdu's rich morphology, right-to-left script, and limited NLP tooling compared to English.

Three models were built and evaluated:

| Model | Accuracy | Implementation |
|---|---|---|
| Logistic Regression | 93% | From scratch |
| Neural Network | 93% | Pytorch|
| Naive Bayes | 91% | From scratch |

---

## Project Structure

```
urdu-news-classifier/
├── data/
│   └── all_articles_formatted.xlsx   # 1140 labeled Urdu articles
├── notebooks/
│   ├── NaiveBayes.ipynb              # NB from scratch with Laplace smoothing
│   ├── LogisticRegression.ipynb      # LR from scratch with Elastic Net
│   └── NeuralNetwork.ipynb           # MLP using Pytorch
├── src/
│   ├── analysis.py                   # EDA and preprocessing pipeline
│   └── scraping.ipynb                # Web scraping scripts
└── requirements.txt
```

---

## Dataset

- **1,140 articles** scraped from three major Urdu news sources:
  - [Express News](https://www.express.pk) — ~80 articles per category
  - [Jang](https://jang.com.pk) — ~90 articles per category
  - [Geo Urdu](https://urdu.geo.tv) — ~60 articles per category
- **5 categories:** Business, Entertainment, Science & Technology, Sports, World
- Balanced distribution with no significant class imbalance

---

## Methodology

### Preprocessing
- Punctuation, digit, and special character removal
- Urdu stopword removal via [UrduHack](https://urduhack.readthedocs.io/)
- Outlier articles removed (outside 5th–95th percentile by length)
- TF-IDF vectorisation with 12,000 features
- 80/20 train-test split

### Models

**Naive Bayes (from scratch)**
- Word frequency matrix per class with Laplace smoothing
- TF-IDF weights used in vectorised log-probability computation
- Formula: `log P(c|x) = log P(c) + X · log P(x|c)`

**Logistic Regression (from scratch)**
- Custom Bag-of-Words tokeniser
- Batch gradient descent with Elastic Net regularisation (lambda=0.0001, lr=0.1)
- One-vs-all strategy across 5 classes, 180 epochs

**Neural Network (Pytorch)**
- Content and title vectorised separately then merged via `np.hstack`
- 3-layer MLP with Adam optimiser and dropout
- SMOTE applied to handle any class imbalance
- Dynamic learning rate scheduling

---

## Results

All three models exceeded 90% accuracy. Key observations:

- **Logistic Regression** performed best despite being the simplest discriminative model, owing to its effectiveness in low-data regimes
- **Neural Network** reached 94% at peak but averaged 92% due to sensitivity to the small training set size
- **Naive Bayes** performed well given its simplistic independence assumption, though it struggled most with the Science & Technology category
- The **Science & Technology** category was consistently the hardest to classify across all models due to vocabulary overlap with Business and World articles

---

## Setup

```bash
git clone https://github.com/yourusername/urdu-news-classifier
cd urdu-news-classifier
pip install -r requirements.txt
```

### Requirements
```
pandas
numpy
scikit-learn
matplotlib
urduhack
tensorflow
keras
imbalanced-learn
openpyxl
beautifulsoup4
requests
torch
torchvision
```

---

## Key Implementation Notes

- Naive Bayes uses a vectorised NumPy dot product for log-probability scoring, making it significantly faster than a naive loop-based implementation
- Logistic Regression discards unseen words at test time rather than raising errors
- The independence assumption in Naive Bayes is violated by natural language, yet the model still achieves 91% — consistent with findings in NLP literature
- TF-IDF weights are used in place of raw counts in the Naive Bayes scoring function, which improves results empirically but deviates from strict probabilistic Multinomial NB

---

## Limitations

- Small training set (~800 samples) limits Neural Network performance; transformers like mBERT would benefit most from more data
- TF-IDF treats morphological variants of Urdu words as separate tokens — a stemmer or lemmatizer would reduce vocabulary fragmentation
- Category overlap between Science & Technology and other categories remains an open problem at this feature representation level

---

## Tech Stack

Python, NumPy, Pandas, scikit-learn, UrduHack, Matplotlib, Pytorch, imbalanced-learn, BeautifulSoup
