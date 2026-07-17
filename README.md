# Sentiment Analysis - Phase 1: Data + Preprocessing

## Setup
```bash
pip install -r requirements.txt
```

## What's in here
- `data_loader.py` - loads the downloaded Kaggle IMDB 50K CSV from the project root as `IMDB Dataset.csv`
- `preprocessing.py` - cleans text: lowercase, strip HTML, remove punctuation/numbers, tokenize, remove stopwords
- `eda.py` - quick exploratory analysis: class balance, text length stats, vocab size

The code uses the downloaded IMDB dataset directly.

## Try it now
```bash
python eda.py
```
This loads `IMDB Dataset.csv` and prints class balance plus text stats.

## Dataset
1. Download `IMDB Dataset.csv` from [Kaggle: IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. Place it in the project root as `IMDB Dataset.csv`.
3. Use `load_imdb_csv()` in your scripts.

## Next phase
Phase 2: TF-IDF + Logistic Regression/Naive Bayes baseline model (see PRD).
