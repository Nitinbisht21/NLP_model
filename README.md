# Sentiment Analysis — Phase 1: Data + Preprocessing

## Setup
```bash
pip install -r requirements.txt
python -m nltk.downloader movie_reviews stopwords wordnet punkt punkt_tab omw-1.4
```

## What's in here
- `src/data_loader.py` — loads data from either:
  - NLTK's built-in `movie_reviews` corpus (2000 docs, works immediately, no download) — use this to build/test your pipeline right now
  - Kaggle IMDB 50K CSV (main dataset) — download from [Kaggle: IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place at `data/IMDB_Dataset.csv`
- `src/preprocessing.py` — cleans text: lowercase, strip HTML, remove punctuation/numbers, tokenize, remove stopwords, lemmatize
- `src/eda.py` — quick exploratory analysis: class balance, text length stats, vocab size

## Try it now (no dataset download needed)
```bash
cd src
python eda.py
```
This loads the NLTK corpus and prints class balance + text stats — confirms your pipeline works before you plug in the bigger Kaggle dataset.

## Next: get the full IMDB dataset
1. Go to the Kaggle link above (you'll need a free Kaggle account)
2. Download `IMDB Dataset.csv`
3. Place it at `data/IMDB_Dataset.csv`
4. Swap `load_nltk_movie_reviews()` for `load_imdb_csv("data/IMDB_Dataset.csv")` in your scripts

## Next phase
Phase 2: TF-IDF + Logistic Regression/Naive Bayes baseline model (see PRD).
