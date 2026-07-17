"""
data_loader.py
Loads sentiment data from one of two sources:
1. NLTK's built-in movie_reviews corpus (quick start, no download needed, 2000 docs)
2. Kaggle IMDB 50K Dataset CSV (main dataset for the project, download separately)

Usage:
    from data_loader import load_nltk_movie_reviews, load_imdb_csv

    df = load_nltk_movie_reviews()          # quick start, works offline
    df = load_imdb_csv("data/IMDB_Dataset.csv")   # main dataset
"""

import pandas as pd
import random


def load_nltk_movie_reviews() -> pd.DataFrame:
    """
    Loads NLTK's built-in movie_reviews corpus.
    Good for quickly testing your pipeline before the full IMDB dataset arrives.
    Returns a DataFrame with columns: ['text', 'label'] where label is 'positive'/'negative'.
    """
    import nltk
    from nltk.corpus import movie_reviews

    # Ensure corpus is available
    try:
        nltk.data.find("corpora/movie_reviews")
    except LookupError:
        nltk.download("movie_reviews")

    docs = [
        (movie_reviews.raw(fileid), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]
    random.seed(42)
    random.shuffle(docs)

    df = pd.DataFrame(docs, columns=["text", "label"])
    # NLTK uses 'pos'/'neg' -> normalize to match IMDB dataset labels
    df["label"] = df["label"].map({"pos": "positive", "neg": "negative"})
    return df


def load_imdb_csv(path: str) -> pd.DataFrame:
    """
    Loads the Kaggle "IMDB Dataset of 50K Movie Reviews" CSV.
    Download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    Place the CSV at the given path (e.g., data/IMDB_Dataset.csv).

    Expected columns in the raw CSV: 'review', 'sentiment'
    Returns a DataFrame with columns: ['text', 'label']
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    return df[["text", "label"]]


def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Simple stratified-ish train/test split (shuffled, split by fraction)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=seed, stratify=df["label"]
    )


if __name__ == "__main__":
    # Quick sanity check
    df = load_nltk_movie_reviews()
    print(f"Loaded {len(df)} documents")
    print(df["label"].value_counts())
    print(df.head(2))
