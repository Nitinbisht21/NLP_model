"""
eda.py
Quick exploratory data analysis: class balance, text length distribution, vocabulary size.
Run this after loading data to sanity-check it before modeling.
"""

import pandas as pd
from data_loader import load_imdb_csv
from preprocessing import preprocess_series


def run_eda(df: pd.DataFrame):
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Total documents: {len(df)}")
    print(f"\nClass balance:\n{df['label'].value_counts()}")

    df["word_count"] = df["text"].apply(lambda t: len(t.split()))
    print(f"\nText length (words) stats:\n{df['word_count'].describe()}")

    print("\nCleaning a sample and checking vocabulary size...")
    cleaned = preprocess_series(df["text"].head(200))  # sample for speed
    vocab = set(" ".join(cleaned).split())
    print(f"Vocabulary size (sample of 200 docs): {len(vocab)}")


if __name__ == "__main__":
    df = load_imdb_csv()
    run_eda(df)
