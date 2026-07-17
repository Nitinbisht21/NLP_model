"""
data_loader.py
Loads sentiment data from the downloaded Kaggle IMDB 50K Dataset CSV.

Usage:
    from data_loader import load_imdb_csv

    df = load_imdb_csv()
"""

import pandas as pd
from pathlib import Path


DEFAULT_IMDB_PATH = Path(__file__).resolve().parent / "IMDB Dataset.csv"


def load_imdb_csv(path: str | Path = DEFAULT_IMDB_PATH) -> pd.DataFrame:
    """
    Loads the Kaggle "IMDB Dataset of 50K Movie Reviews" CSV.
    Download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    Place the CSV at the given path, or in the project root as 'IMDB Dataset.csv'.

    Expected columns in the raw CSV: 'review', 'sentiment'
    Returns a DataFrame with columns: ['text', 'label']
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find IMDB dataset at {path}. "
            "Pass the CSV path explicitly or place it in the project root "
            "as 'IMDB Dataset.csv'."
        )

    df = pd.read_csv(path)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    missing_columns = {"text", "label"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    return df[["text", "label"]]


def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Simple stratified-ish train/test split (shuffled, split by fraction)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=seed, stratify=df["label"]
    )


if __name__ == "__main__":
    df = load_imdb_csv()
    print(f"Loaded {len(df)} documents")
    print(df["label"].value_counts())
    print(df.head(2))
