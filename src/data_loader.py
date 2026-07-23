import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_FILE = "IMDB Dataset.csv"


def load_imdb_csv(file_path=DATASET_FILE):
    """Load the IMDB dataset and return only the text and label columns."""
    
    df = pd.read_csv(file_path)

    # Rename columns to simple project names.
    df = df.rename(columns={"review": "text", "sentiment": "label"})

    # Keep only the columns we need.
    return df[["text", "label"]]


def split_data(df, test_size=0.2):
    """Split the dataset into training and testing data."""

    text = df["text"]
    labels = df["label"]

    return train_test_split(text, labels, test_size=test_size, random_state=42)


if __name__ == "__main__":
    df = load_imdb_csv()

    print(f"Loaded {len(df)} documents")
    print(df["label"].value_counts())
    print(df.head())