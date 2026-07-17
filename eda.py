from data_loader import load_imdb_csv
from preprocessing import preprocess_series


def run_eda(df):
    """Print simple information about the dataset."""
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print("Total reviews:", len(df))

    print("\nSentiment count:")
    print(df["label"].value_counts())

    df["word_count"] = df["text"].apply(count_words)

    print("\nReview length details:")
    print(df["word_count"].describe())

    print("\nCleaning first 200 reviews...")
    cleaned_reviews = preprocess_series(df["text"].head(500))
    all_words = " ".join(cleaned_reviews).split()
    unique_words = set(all_words)

    print("Unique words in first 200 cleaned reviews:", len(unique_words))


def count_words(text):
    """Count words in one review."""
    return len(text.split())


if __name__ == "__main__":
    df = load_imdb_csv()
    run_eda(df)