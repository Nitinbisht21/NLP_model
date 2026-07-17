import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text):
    """Clean one movie review."""
    # 1. Make all letters lowercase.
    text = text.lower()

    # 2. Remove HTML tags like <br />.
    text = re.sub(r"<.*?>", " ", text)

    # 3. Remove numbers and punctuation.
    text = re.sub(r"[^a-z\s]", " ", text)

    # 4. Split the sentence into words.
    words = text.split()

    # 5. Remove common words like "the", "is", "and".
    clean_words = []
    for word in words:
        if word not in STOPWORDS:
            clean_words.append(word)

    # 6. Join the words back into one string.
    return " ".join(clean_words)


def preprocess_series(text_series):
    """Clean every review in a pandas column."""
    return text_series.apply(clean_text)


if __name__ == "__main__":
    sample = "This movie was AMAZING!! <br/> The acting was great, 10/10 would watch again."
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))