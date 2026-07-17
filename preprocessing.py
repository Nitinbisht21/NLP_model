"""
preprocessing.py
Text cleaning and preprocessing pipeline for the sentiment analysis project.

Steps: lowercase -> remove HTML/punctuation/numbers -> tokenize -> remove stopwords

Usage:
    from preprocessing import clean_text, preprocess_series

    clean = clean_text("This movie was AMAZING!! <br/> 10/10 would watch again.")
    df["clean_text"] = preprocess_series(df["text"])
"""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)

HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z]+")


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Cleans a single piece of text:
    1. Lowercase
    2. Strip HTML tags (common in scraped reviews, e.g. IMDB's <br/>)
    3. Remove punctuation and numbers
    4. Tokenize
    5. Remove stopwords (optional)
    """
    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = TOKEN_RE.findall(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    return " ".join(tokens)


def preprocess_series(text_series, remove_stopwords: bool = True):
    """Applies clean_text() to a pandas Series of raw text. Returns a new Series."""
    return text_series.apply(lambda t: clean_text(t, remove_stopwords))


if __name__ == "__main__":
    sample = "This movie was AMAZING!! <br/> The acting was great, 10/10 would watch again."
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
