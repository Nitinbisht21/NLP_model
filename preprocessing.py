"""
preprocessing.py
Text cleaning and preprocessing pipeline for the sentiment analysis project.

Steps: lowercase -> remove HTML/punctuation/numbers -> tokenize -> remove stopwords -> lemmatize

Usage:
    from preprocessing import clean_text, preprocess_series

    clean = clean_text("This movie was AMAZING!! <br/> 10/10 would watch again.")
    df["clean_text"] = preprocess_series(df["text"])
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure required NLTK data is present
for pkg in ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    try:
        nltk.data.find(
            f"corpora/{pkg}" if pkg in ("stopwords", "wordnet", "omw-1.4") else f"tokenizers/{pkg}"
        )
    except LookupError:
        nltk.download(pkg)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Cleans a single piece of text:
    1. Lowercase
    2. Strip HTML tags (common in scraped reviews, e.g. IMDB's <br/>)
    3. Remove punctuation and numbers
    4. Tokenize
    5. Remove stopwords (optional)
    6. Lemmatize (optional)
    """
    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_series(text_series, remove_stopwords: bool = True, lemmatize: bool = True):
    """Applies clean_text() to a pandas Series of raw text. Returns a new Series."""
    return text_series.apply(lambda t: clean_text(t, remove_stopwords, lemmatize))


if __name__ == "__main__":
    sample = "This movie was AMAZING!! <br/> The acting was great, 10/10 would watch again."
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
