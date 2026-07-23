import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_loader import load_imdb_csv, split_data
from src.preprocessing import preprocess_series


def build_tfidf_vectorizer(max_features: int = 10000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))


def train_logistic_regression(X_train_vec, y_train) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return model


def train_naive_bayes(X_train_vec, y_train) -> MultinomialNB:

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model


def evaluate_model(model, X_test_vec, y_test, name: str):
    """Prints accuracy, precision/recall/F1, and confusion matrix for a model."""
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print(f"\n{'=' * 55}")
    print(f"{name} — Accuracy: {acc:.4f}")
    print("=" * 55)
    print(classification_report(y_test, preds))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, preds, labels=["negative", "positive"]))

    return acc


def main():
    print("Loading dataset...")
    df = load_imdb_csv("IMDB Dataset.csv")
    print(f"Loaded {len(df)} reviews")

    print("\nCleaning text (this can take a minute on the full 50K dataset)...")
    t0 = time.time()
    df["clean_text"] = preprocess_series(df["text"])
    print(f"Cleaning done in {time.time() - t0:.1f}s")

    print("\nSplitting into train/test...")
    df_for_split = df[["clean_text", "label"]].rename(columns={"clean_text": "text"})
    X_train, X_test, y_train, y_test = split_data(df_for_split)

    print("\nVectorizing with TF-IDF...")
    vectorizer = build_tfidf_vectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)  # fit + transform on train
    X_test_vec = vectorizer.transform(X_test)         # transform ONLY on test (no fit!)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train_vec, y_train)
    lr_acc = evaluate_model(lr_model, X_test_vec, y_test, "Logistic Regression")

    print("\nTraining Naive Bayes...")
    nb_model = train_naive_bayes(X_train_vec, y_train)
    nb_acc = evaluate_model(nb_model, X_test_vec, y_test, "Naive Bayes")

    # Save the better-performing model + vectorizer for Phase 4 (FastAPI)
    best_model, best_name = (lr_model, "logistic_regression") if lr_acc >= nb_acc else (nb_model, "naive_bayes")
    joblib.dump(best_model, f"{best_name}_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print(f"\nBest model: {best_name} (accuracy={max(lr_acc, nb_acc):.4f}) — saved to {best_name}_model.pkl")
    print("Vectorizer saved to tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
