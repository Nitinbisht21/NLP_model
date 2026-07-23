import time
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_loader import load_imdb_csv, split_data
from src.preprocessing import preprocess_series

# ---- Config ----
MAX_VOCAB_SIZE = 10000    # only consider the top 10,000 most frequent words
MAX_SEQUENCE_LEN = 200    # pad/truncate every review to 200 tokens
EMBEDDING_DIM = 100       # each word becomes a 100-number dense vector
LSTM_UNITS = 64           # size of the LSTM's internal memory
EPOCHS = 5
BATCH_SIZE = 64


def prepare_sequences(X_train, X_test):
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")

    return X_train_pad, X_test_pad, tokenizer


def build_lstm_model(vocab_size: int) -> Sequential:
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LEN),
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        loss="binary_crossentropy",   # standard loss for binary classification
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading dataset...")
    df = load_imdb_csv("IMDB Dataset.csv")
    print(f"Loaded {len(df)} reviews")

    print("\nCleaning text...")
    t0 = time.time()
    df["clean_text"] = preprocess_series(df["text"])
    print(f"Cleaning done in {time.time() - t0:.1f}s")

    print("\nSplitting into train/test...")
    df_for_split = df[["clean_text", "label"]].rename(columns={"clean_text": "text"})
    X_train, X_test, y_train, y_test = split_data(df_for_split)

    # Convert labels to 0/1 for the neural network
    y_train_bin = (y_train == "positive").astype(int).values
    y_test_bin = (y_test == "positive").astype(int).values

    print("\nTokenizing and padding sequences...")
    X_train_pad, X_test_pad, tokenizer = prepare_sequences(X_train, X_test)
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    print(f"Vocabulary size used: {vocab_size}")

    print("\nBuilding LSTM model...")
    model = build_lstm_model(vocab_size)
    model.summary()

    print("\nTraining (this is the slow part — GPU on your RTX 4050 will help a lot here)...")
    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    t0 = time.time()
    history = model.fit(
        X_train_pad, y_train_bin,
        validation_split=0.1,     # hold out 10% of train data to monitor overfitting live
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    print(f"Training done in {time.time() - t0:.1f}s")

    print("\nEvaluating on test set...")
    probs = model.predict(X_test_pad, batch_size=BATCH_SIZE)
    preds = (probs > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test_bin, preds)
    print(f"\n{'=' * 55}")
    print(f"LSTM — Test Accuracy: {acc:.4f}")
    print("=" * 55)
    print(classification_report(y_test_bin, preds, target_names=["negative", "positive"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test_bin, preds))

    # Save model + tokenizer for later use (e.g., comparing in FastAPI app)
    model.save("lstm_sentiment_model.keras")
    import pickle
    with open("lstm_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("\nModel saved to lstm_sentiment_model.keras")
    print("Tokenizer saved to lstm_tokenizer.pkl")

    print(f"\n--- Comparison reminder ---")
    print(f"Phase 2 Logistic Regression: 89.35% (see model.py results)")
    print(f"Phase 3 LSTM: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
