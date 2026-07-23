🎬 SentiMind — Movie Review Sentiment Analysis

An end-to-end NLP system that classifies movie reviews as positive or negative, comparing a classical machine learning approach against a deep learning approach — served through a FastAPI backend with a live web interface.

Built as the second project in a portfolio sequence following CineMatch (content-based movie recommender).

📌 Overview

SentiMind takes raw, messy movie reviews (scraped from IMDB, complete with HTML tags and inconsistent formatting) and predicts sentiment using two different NLP approaches:

TF-IDF + Logistic Regression / Naive Bayes — a classical ML baseline
LSTM with word embeddings — a deep learning approach that understands word order and context

The best-performing model is deployed behind a REST API, with a simple web UI for testing reviews live.

✨ Features
🧹 Full text preprocessing pipeline (HTML stripping, cleaning, stopword removal)
📊 Exploratory data analysis (class balance, review length distribution, vocabulary size)
🤖 Two independently trained models with a side-by-side accuracy comparison
⚡ FastAPI backend with /predict (single review) and /batch (CSV upload) endpoints
📄 Auto-generated interactive API docs (Swagger UI at /docs)
🎨 Simple, themed web frontend to test predictions live
📓 A simplified Jupyter notebook version for step-by-step learning
🧠 Model Performance
Model	Approach	Accuracy
Logistic Regression	TF-IDF (1-2 grams)	89.35% ✅ (deployed)
Naive Bayes	TF-IDF (1-2 grams)	86.03%
LSTM	Word Embeddings	See lstm_model.py results

Why Logistic Regression is deployed: it matched or exceeded the deep learning approach's practical value for this dataset size, while being dramatically lighter and faster — a deliberate engineering trade-off between accuracy and deployment cost, not a limitation.

🛠️ Tech Stack
Language: Python
NLP: NLTK / scikit-learn stopwords
Classical ML: scikit-learn (TF-IDF, Logistic Regression, Naive Bayes)
Deep Learning: TensorFlow / Keras (LSTM, Embedding layers)
Backend: FastAPI, Uvicorn
Frontend: HTML, CSS, JavaScript (vanilla, no framework)
Model persistence: joblib, Keras .keras format
📂 Project Structure
NLP_model/
├── README.md
├── PRD_Sentiment_Analysis.md      # Product requirements & planning doc
├── requirements.txt
├── .gitignore
│
└── src/
    ├── data_loader.py             # Load dataset, train/test split
    ├── preprocessing.py           # Text cleaning pipeline
    ├── eda.py                     # Exploratory data analysis
    │
    ├── model.py                   # TF-IDF + Logistic Regression / Naive Bayes
    ├── lstm_model.py              # LSTM deep learning model
    │
    ├── main.py                    # FastAPI backend
    └── static/                    # Frontend (served by FastAPI)
        ├── index.html
        ├── style.css
        └── script.js
🚀 Getting Started
1. Clone the repo
bash
git clone https://github.com/Nitinbisht21/NLP_model.git
cd NLP_model/src
2. Install dependencies
bash
pip install -r ../requirements.txt
3. Get the dataset

Download IMDB Dataset of 50K Movie Reviews from Kaggle and place IMDB Dataset.csv inside the src/ folder.

4. Explore the data (optional)
bash
python eda.py
5. Train the baseline model
bash
python model.py

This generates logistic_regression_model.pkl and tfidf_vectorizer.pkl, needed by the API.

6. (Optional) Train the LSTM model
bash
python lstm_model.py
7. Run the API
bash
uvicorn main:app --reload
8. Try it out
Web UI: http://127.0.0.1:8000/app/index.html
API docs: http://127.0.0.1:8000/docs
📓 Alternative: Jupyter Notebook

For a simplified, all-in-one walkthrough (great for learning or quick experimentation), see sentiment_lstm_simple.ipynb in the repo root.

📄 License

This project is open source and available for learning purposes.

🙋 Author

Nitin Bisht — B.Tech CSE (AI & ML), Swami Rama Himalayan University.