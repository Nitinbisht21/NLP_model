# Product Requirements Document: Sentiment Analysis System

**Project Name:** SentiMind (working title — feel free to rename)
**Author:** Nitin Bisht
**Version:** 1.0
**Date:** July 14, 2026
**Status:** Planning

---

## 1. Overview

### 1.1 Problem Statement
Reviews and social posts contain valuable opinion data, but manually reading through hundreds/thousands of them to gauge overall sentiment is slow and error-prone. This project builds an ML system that automatically classifies text as positive, negative, or neutral sentiment.

### 1.2 Objective
Build and deploy a content-based sentiment analysis system that classifies text sentiment using classical ML and deep learning approaches, with a comparative analysis and a live interactive demo.

### 1.3 Success Criteria
- Working model with ≥85% accuracy on test set (IMDB) or ≥75% (Twitter, noisier data)
- Deployed, publicly accessible demo (Streamlit/FastAPI)
- Clean GitHub repo with README, following the same documentation quality as CineMatch
- At least one LinkedIn post showcasing the project

---

## 2. Scope

### 2.1 In Scope
- Binary or 3-class sentiment classification (positive/negative, or positive/negative/neutral)
- Text preprocessing pipeline (cleaning, tokenization, lemmatization)
- Two modeling approaches for comparison: classical ML (TF-IDF + Logistic Regression/Naive Bayes) and deep learning (LSTM or similar)
- Model evaluation and comparison
- Web-based demo interface
- Deployment to a public URL

### 2.2 Out of Scope (v1)
- Multi-language sentiment analysis
- Real-time streaming/live Twitter API ingestion
- Aspect-based sentiment analysis (sentiment per topic within a review)
- Transformer fine-tuning (BERT etc.) — reserved as a possible v2 stretch goal

---

## 3. Dataset

| Option | Pros | Cons |
|---|---|---|
| IMDB Movie Reviews (Kaggle, 50K reviews) | Clean, well-labeled, ties into CineMatch domain | Long-form text only |
| Twitter Sentiment140 (1.6M tweets) | Short text, more "real-world" noisy data | Noisier, needs more cleaning |

**Recommendation:** Start with IMDB for the clean baseline, optionally extend to Twitter data as a stretch goal to show robustness across text lengths.

---

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | System shall accept raw text input from a user via web interface |
| FR2 | System shall preprocess input (lowercase, remove punctuation/stopwords, lemmatize) |
| FR3 | System shall output a sentiment label (positive/negative[/neutral]) with confidence score |
| FR4 | System shall display a comparison of classical ML vs deep learning model predictions |
| FR5 | System shall support batch upload (CSV) for bulk sentiment scoring |
| FR6 | System shall visualize results (word cloud, sentiment distribution chart) |

---

## 5. Technical Approach

### 5.1 Pipeline
1. **Data ingestion** — load dataset, initial exploration (class balance, text length distribution)
2. **Preprocessing** — tokenization, stopword removal, lemmatization (spaCy/NLTK)
3. **Feature extraction** — TF-IDF vectorization (baseline); Word2Vec/GloVe embeddings (for DL model)
4. **Modeling** —
   - Baseline: Logistic Regression / Multinomial Naive Bayes on TF-IDF
   - Advanced: LSTM or Bi-LSTM on word embeddings
5. **Evaluation** — accuracy, precision/recall/F1, confusion matrix; compare both models
6. **Deployment** — Streamlit app (fast, matches your CineMatch experience) or FastAPI (new skill, better for "production" positioning)

### 5.2 Tech Stack
- **Language:** Python
- **NLP:** NLTK / spaCy
- **ML:** scikit-learn (TF-IDF, Logistic Regression, Naive Bayes)
- **DL:** TensorFlow/Keras or PyTorch (LSTM)
- **Deployment:** FastAPI (backend API with `/predict` and `/batch` endpoints, auto-generated Swagger docs at `/docs`)
- **Visualization:** Matplotlib/Seaborn, WordCloud

### 5.3 Hardware
Your RTX 4050 (16GB DDR5) is sufficient for LSTM training on this dataset size — no cloud GPU needed for v1.

---

## 6. Milestones

| Phase | Deliverable | Est. Time |
|---|---|---|
| 1 | Dataset exploration + preprocessing pipeline | 2-3 days |
| 2 | Baseline model (TF-IDF + Logistic Regression) | 2 days |
| 3 | Deep learning model (LSTM) + comparison | 3-4 days |
| 4 | FastAPI backend (predict + batch endpoints) + simple frontend + visualizations | 2-3 days |
| 5 | Deployment + README + GitHub polish | 1-2 days |
| 6 | LinkedIn post + documentation | 1 day |

**Total estimate:** ~2 weeks part-time

---

## 7. Risks & Open Questions
- **Risk:** LSTM training time/tuning could take longer than expected on first attempt — mitigate by starting with a small embedding dimension and few epochs, then scaling up.
- **Open question:** IMDB only, or also add Twitter data for a "generalization" story in the README?
- **Decided:** Using FastAPI for the backend — better engineering credibility, auto-generated `/docs`, and a new skill beyond CineMatch's Streamlit deployment.

---

## 8. Future Enhancements (v2+)
- Fine-tune a transformer model (DistilBERT) and compare against LSTM
- Aspect-based sentiment (e.g., "acting was great but plot was weak")
- Real-time Twitter/Reddit API integration
- Multi-language support
