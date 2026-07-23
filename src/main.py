"""
main.py — Phase 4: FastAPI Backend

Serves the trained sentiment model (Logistic Regression + TF-IDF from Phase 2)
behind a REST API with two endpoints:

  POST /predict  -> single text prediction
  POST /batch    -> multiple texts at once (CSV upload)
  GET  /docs     -> auto-generated interactive Swagger UI (built into FastAPI)

Run locally with:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs in your browser to test it interactively.
"""

import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import io

from preprocessing import clean_text

app = FastAPI(
    title="Movie Review Sentiment API",
    description="Classifies movie reviews as positive or negative using TF-IDF + Logistic Regression.",
    version="1.0",
)

# Allow the frontend (served from the same app, or opened separately) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the simple frontend at /app (e.g. http://127.0.0.1:8000/app/index.html)
app.mount("/app", StaticFiles(directory="static", html=True), name="static")

# ---- Load the trained model + vectorizer once at startup (not on every request) ----
# This avoids the huge overhead of reloading a 377KB+ file on every API call.
try:
    model = joblib.load("logistic_regression_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    model = None
    vectorizer = None


# ---- Request/response schemas ----
# Pydantic models define exactly what shape of JSON the API expects/returns.
# FastAPI uses these to auto-validate incoming requests and auto-generate docs.
class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    text: str
    sentiment: str
    confidence: float


def predict_single(text: str) -> PredictionOutput:
    """Shared prediction logic used by both /predict and /batch."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])  # transform ONLY — never re-fit at inference time
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = float(max(proba))
    return PredictionOutput(text=text, sentiment=pred, confidence=round(confidence, 4))


@app.get("/")
def root():
    """Simple health check — confirms the API is running and the model loaded correctly."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "message": "Go to /docs for interactive API testing",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    """
    Predicts sentiment for a single piece of text.

    Example request body:
        { "text": "This movie was absolutely wonderful!" }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model.py first to generate it.")
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    return predict_single(input.text)


@app.post("/batch")
async def batch_predict(file: UploadFile = File(...)):
    """
    Predicts sentiment for many reviews at once via CSV upload.
    The CSV must have a column named 'text' or 'review'.

    Returns a list of predictions, one per row.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model.py first to generate it.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Accept either 'text' or 'review' as the column name
    text_col = "text" if "text" in df.columns else "review" if "review" in df.columns else None
    if text_col is None:
        raise HTTPException(status_code=400, detail="CSV must have a 'text' or 'review' column.")

    results = [predict_single(str(t)) for t in df[text_col]]
    return {"count": len(results), "predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
