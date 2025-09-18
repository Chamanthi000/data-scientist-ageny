from __future__ import annotations
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import joblib
from ..utils.io import ensure_dir

app = FastAPI(title="Agentic Data Scientist API")

MODEL_PATH = Path('examples/output/model.joblib')

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_PATH.exists():
        return JSONResponse({"error": "Model not trained yet. Run the pipeline first."}, status_code=400)
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(file.file)
    proba = model.predict_proba(df)[:, 1]
    return {"pred_proba": proba.tolist()}
