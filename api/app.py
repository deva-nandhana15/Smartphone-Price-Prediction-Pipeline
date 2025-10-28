"""
FastAPI app for model inference and optional retrain trigger.
Endpoints:
- GET /status
- POST /predict  (single JSON or list)
- POST /train    (optional trigger - will run train script and return queued)
- GET /metadata  (returns preprocess metadata)
"""

import os
import json
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import tensorflow as tf
import numpy as np
from .preprocess import Preprocessor

# Config paths (relative to repo)
METADATA_PATH = os.environ.get("METADATA_PATH", "../preprocess/metadata.json")
SCALER_PATH = os.environ.get("SCALER_PATH", "../preprocess/scaler.joblib")
MODEL_DIR = os.environ.get("MODEL_DIR", "../models/price_predictor")

app = FastAPI(title="Mobile Price Prediction API")

# load model (latest)
def load_latest_model(model_dir):
    if not os.path.exists(model_dir):
        return None, None
    versions = [d for d in os.listdir(model_dir) if d.isdigit()]
    if not versions:
        return None, None
    latest = str(max(map(int, versions)))
    model_path = os.path.join(model_dir, latest)
    model = tf.keras.models.load_model(model_path)
    return model, latest

MODEL, MODEL_VERSION = load_latest_model(MODEL_DIR)
PREPROCESSOR = Preprocessor(METADATA_PATH, SCALER_PATH)

class PredictRequest(BaseModel):
    # Accept arbitrary dict
    __root__: Dict[str, Any]

@app.get("/status")
def status():
    return {"model_loaded": MODEL is not None, "model_version": MODEL_VERSION}

@app.get("/metadata")
def metadata():
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="metadata not found")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

@app.post("/predict")
def predict(req: PredictRequest):
    payload = req.__root__
    # support both list and single dict (frontend may send array)
    if isinstance(payload, list):
        out = []
        for p in payload:
            vec = PREPROCESSOR.preprocess_single(p)
            if MODEL is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            pred = MODEL.predict(vec)
            out.append(float(pred.flatten()[0]))
        return {"predictions": out}
    elif isinstance(payload, dict):
        vec = PREPROCESSOR.preprocess_single(payload)
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        pred = MODEL.predict(vec)
        return {"prediction": float(pred.flatten()[0])}
    else:
        raise HTTPException(status_code=400, detail="Invalid payload")

@app.post("/train")
def train():
    """
    Trigger a retrain. This runs train.py as a subprocess. In production, you'd
    trigger a job on Kubernetes / Airflow instead.
    """
    # safety: do not run concurrent training if one already running - naive approach
    # For this example, we'll start subprocess and return 202 with pid.
    cmd = [
        "python", "../tf_training/train.py",
        "--preprocessed-parquet", "../preprocessed/phones.parquet",
        "--metadata", "../preprocess/metadata.json",
        "--scaler", "../preprocess/scaler.joblib",
        "--output-dir", "../models/price_predictor",
        "--epochs", "10",
        "--batch-size", "64"
    ]
    proc = subprocess.Popen(cmd)
    return {"status": "training_started", "pid": proc.pid}
