"""
web/app.py - FastAPI Interactive Web Server
DecodeLabs Project 2: Data Classification Using AI
"""

import os
import sys
from typing import Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset, get_dataset_profile, split_and_scale_data, TARGET_NAMES
from src.model import KNNClassifierModel, find_optimal_k, train_benchmark_models
from src.evaluate import compute_classification_metrics
from src.predict import IrisPredictor

app = FastAPI(
    title="DecodeLabs Project 2 - AI Data Classification",
    description="Interactive Web Interface for Supervised Learning & KNN Classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(__file__)
static_dir = os.path.join(base_dir, "static")
template_dir = os.path.join(base_dir, "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=template_dir)

# Initialize Predictor
output_dir = os.path.join(os.path.dirname(base_dir), "outputs")
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "knn_model.joblib")
scaler_path = os.path.join(output_dir, "scaler.joblib")

# Pre-train baseline if artifacts do not exist yet
X_full, y_full = load_dataset()
X_tr_s, X_te_s, y_tr, y_te, _ = split_and_scale_data(
    X_full, y_full, test_size=0.20, random_state=42, save_scaler_path=scaler_path
)
baseline_knn = KNNClassifierModel(n_neighbors=5)
baseline_knn.fit(X_tr_s, y_tr)
baseline_knn.save(model_path)

predictor = IrisPredictor(model_path=model_path, scaler_path=scaler_path)


class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class TrainRequest(BaseModel):
    k_neighbors: int = 5
    test_size: float = 0.20


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/dataset-info")
def get_dataset_info() -> Dict[str, Any]:
    X, y = load_dataset()
    profile = get_dataset_profile(X, y)
    sample_rows = []
    for i in range(min(10, len(X))):
        row = X.iloc[i].to_dict()
        row["target"] = int(y.iloc[i])
        row["species"] = TARGET_NAMES[row["target"]]
        sample_rows.append(row)
    
    return {
        "profile": profile,
        "sample_data": sample_rows,
    }


@app.get("/api/elbow-analysis")
def get_elbow_analysis(max_k: int = 20) -> Dict[str, Any]:
    X, y = load_dataset()
    X_tr_s, X_te_s, y_tr, y_te, _ = split_and_scale_data(X, y, test_size=0.20, random_state=42)
    return find_optimal_k(X_tr_s, y_tr, X_te_s, y_te, max_k=max_k)


@app.post("/api/train-evaluate")
def train_and_evaluate(req: TrainRequest) -> Dict[str, Any]:
    X, y = load_dataset()
    X_tr_s, X_te_s, y_tr, y_te, scaler = split_and_scale_data(
        X, y, test_size=req.test_size, random_state=42, save_scaler_path=scaler_path
    )
    
    knn = KNNClassifierModel(n_neighbors=req.k_neighbors)
    knn.fit(X_tr_s, y_tr)
    knn.save(model_path)
    
    # Reload global predictor
    predictor._load_artifacts()

    y_pred = knn.predict(X_te_s)
    metrics = compute_classification_metrics(y_te, y_pred)
    benchmarks = train_benchmark_models(X_tr_s, y_tr, X_te_s, y_te, knn_k=req.k_neighbors)
    
    clean_benchmarks = {
        name: {"accuracy": b["accuracy"], "f1_score": b["f1_score"]}
        for name, b in benchmarks.items()
    }

    return {
        "k_used": req.k_neighbors,
        "test_samples_count": len(y_te),
        "train_samples_count": len(y_tr),
        "metrics": metrics,
        "benchmarks": clean_benchmarks,
    }


@app.post("/api/predict")
def predict_flower(req: PredictRequest) -> Dict[str, Any]:
    return predictor.predict_sample(
        sepal_length=req.sepal_length,
        sepal_width=req.sepal_width,
        petal_length=req.petal_length,
        petal_width=req.petal_width,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
