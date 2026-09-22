"""
predict.py - Real-Time Inference and Proximity Engine
DecodeLabs Project 2: Data Classification Using AI
"""

import os
from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd
import joblib
from src.data_loader import TARGET_NAMES, FEATURE_NAMES
from src.model import KNNClassifierModel


class IrisPredictor:
    """
    Handles inference for new sample inputs using the trained model and scaler.
    """

    def __init__(
        self,
        model_path: str = "outputs/knn_model.joblib",
        scaler_path: str = "outputs/scaler.joblib",
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model: Optional[KNNClassifierModel] = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)

    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def predict_sample(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> Dict[str, Any]:
        """
        Predicts the species for a single flower measurement tuple.
        """
        if not self.is_ready():
            raise RuntimeError("Model or Scaler not loaded. Run training first.")

        raw_features = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=FEATURE_NAMES,
        )
        scaled_features = self.scaler.transform(raw_features)

        # Prediction & Probabilities
        pred_idx = int(self.model.predict(scaled_features)[0])
        probabilities = self.model.predict_proba(scaled_features)[0]

        # Top nearest neighbors and distances
        distances, indices = self.model.find_k_neighbors(scaled_features)

        prob_breakdown = {
            TARGET_NAMES[i]: float(probabilities[i]) for i in range(len(TARGET_NAMES))
        }

        return {
            "predicted_class_index": pred_idx,
            "predicted_species": TARGET_NAMES[pred_idx],
            "confidence": float(probabilities[pred_idx]),
            "class_probabilities": prob_breakdown,
            "scaled_features": scaled_features[0].tolist(),
            "nearest_neighbor_distances": distances[0].tolist(),
            "raw_inputs": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
            },
        }

    def predict_batch(self, samples: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Batch prediction for a list of measurement 4-tuples.
        """
        results = []
        for sample in samples:
            res = self.predict_sample(sample[0], sample[1], sample[2], sample[3])
            results.append(res)
        return results
