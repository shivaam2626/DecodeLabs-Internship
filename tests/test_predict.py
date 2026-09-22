"""
test_predict.py - Unit tests for inference predictor
"""

import os
import pytest
from src.data_loader import load_dataset, split_and_scale_data
from src.model import KNNClassifierModel
from src.predict import IrisPredictor


@pytest.fixture(scope="module", autouse=True)
def setup_artifacts(tmp_path_factory):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    model_path = os.path.join(output_dir, "knn_model.joblib")

    X, y = load_dataset()
    X_train_scaled, _, y_train, _, _ = split_and_scale_data(
        X, y, test_size=0.20, random_state=42, save_scaler_path=scaler_path
    )
    knn = KNNClassifierModel(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn.save(model_path)


def test_predict_sample():
    predictor = IrisPredictor()
    assert predictor.is_ready()

    # Known Setosa sample: Sepal 5.1 x 3.5, Petal 1.4 x 0.2
    res = predictor.predict_sample(5.1, 3.5, 1.4, 0.2)
    assert res["predicted_species"] == "setosa"
    assert res["predicted_class_index"] == 0
    assert res["confidence"] > 0.8
    assert "class_probabilities" in res
    assert "nearest_neighbor_distances" in res

    # Known Virginica sample: Sepal 6.9 x 3.1, Petal 5.4 x 2.1
    res2 = predictor.predict_sample(6.9, 3.1, 5.4, 2.1)
    assert res2["predicted_species"] == "virginica"
    assert res2["predicted_class_index"] == 2


def test_predict_batch():
    predictor = IrisPredictor()
    samples = [
        [5.1, 3.5, 1.4, 0.2],
        [5.9, 3.0, 4.2, 1.5],
        [6.9, 3.1, 5.4, 2.1],
    ]
    batch_res = predictor.predict_batch(samples)
    assert len(batch_res) == 3
    species_list = [r["predicted_species"] for r in batch_res]
    assert species_list == ["setosa", "versicolor", "virginica"]
