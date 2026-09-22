"""
test_model.py - Unit tests for KNN model, optimal K selection, and benchmarks
"""

import pytest
import numpy as np
from src.data_loader import load_dataset, split_and_scale_data
from src.model import KNNClassifierModel, find_optimal_k, train_benchmark_models
from src.evaluate import compute_classification_metrics


@pytest.fixture
def dataset_splits():
    X, y = load_dataset()
    return split_and_scale_data(X, y, test_size=0.20, random_state=42, stratify=True)


def test_knn_fit_predict(dataset_splits):
    X_train_scaled, X_test_scaled, y_train, y_test, _ = dataset_splits
    
    knn = KNNClassifierModel(n_neighbors=5)
    assert not knn.is_fitted
    knn.fit(X_train_scaled, y_train)
    assert knn.is_fitted

    preds = knn.predict(X_test_scaled)
    assert len(preds) == len(y_test)
    
    # Check accuracy is above standard benchmark threshold (> 90%)
    acc = np.mean(preds == y_test)
    assert acc >= 0.90


def test_find_optimal_k(dataset_splits):
    X_train_scaled, X_test_scaled, y_train, y_test, _ = dataset_splits
    res = find_optimal_k(X_train_scaled, y_train, X_test_scaled, y_test, max_k=15)
    
    assert "optimal_k" in res
    assert 1 <= res["optimal_k"] <= 15
    assert len(res["k_range"]) == 15
    assert len(res["test_errors"]) == 15
    assert len(res["test_accuracies"]) == 15


def test_train_benchmark_models(dataset_splits):
    X_train_scaled, X_test_scaled, y_train, y_test, _ = dataset_splits
    benchmarks = train_benchmark_models(X_train_scaled, y_train, X_test_scaled, y_test, knn_k=5)
    
    assert len(benchmarks) >= 4
    for model_name, data in benchmarks.items():
        assert "accuracy" in data
        assert "f1_score" in data
        assert data["accuracy"] >= 0.85
        assert data["f1_score"] >= 0.85


def test_classification_metrics(dataset_splits):
    X_train_scaled, X_test_scaled, y_train, y_test, _ = dataset_splits
    knn = KNNClassifierModel(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    preds = knn.predict(X_test_scaled)

    metrics = compute_classification_metrics(y_test, preds)
    assert "accuracy" in metrics
    assert "weighted_f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert "per_class_metrics" in metrics
    assert len(metrics["per_class_metrics"]) == 3
    for name, pcm in metrics["per_class_metrics"].items():
        assert pcm["true_positive"] + pcm["false_negative"] == 10  # 10 test samples per class
