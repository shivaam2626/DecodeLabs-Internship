"""
model.py - Classification Models, K-Nearest Neighbors, and Hyperparameter Tuning
DecodeLabs Project 2: Data Classification Using AI
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


class KNNClassifierModel:
    """
    K-Nearest Neighbors Classifier wrapper with hyperparameter tuning,
    proximity logic, and model persistence.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "minkowski", p: int = 2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, p=p)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "KNNClassifierModel":
        """
        Fits the KNN model (memorizes feature space).
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for samples in X.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns prediction probabilities for samples in X.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        return self.model.predict_proba(X)

    def find_k_neighbors(self, X: np.ndarray, n_neighbors: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the K-neighbors of a point (distances and indices).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding neighbors.")
        k = n_neighbors or self.n_neighbors
        return self.model.kneighbors(X, n_neighbors=k)

    def save(self, file_path: str) -> None:
        """
        Saves the trained model to disk.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> "KNNClassifierModel":
        """
        Loads a trained model from disk.
        """
        return joblib.load(file_path)


def find_optimal_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_k: int = 25,
) -> Dict[str, Any]:
    """
    Computes Error Rate and Accuracy across K values (1 to max_k).
    Identifies the optimal K (Elbow method) to balance overfitting (K=1)
    and underfitting (large K).
    """
    k_range = list(range(1, min(max_k + 1, len(X_train))))
    train_errors = []
    test_errors = []
    test_accuracies = []
    f1_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)

        train_err = float(np.mean(y_train_pred != y_train))
        test_err = float(np.mean(y_test_pred != y_test))
        acc = float(accuracy_score(y_test, y_test_pred))
        f1 = float(f1_score(y_test, y_test_pred, average="weighted"))

        train_errors.append(train_err)
        test_errors.append(test_err)
        test_accuracies.append(acc)
        f1_scores.append(f1)

    # Optimal K: pick best error rate, favoring K >= 3 to avoid K=1 noise/overfitting
    min_error = min(test_errors)
    candidate_ks = [k for k, err in zip(k_range, test_errors) if err == min_error]
    # Filter for K >= 3 to follow the elbow robustness guideline, or fall back to candidates
    robust_candidates = [k for k in candidate_ks if k >= 3 and k % 2 != 0]
    if robust_candidates:
        optimal_k = robust_candidates[0]
    else:
        odd_candidates = [k for k in candidate_ks if k % 2 != 0]
        optimal_k = odd_candidates[0] if odd_candidates else candidate_ks[0]

    return {
        "k_range": list(k_range),
        "train_errors": train_errors,
        "test_errors": test_errors,
        "test_accuracies": test_accuracies,
        "f1_scores": f1_scores,
        "optimal_k": optimal_k,
        "min_error_rate": min_error,
        "best_accuracy": float(max(test_accuracies)),
    }


def train_benchmark_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    knn_k: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Trains and benchmarks multiple classification algorithms to demonstrate
    mastery of supervised learning pipelines.
    """
    from sklearn.calibration import CalibratedClassifierCV
    
    models = {
        f"K-Nearest Neighbors (K={knn_k})": KNeighborsClassifier(n_neighbors=knn_k),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50),
        "Support Vector Classifier (SVC)": CalibratedClassifierCV(SVC(random_state=42)),
    }

    benchmark_results = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))

        benchmark_results[name] = {
            "model_object": clf,
            "accuracy": acc,
            "f1_score": f1,
            "test_predictions": y_pred.tolist(),
        }

    return benchmark_results
