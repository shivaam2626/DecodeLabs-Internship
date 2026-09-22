"""
data_loader.py - Data Ingestion, Profiling, Scaling, and Splitting
DecodeLabs Project 2: Data Classification Using AI
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

FEATURE_SHORT_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def load_dataset(csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the Iris dataset from scikit-learn and caches/saves to CSV if needed.
    Returns:
        df_features: DataFrame containing the 4 measurements.
        target: Series containing the numeric class target (0, 1, 2).
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X = df[FEATURE_NAMES] if all(c in df.columns for c in FEATURE_NAMES) else df[FEATURE_SHORT_NAMES]
        if "target" in df.columns:
            y = df["target"]
        elif "species" in df.columns:
            mapping = {name: i for i, name in enumerate(TARGET_NAMES)}
            y = df["species"].map(mapping)
        else:
            raise ValueError("Target column not found in CSV.")
        return X, y

    # Load from scikit-learn standard dataset
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    
    # Ensure data directory exists and save copy
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_file = os.path.join(data_dir, "iris.csv")
    if not os.path.exists(csv_file):
        df_export = df.copy()
        df_export["species_name"] = df_export["target"].map(lambda idx: TARGET_NAMES[idx])
        df_export.to_csv(csv_file, index=False)

    X = df[iris.feature_names]
    y = df["target"]
    return X, y


def get_dataset_profile(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Generates summary statistics and data profile adhering to the IPO blueprint.
    """
    df_full = X.copy()
    df_full["target"] = y
    df_full["species"] = y.map(lambda i: TARGET_NAMES[i])

    class_counts = df_full["species"].value_counts().to_dict()
    stats = X.describe().to_dict()

    return {
        "num_samples": len(X),
        "num_features": X.shape[1],
        "feature_names": list(X.columns),
        "target_names": TARGET_NAMES,
        "class_distribution": class_counts,
        "summary_statistics": stats,
        "is_balanced": len(set(class_counts.values())) == 1,
    }


def split_and_scale_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
    save_scaler_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Executes the Process step of the IPO framework:
    1. Random & Stratified Train-Test Split (e.g. 80% Train, 20% Test)
    2. Feature Scaling via StandardScaler (Mean = 0, Variance = 1)
       Note: Fit on train set, transform on train and test to prevent data leakage.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    stratify_target = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_target,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump(scaler, save_scaler_path)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.array(y_train),
        y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.array(y_test),
        scaler,
    )
