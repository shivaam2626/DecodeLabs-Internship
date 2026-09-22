"""
test_data_loader.py - Unit tests for data loading, profiling, scaling, and splitting
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_loader import (
    load_dataset,
    get_dataset_profile,
    split_and_scale_data,
    TARGET_NAMES,
    FEATURE_NAMES,
)


def test_load_dataset():
    X, y = load_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 150
    assert len(y) == 150
    assert X.shape[1] == 4
    assert set(y.unique()) == {0, 1, 2}


def test_get_dataset_profile():
    X, y = load_dataset()
    profile = get_dataset_profile(X, y)
    assert profile["num_samples"] == 150
    assert profile["num_features"] == 4
    assert profile["is_balanced"] is True
    assert profile["target_names"] == TARGET_NAMES
    assert profile["class_distribution"] == {"setosa": 50, "versicolor": 50, "virginica": 50}


def test_split_and_scale_data():
    X, y = load_dataset()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size=0.20, random_state=42, stratify=True
    )
    
    # 80/20 split check
    assert len(X_train_scaled) == 120
    assert len(X_test_scaled) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30

    # Stratified balance check
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    assert np.all(counts_train == 40)
    assert np.all(counts_test == 10)

    # StandardScaler verification: mean ~ 0, variance ~ 1 on train
    train_mean = np.mean(X_train_scaled, axis=0)
    train_std = np.std(X_train_scaled, axis=0)
    np.testing.assert_allclose(train_mean, np.zeros(4), atol=1e-7)
    np.testing.assert_allclose(train_std, np.ones(4), atol=1e-7)
