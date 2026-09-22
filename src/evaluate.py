"""
evaluate.py - Output Validation, Diagnostic Metrics, and Visualizations
DecodeLabs Project 2: Data Classification Using AI
"""

import os
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)
from src.data_loader import TARGET_NAMES


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix,
    and individual class breakdowns.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    per_class_metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        # One-vs-Rest calculation for TP, FP, FN, TN
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(np.sum(cm) - (tp + fp + fn))

        per_class_metrics[name] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1_score": float(f1[i]),
            "support": int(sup[i]),
        }

    return {
        "accuracy": acc,
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1_score": float(weighted_f1),
        "macro_f1_score": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "classification_report_text": classification_report(
            y_true, y_pred, target_names=TARGET_NAMES
        ),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    title: str = "DecodeLabs AI - Confusion Matrix",
) -> None:
    """
    Plots and saves an annotated, high-contrast confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 6), dpi=300)
    
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
        cbar=True,
        linewidths=1.5,
        linecolor="#1e293b",
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(title, fontsize=14, weight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Actual True Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_elbow_curve(
    elbow_data: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Plots Error Rate vs K-Value to illustrate:
    - K=1 (Potential Overfitting/Noise Sensitivity)
    - Optimal K (The Elbow)
    - Large K (Underfitting)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=300)

    k_range = elbow_data["k_range"]
    test_errors = elbow_data["test_errors"]
    train_errors = elbow_data["train_errors"]
    optimal_k = elbow_data["optimal_k"]

    plt.plot(
        k_range,
        test_errors,
        color="#2563eb",
        marker="o",
        markersize=7,
        linewidth=2,
        label="Test Error Rate",
    )
    plt.plot(
        k_range,
        train_errors,
        color="#94a3b8",
        linestyle="--",
        linewidth=1.5,
        label="Train Error Rate",
    )

    # Highlight optimal K
    opt_idx = k_range.index(optimal_k)
    plt.plot(
        optimal_k,
        test_errors[opt_idx],
        marker="*",
        markersize=16,
        color="#dc2626",
        label=f"Optimal K = {optimal_k}",
    )

    plt.title("Elbow Method: Tuning Parameter K vs Error Rate", fontsize=13, weight="bold")
    plt.xlabel("K-Value (Number of Nearest Neighbors)", fontsize=11)
    plt.ylabel("Error Rate (Fraction of Misclassifications)", fontsize=11)
    plt.xticks(k_range)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_decision_boundaries(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_path: str,
    feature_indices: Tuple[int, int] = (2, 3),
    feature_labels: Tuple[str, str] = ("Petal Length (scaled)", "Petal Width (scaled)"),
) -> None:
    """
    Plots 2D decision boundaries of the trained classifier over two chosen features.
    """
    from sklearn.neighbors import KNeighborsClassifier
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Train a 2D model for clean 2D surface rendering
    idx1, idx2 = feature_indices
    X_2d = X_train[:, [idx1, idx2]]
    
    clf_2d = KNeighborsClassifier(n_neighbors=getattr(model, "n_neighbors", 5))
    clf_2d.fit(X_2d, y_train)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6), dpi=300)
    palette = ["#bfdbfe", "#bbf7d0", "#fef08a"]
    scatter_colors = ["#1d4ed8", "#15803d", "#b45309"]

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(palette)

    plt.contourf(xx, yy, Z, alpha=0.6, cmap=custom_cmap)

    for cls_idx, name in enumerate(TARGET_NAMES):
        pts = X_2d[y_train == cls_idx]
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            color=scatter_colors[cls_idx],
            edgecolor="black",
            s=45,
            label=name.capitalize(),
        )

    plt.title(f"KNN Decision Boundaries (K={clf_2d.n_neighbors})", fontsize=13, weight="bold")
    plt.xlabel(feature_labels[0], fontsize=11)
    plt.ylabel(feature_labels[1], fontsize=11)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_benchmark_comparison(
    benchmark_results: Dict[str, Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Plots a comparison bar chart of classification models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    names = list(benchmark_results.keys())
    accuracies = [benchmark_results[m]["accuracy"] * 100 for m in names]
    f1_scores = [benchmark_results[m]["f1_score"] * 100 for m in names]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(x - width/2, accuracies, width, label="Accuracy (%)", color="#3b82f6")
    plt.bar(x + width/2, f1_scores, width, label="F1-Score (%)", color="#10b981")

    plt.ylabel("Score (%)", fontsize=11)
    plt.title("Algorithm Comparison Benchmark", fontsize=13, weight="bold")
    plt.xticks(x, names, rotation=15, ha="right", fontsize=10)
    plt.ylim(80, 105)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
