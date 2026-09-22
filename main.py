"""
main.py - End-to-End Execution Pipeline
DecodeLabs Project 2: Data Classification Using AI
"""

import os
import json
import numpy as np
from src.data_loader import load_dataset, get_dataset_profile, split_and_scale_data
from src.model import KNNClassifierModel, find_optimal_k, train_benchmark_models
from src.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_elbow_curve,
    plot_decision_boundaries,
    plot_benchmark_comparison,
)
from src.predict import IrisPredictor


def run_pipeline():
    print("=" * 70)
    print("  DECODELABS INDUSTRIAL TRAINING KIT - PROJECT 2: DATA CLASSIFICATION")
    print("=" * 70)

    # 1. INPUT PHASE
    print("\n[PHASE 1: INPUT] Loading Iris Benchmark Dataset...")
    X, y = load_dataset()
    profile = get_dataset_profile(X, y)
    print(f"-> Total Samples: {profile['num_samples']}")
    print(f"-> Dimensions (Features): {profile['num_features']} {profile['feature_names']}")
    print(f"-> Classes: {profile['target_names']} (Balanced: {profile['is_balanced']})")
    print(f"-> Class Distribution: {profile['class_distribution']}")

    # 2. PROCESS PHASE
    print("\n[PHASE 2: PROCESS] Feature Scaling and Stratified Split...")
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    model_path = os.path.join(output_dir, "knn_model.joblib")

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size=0.20, random_state=42, stratify=True, save_scaler_path=scaler_path
    )
    print(f"-> Training Set: {X_train_scaled.shape[0]} samples (80%)")
    print(f"-> Test Set: {X_test_scaled.shape[0]} samples (20%)")
    print("-> StandardScaler applied (mean=0, variance=1 across features).")

    print("\n[PHASE 2.1: HYPERPARAMETER TUNING] Finding Optimal K via Elbow Method...")
    elbow_results = find_optimal_k(X_train_scaled, y_train, X_test_scaled, y_test, max_k=20)
    optimal_k = elbow_results["optimal_k"]
    print(f"-> Evaluated K=1 to K=20.")
    print(f"-> Optimal K (Elbow Point): {optimal_k} (Test Error Rate: {elbow_results['min_error_rate']:.4f})")

    print(f"\n[PHASE 2.2: MODEL TRAINING] Fitting KNN Model (K={optimal_k})...")
    knn_model = KNNClassifierModel(n_neighbors=optimal_k)
    knn_model.fit(X_train_scaled, y_train)
    knn_model.save(model_path)
    print("-> Model fitted and saved to disk.")

    print("\n[PHASE 2.3: BENCHMARK COMPARISON] Training Alternative Supervised Algorithms...")
    benchmark_res = train_benchmark_models(X_train_scaled, y_train, X_test_scaled, y_test, knn_k=optimal_k)
    for model_name, res in benchmark_res.items():
        print(f"   * {model_name:<32} | Accuracy: {res['accuracy']*100:.2f}% | F1-Score: {res['f1_score']*100:.2f}%")

    # 3. OUTPUT PHASE
    print("\n[PHASE 3: OUTPUT & DIAGNOSTICS] Evaluating KNN Test Performance...")
    y_pred = knn_model.predict(X_test_scaled)
    metrics = compute_classification_metrics(y_test, y_pred)

    print(f"-> Overall Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"-> Weighted F1-Score:     {metrics['weighted_f1_score'] * 100:.2f}%")
    print(f"-> Macro F1-Score:        {metrics['macro_f1_score'] * 100:.2f}%\n")
    print("Classification Report:")
    print(metrics["classification_report_text"])

    print("\nDetailed Per-Class Diagnostic Matrix:")
    for cls_name, cls_m in metrics["per_class_metrics"].items():
        print(
            f"   * {cls_name.capitalize():<10} | TP={cls_m['true_positive']:<2} FP={cls_m['false_positive']:<2} "
            f"FN={cls_m['false_negative']:<2} TN={cls_m['true_negative']:<2} | "
            f"Prec={cls_m['precision']*100:.1f}% Rec={cls_m['recall']*100:.1f}% F1={cls_m['f1_score']*100:.1f}%"
        )

    # Generate Visual Artifacts
    print("\n[GENERATING VISUAL ARTIFACTS] Saving high-resolution plots to outputs/...")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    elbow_path = os.path.join(output_dir, "elbow_curve.png")
    boundary_path = os.path.join(output_dir, "decision_boundary.png")
    benchmark_path = os.path.join(output_dir, "benchmark_comparison.png")
    metrics_json_path = os.path.join(output_dir, "metrics.json")

    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), cm_path)
    plot_elbow_curve(elbow_results, elbow_path)
    plot_decision_boundaries(knn_model.model, X_train_scaled, y_train, boundary_path)
    plot_benchmark_comparison(benchmark_res, benchmark_path)

    # Save metrics JSON
    with open(metrics_json_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "optimal_k": optimal_k,
                "elbow_data": {
                    "k_range": elbow_results["k_range"],
                    "test_errors": elbow_results["test_errors"],
                    "test_accuracies": elbow_results["test_accuracies"],
                },
            },
            f,
            indent=2,
        )

    print(f"-> Saved: {cm_path}")
    print(f"-> Saved: {elbow_path}")
    print(f"-> Saved: {boundary_path}")
    print(f"-> Saved: {benchmark_path}")
    print(f"-> Saved: {metrics_json_path}")

    # 4. SAMPLE INFERENCE TEST
    print("\n[PHASE 4: LIVE INFERENCE TEST] Testing Unseen Flower Sample...")
    predictor = IrisPredictor(model_path=model_path, scaler_path=scaler_path)
    sample_res = predictor.predict_sample(
        sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2
    )
    print(f"-> Sample Input: Sepal 5.1 x 3.5, Petal 1.4 x 0.2")
    print(f"-> Predicted Species: {sample_res['predicted_species'].upper()} (Confidence: {sample_res['confidence']*100:.1f}%)")
    print(f"-> Class Probabilities: {sample_res['class_probabilities']}")
    print("=" * 70)
    print(" Pipeline Execution Complete! Ready for Interactive Web Dashboard & Evaluation.")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
