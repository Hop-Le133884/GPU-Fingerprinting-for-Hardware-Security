#!/usr/bin/env python3
"""Evaluate trained GPU fingerprinting classifiers.

Loads models saved by ml_classification.py and the held-out test set,
then prints per-classifier performance metrics:
  - Accuracy
  - Precision, Recall, F1-score (per class and macro average)
  - False Positive Rate and False Negative Rate (per class)
  - Confusion matrix

Usage:
    python evaluation.py [--models-dir models]
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAMES = ["RandomForest", "SVM", "XGBoost", "DecisionTree"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_test_data(models_dir: str):
    """Return (X_test, y_test, label_encoder) from persisted test split."""
    test_path = os.path.join(models_dir, "test_data.npz")
    le_path = os.path.join(models_dir, "label_encoder.joblib")

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test data not found at '{test_path}'. "
            "Run ml_classification.py first."
        )
    data = np.load(test_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    le = joblib.load(le_path)
    return X_test, y_test, le


def compute_fpr_fnr(cm: np.ndarray):
    """Return per-class False Positive Rate and False Negative Rate from a confusion matrix."""
    n_classes = cm.shape[0]
    fpr = np.zeros(n_classes)
    fnr = np.zeros(n_classes)
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr[i] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fpr, fnr


def evaluate_model(
    model_name: str, clf, X_test: np.ndarray, y_test: np.ndarray, class_names: list
) -> None:
    """Print a full evaluation report for one classifier."""
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    fpr, fnr = compute_fpr_fnr(cm)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {model_name}")
    print(sep)
    print(f"  Overall Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print()

    # Per-class metrics table
    header = (
        f"  {'Class':<22} {'Precision':>10} {'Recall':>10} "
        f"{'F1-Score':>10} {'FPR':>8} {'FNR':>8} {'Support':>8}"
    )
    print(header)
    print("  " + "-" * 80)
    for i, cls in enumerate(class_names):
        print(
            f"  {cls:<22} {precision[i]:>10.4f} {recall[i]:>10.4f} "
            f"{f1[i]:>10.4f} {fpr[i]:>8.4f} {fnr[i]:>8.4f} {support[i]:>8}"
        )

    # Macro averages
    print("  " + "-" * 80)
    print(
        f"  {'Macro Average':<22} {np.mean(precision):>10.4f} "
        f"{np.mean(recall):>10.4f} {np.mean(f1):>10.4f} "
        f"{np.mean(fpr):>8.4f} {np.mean(fnr):>8.4f}"
    )

    # Confusion matrix
    print()
    print("  Confusion Matrix (rows = true, cols = predicted):")
    col_header = "  " + " " * 24 + "".join(f"{c:>12}" for c in class_names)
    print(col_header)
    for i, cls in enumerate(class_names):
        row_str = "".join(f"{cm[i, j]:>12}" for j in range(len(class_names)))
        print(f"  {cls:<24}{row_str}")

    # sklearn's classification_report for reference
    print()
    print("  Detailed Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GPU fingerprinting classifiers."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing saved model files (default: models/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_test, y_test, le = load_test_data(args.models_dir)
    class_names = list(le.classes_)

    print(f"Loaded test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Classes: {class_names}")

    for model_name in MODEL_NAMES:
        model_path = os.path.join(args.models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            print(f"\n  SKIP: Model file '{model_path}' not found.")
            continue
        clf = joblib.load(model_path)
        evaluate_model(model_name, clf, X_test, y_test, class_names)

    print("\n" + "=" * 70)
    print("  Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
