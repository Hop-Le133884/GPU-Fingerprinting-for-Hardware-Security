#!/usr/bin/env python3
"""Train ML classifiers for GPU fingerprinting.

Loads the feature CSV produced by feature_extraction.py, trains four classifiers
(Random Forest, SVM, XGBoost, Decision Tree) with an 80/20 stratified split,
and saves the trained models plus the held-out test set for evaluation.

Usage:
    python ml_classification.py [--features features.csv] [--models-dir models]
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_features(features_path: str):
    """Return X (feature matrix), y (encoded labels), label encoder, and feature names."""
    df = pd.read_csv(features_path)

    if "gpu_label" not in df.columns:
        raise ValueError("'gpu_label' column not found in features file.")

    feature_cols = [c for c in df.columns if c != "gpu_label"]
    X = df[feature_cols].to_numpy(dtype=float)
    raw_labels = df["gpu_label"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(raw_labels)

    return X, y, le, feature_cols


def train_and_save(
    clf_name: str,
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_dir: str,
) -> str:
    """Fit a classifier and persist it; return the saved file path."""
    print(f"  Training {clf_name} ...")
    clf.fit(X_train, y_train)
    model_path = os.path.join(models_dir, f"{clf_name}.joblib")
    joblib.dump(clf, model_path)
    print(f"  Saved → {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GPU fingerprinting classifiers."
    )
    parser.add_argument(
        "--features",
        type=str,
        default="features.csv",
        help="Path to the features CSV produced by feature_extraction.py (default: features.csv)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save trained model files (default: models/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.features):
        raise FileNotFoundError(
            f"Features file '{args.features}' not found. "
            "Run feature_extraction.py first."
        )

    os.makedirs(args.models_dir, exist_ok=True)

    print(f"Loading features from '{args.features}' ...")
    X, y, le, feature_cols = load_features(args.features)
    print(f"  {X.shape[0]} samples  {X.shape[1]} features  {len(le.classes_)} classes")
    print(f"  Classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"  Train: {len(X_train)} samples  "
        f"Test:  {len(X_test)} samples  "
        f"(split {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)})"
    )

    # Persist test split and metadata so evaluation.py can reload them
    test_data_path = os.path.join(args.models_dir, "test_data.npz")
    np.savez(test_data_path, X_test=X_test, y_test=y_test)
    joblib.dump(le, os.path.join(args.models_dir, "label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(args.models_dir, "feature_cols.joblib"))
    print(f"  Test data saved → {test_data_path}")

    print("\nTraining classifiers ...")
    for name, clf in CLASSIFIERS.items():
        train_and_save(name, clf, X_train, y_train, args.models_dir)

    print(f"\nAll models saved to '{args.models_dir}/'.")


if __name__ == "__main__":
    main()
