#!/usr/bin/env python3
"""Train and evaluate ML classifiers on GPU fingerprinting feature vectors."""

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier


def evaluate(name: str, y_true, y_pred, classes) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    fp   = int((cm.sum(axis=0) - np.diag(cm)).sum())
    fn   = int((cm.sum(axis=1) - np.diag(cm)).sum())
    n    = len(y_true)
    nc   = len(classes)
    fpr  = fp / (n * (nc - 1)) if nc > 1 else 0.0
    fnr  = fn / n if n > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-score:   {f1:.4f}")
    print(f"  FP Rate:    {fpr:.4f}")
    print(f"  FN Rate:    {fnr:.4f}")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "fnr": fnr}


def save_confusion_matrix(name: str, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} — Confusion Matrix (5-fold CV)")
    plt.tight_layout()
    fname = f"confusion_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GPU fingerprinting classifiers.")
    parser.add_argument("--input", default="features.csv", help="Feature matrix CSV from feature_extraction.py")
    parser.add_argument("--folds", type=int, default=5, help="Cross-validation folds (default 5)")
    parser.add_argument("--save-models", action="store_true", default=True,
                        help="Save trained models to .pkl files (required for demo.py)")
    parser.add_argument("--drop-trivial", action="store_true",
                        help="Drop VRAM-size features (mem_free_mib, mem_total_mib, mem_used_mib, mem_used_ratio) "
                             "to evaluate behavioral-only fingerprinting")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = df.columns.str.strip()
    df["label"] = df["label"].str.strip()
    print(f"Loaded {len(df)} samples, {df.shape[1]-1} features")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    feature_cols = [c for c in df.columns if c != "label"]

    if args.drop_trivial:
        trivial_prefixes = ("mem_free_mib_", "mem_total_mib_", "mem_used_mib_", "mem_used_ratio_")
        dropped = [c for c in feature_cols if c.startswith(trivial_prefixes)]
        feature_cols = [c for c in feature_cols if not c.startswith(trivial_prefixes)]
        print(f"Dropped {len(dropped)} trivial VRAM-size features: {dropped}")
        print(f"Remaining features: {len(feature_cols)}\n")

    X = df[feature_cols].values
    classes = le.classes_

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    models = {
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, probability=True, random_state=42)),
        ]),
        "XGBoost": Pipeline([
            ("clf", XGBClassifier(n_estimators=200, eval_metric="mlogloss",
                                  random_state=42, verbosity=0)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(random_state=42)),
        ]),
    }

    results = []
    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=cv)
        results.append(evaluate(name, y, y_pred, classes))
        save_confusion_matrix(name, y, y_pred, classes)

        if args.save_models:
            model.fit(X, y)
            suffix = "_no_mem" if args.drop_trivial else ""
            fname = f"model_{name.lower().replace(' ', '_')}{suffix}.pkl"
            joblib.dump({"model": model, "label_encoder": le, "feature_cols": feature_cols}, fname)
            print(f"  Saved model: {fname}")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    summary = pd.DataFrame(results).set_index("model")
    print(summary.to_string(float_format="{:.4f}".format))
    summary.to_csv("classification_results.csv")
    print("\nSaved classification_results.csv")

    if args.save_models:
        rf = models["Random Forest"]
        importances = rf.named_steps["clf"].feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        print("\nTop 15 most important features (Random Forest):")
        for rank, i in enumerate(top_idx, 1):
            print(f"  {rank:2d}. {feature_cols[i]:<50} {importances[i]:.4f}")


if __name__ == "__main__":
    main()
