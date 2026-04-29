#!/usr/bin/env python3
"""Statistical feature extraction for GPU fingerprinting.

Reads the three raw GPU telemetry CSV files, slices each into overlapping
30-second windows (15-second step), computes statistical features per window,
and writes a combined feature CSV ready for ML classification.

Usage:
    python feature_extraction.py [--window 30] [--step 15] [--output features.csv]
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CSV files and their human-readable GPU labels (derived from file name / gpu_name column)
GPU_CSV_FILES = [
    "NVIDIA_GeForce_RTX_3050_4GB_Laptop_GPU_gpu_telemetry.csv",
    "NVIDIA_GeForce_RTX_3060_Laptop_GPU_gpu_telemetry.csv",
    "NVIDIA_GeForce_RTX_3090_gpu_telemetry.csv",
]

# Columns that identify the device rather than capture its dynamic behaviour.
# Exclude these from statistical feature computation.
METADATA_COLS = {
    "timestamp",
    "gpu_index",
    "gpu_name",
    "gpu_uuid",
    "driver_version",
    "pci_bus_id",
    "performance_state",
}

# Statistical aggregations applied to every numeric telemetry column per window.
STATS = ["mean", "std", "var", "min", "max", "range", "skewness", "kurtosis"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_from_name(gpu_name: str) -> str:
    """Return a compact, filesystem-safe GPU label from the full NVML model name."""
    name = gpu_name.strip()
    if "3090" in name:
        return "RTX_3090"
    if "3060" in name:
        return "RTX_3060"
    if "3050" in name:
        return "RTX_3050"
    # Fallback: sanitise the raw name
    return name.replace(" ", "_").replace("/", "_")


def _compute_window_features(window: pd.DataFrame, feature_cols: list) -> dict:
    """Return a flat dict of {col}_{stat} values for one window."""
    row: dict = {}
    for col in feature_cols:
        series = window[col].dropna()
        if series.empty:
            values = np.zeros(1)
        else:
            values = series.to_numpy(dtype=float)

        col_min = float(np.min(values))
        col_max = float(np.max(values))

        row[f"{col}_mean"] = float(np.mean(values))
        row[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row[f"{col}_var"] = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
        row[f"{col}_min"] = col_min
        row[f"{col}_max"] = col_max
        row[f"{col}_range"] = col_max - col_min
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row[f"{col}_skewness"] = (
                float(np.nan_to_num(skew(values))) if len(values) > 2 else 0.0
            )
            row[f"{col}_kurtosis"] = (
                float(np.nan_to_num(kurtosis(values))) if len(values) > 3 else 0.0
            )
    return row


def extract_features_from_file(
    csv_path: str, window_size: int, step_size: int
) -> pd.DataFrame:
    """Load one telemetry CSV and return a DataFrame of per-window feature rows."""
    df = pd.read_csv(csv_path)

    # Determine GPU label from the data itself
    if "gpu_name" in df.columns and not df["gpu_name"].dropna().empty:
        gpu_label = _label_from_name(str(df["gpu_name"].dropna().iloc[0]))
    else:
        # Fall back to the file name
        base = os.path.splitext(os.path.basename(csv_path))[0]
        gpu_label = base.replace("_gpu_telemetry", "")

    # Sort by time so windows are chronological
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Identify numeric telemetry columns (all numeric cols except metadata)
    feature_cols = [
        c
        for c in df.columns
        if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Fill NaN values: forward-fill first, then back-fill for leading gaps
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # Slide windows
    n = len(df)
    rows = []
    start = 0
    while start + window_size <= n:
        window = df.iloc[start : start + window_size]
        feature_row = _compute_window_features(window, feature_cols)
        feature_row["gpu_label"] = gpu_label
        rows.append(feature_row)
        start += step_size

    if not rows:
        print(
            f"  WARNING: Not enough samples in {csv_path} "
            f"(need ≥ {window_size}, got {n}) — file skipped."
        )
        return pd.DataFrame()

    print(f"  {gpu_label}: {n} samples → {len(rows)} windows")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract statistical features from GPU telemetry CSVs."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Window size in samples (default: 30, i.e. 30 seconds at 1 Hz)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=15,
        help="Step (stride) between windows in samples (default: 15, resulting in 15-second overlap with 30-second windows at 1 Hz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features.csv",
        help="Output CSV path (default: features.csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing the raw telemetry CSV files (default: current dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        f"Feature extraction  |  window={args.window} samples  "
        f"step={args.step} samples  output='{args.output}'"
    )

    all_frames = []
    for fname in GPU_CSV_FILES:
        path = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found.")
            continue
        print(f"Processing {fname} ...")
        frame = extract_features_from_file(path, args.window, args.step)
        if not frame.empty:
            all_frames.append(frame)

    if not all_frames:
        raise RuntimeError(
            "No telemetry files were processed. "
            "Check that the CSV files are present in the data directory."
        )

    combined = pd.concat(all_frames, ignore_index=True)

    # Replace any remaining NaN (e.g. std/skew of constant-only windows) with 0
    combined.fillna(0.0, inplace=True)

    combined.to_csv(args.output, index=False)
    print(
        f"\nDone. {len(combined)} feature rows written to '{args.output}'  "
        f"(columns: {len(combined.columns) - 1} features + 'gpu_label')"
    )
    print("Label distribution:")
    print(combined["gpu_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
