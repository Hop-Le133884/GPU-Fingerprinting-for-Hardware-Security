#!/usr/bin/env python3
"""Extract statistical feature vectors from GPU telemetry CSVs using sliding windows."""

import argparse
import glob
import pandas as pd
from scipy.stats import skew

METADATA_COLS = [
    "timestamp", "gpu_index", "gpu_uuid",
    "driver_version", "pci_bus_id",
]
LABEL_COL = "gpu_name"


def extract_windows(df: pd.DataFrame, feature_cols: list, window: int, step: int, label: str) -> list:
    records = []
    n = len(df)
    for start in range(0, n - window + 1, step):
        w = df[feature_cols].iloc[start : start + window]
        record = {"label": label}
        for col in feature_cols:
            vals = w[col].dropna().astype(float)
            if len(vals) < 2:
                record[f"{col}_mean"] = 0.0
                record[f"{col}_std"]  = 0.0
                record[f"{col}_var"]  = 0.0
                record[f"{col}_skew"] = 0.0
            else:
                s = float(skew(vals))
                record[f"{col}_mean"] = float(vals.mean())
                record[f"{col}_std"]  = float(vals.std())
                record[f"{col}_var"]  = float(vals.var())
                record[f"{col}_skew"] = 0.0 if (s != s) else s  # replace NaN skew (constant column) with 0
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description="Sliding-window feature extraction from GPU telemetry CSVs.")
    parser.add_argument("--input", nargs="+", default=None,
                        help="Telemetry CSV files (default: all *gpu_telemetry.csv in current dir)")
    parser.add_argument("--output", default="features.csv", help="Output feature matrix CSV")
    parser.add_argument("--window", type=int, default=30,
                        help="Window size in samples (default 30 → 30s at 1s interval)")
    parser.add_argument("--step", type=int, default=15,
                        help="Step size in samples (default 15 → 50%% overlap)")
    args = parser.parse_args()

    paths = args.input if args.input else sorted(glob.glob("*gpu_telemetry.csv"))
    if not paths:
        raise FileNotFoundError("No telemetry CSV files found. Pass --input or run from the data directory.")

    all_records = []
    for path in paths:
        df = pd.read_csv(path)
        label = df[LABEL_COL].iloc[0]
        feature_cols = [c for c in df.columns if c not in METADATA_COLS + [LABEL_COL]]
        records = extract_windows(df, feature_cols, args.window, args.step, label)
        all_records.extend(records)
        print(f"{path}: {len(df)} rows → {len(records)} windows  [{label}]")

    out = pd.DataFrame(all_records)
    # Drop columns that are entirely NaN (e.g. skew of constant-valued metrics)
    out = out.dropna(axis=1, how="all")
    # Drop rows that have any remaining NaN (e.g. windows before workload started)
    out = out.dropna(axis=0)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} feature vectors ({out.shape[1]-1} features + label) to {args.output}")
    print("Label distribution:")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
