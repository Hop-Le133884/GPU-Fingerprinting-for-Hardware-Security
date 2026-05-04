#!/usr/bin/env python3
"""Live GPU fingerprinting demo: collect a short telemetry window and classify the device."""

import argparse
import threading
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
import joblib
from scipy.stats import skew

import torch
from pynvml import (
    NVMLError, NVML_PCIE_UTIL_RX_BYTES, NVML_PCIE_UTIL_TX_BYTES,
    nvmlDeviceGetClockInfo, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, nvmlDeviceGetPciInfo,
    nvmlDeviceGetPowerUsage, nvmlDeviceGetPerformanceState,
    nvmlDeviceGetTemperature, nvmlDeviceGetTotalEnergyConsumption,
    nvmlDeviceGetUUID, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetCurrPcieLinkGeneration, nvmlDeviceGetCurrPcieLinkWidth,
    nvmlDeviceGetMaxPcieLinkGeneration, nvmlDeviceGetMaxPcieLinkWidth,
    nvmlDeviceGetPcieThroughput, nvmlDeviceGetCurrentClocksThrottleReasons,
    nvmlShutdown, nvmlInit, nvmlSystemGetDriverVersion,
)

from collect_gpu_telemetry import (
    get_gpu_handle, sample_telemetry, run_gpu_workload,
    _workload_lock, _latest_workload,
)

METADATA_COLS = {"timestamp", "gpu_index", "gpu_uuid", "driver_version", "pci_bus_id", "gpu_name"}


def collect_live_samples(handle, device, n: int, interval: float) -> list:
    stop_event = threading.Event()
    workload_thread = threading.Thread(
        target=run_gpu_workload,
        args=(device, stop_event, 3.0, 0.0, 2048),
        daemon=True,
    )
    workload_thread.start()

    print(f"Warming up workload (15s)...")
    time.sleep(15)

    print(f"Collecting {n} samples at {interval}s interval...")
    samples = []
    for i in range(n):
        s = sample_telemetry(handle)
        with _workload_lock:
            s["matmul_gflops"] = _latest_workload["matmul_gflops"]
            s["workload_time_s"] = _latest_workload["workload_time_s"]
        s["power_per_gpu_util"] = (
            s["power_w"] / s["gpu_util"] if s.get("gpu_util") else None
        )
        s["power_per_gflop"] = (
            s["power_w"] / s["matmul_gflops"] if s.get("matmul_gflops") else None
        )
        samples.append(s)
        print(f"  [{i+1:2d}/{n}] gpu={s['gpu_util']}%  temp={s['temperature_c']}C  power={s['power_w']:.1f}W")
        time.sleep(interval)

    stop_event.set()
    workload_thread.join(timeout=5)
    return samples


def build_feature_vector(samples: list, feature_cols: list) -> np.ndarray:
    import pandas as pd
    df = pd.DataFrame(samples)
    vec = []
    for col in feature_cols:
        base = col.rsplit("_", 1)[0]
        stat = col.rsplit("_", 1)[1]
        if base not in df.columns:
            vec.append(0.0)
            continue
        vals = df[base].dropna().astype(float)
        if len(vals) < 2:
            vec.append(0.0)
            continue
        if stat == "mean":
            vec.append(float(vals.mean()))
        elif stat == "std":
            vec.append(float(vals.std()))
        elif stat == "var":
            vec.append(float(vals.var()))
        elif stat == "skew":
            s = float(skew(vals))
            vec.append(0.0 if (s != s) else s)  # replace NaN (constant column) with 0
        else:
            vec.append(0.0)
    return np.array(vec).reshape(1, -1)


def main():
    parser = argparse.ArgumentParser(description="Live GPU fingerprinting demo.")
    parser.add_argument("--model", default="model_random_forest.pkl",
                        help="Trained model .pkl saved by classifier.py (default: model_random_forest.pkl)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default 0)")
    parser.add_argument("--samples", type=int, default=90,
                        help="Number of telemetry samples to collect (default 90 → ~90s, uses multi-window voting)")
    parser.add_argument("--interval", type=float, default=1.0, help="Sample interval in seconds (default 1.0)")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    le = bundle["label_encoder"]
    feature_cols = bundle["feature_cols"]
    print(f"Loaded: {args.model}")
    print(f"Known GPUs: {list(le.classes_)}\n")

    nvmlInit()
    if not torch.cuda.is_available():
        nvmlShutdown()
        raise RuntimeError("CUDA not available.")
    device = torch.device(f"cuda:{args.gpu}")
    handle = get_gpu_handle(args.gpu)

    samples = collect_live_samples(handle, device, args.samples, args.interval)
    nvmlShutdown()

    # Slide 30-sample windows over collected samples, average probabilities across all windows
    window = 30
    step = 15
    all_probs = []
    n = len(samples)
    for start in range(0, n - window + 1, step):
        X = build_feature_vector(samples[start:start + window], feature_cols)
        if hasattr(model, "predict_proba"):
            all_probs.append(model.predict_proba(X)[0])

    if all_probs:
        probs = np.mean(all_probs, axis=0)
        n_windows = len(all_probs)
    else:
        X = build_feature_vector(samples, feature_cols)
        probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
        n_windows = 1

    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0]

    print(f"\n{'='*55}")
    print(f"  CLASSIFICATION RESULT  ({n_windows} windows averaged)")
    print(f"{'='*55}")
    print(f"  Predicted GPU: {pred_label}")
    if probs is not None:
        print("\n  Confidence scores:")
        for cls, p in sorted(zip(le.classes_, probs), key=lambda x: -x[1]):
            bar = "█" * int(p * 40)
            print(f"    {cls:<45} {p:5.1%}  {bar}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
