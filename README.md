# GPU Fingerprinting for Hardware Security
**Classifying Unique Device Features in Embedded Systems**

Kenneth Fulton · Hop Le — Texas A&M University San Antonio

---

## Project Overview

This project builds an end-to-end GPU hardware fingerprinting pipeline. It collects NVIDIA GPU telemetry under controlled CUDA workloads, extracts statistical features via sliding windows, and trains ML classifiers to identify GPU devices by model. Motivated by DRAWNAPART (NDSS 2022), which proved that GPU hardware manufacturing variations produce measurable device signatures, this work demonstrates that system-level NVML telemetry is sufficient to classify GPU devices — establishing a silent hardware tracking risk outside browser environments.

**GPUs tested:** RTX 3050 4GB Laptop · RTX 3060 Laptop · RTX 3090

---

## Prerequisites

- NVIDIA GPU with driver ≥ 520
- CUDA-capable PyTorch build (`torch.cuda.is_available()` must return `True`)
- Python 3.12+

**Recommended: uv**
```bash
uv init
uv add torch nvidia-ml-py pandas numpy scipy scikit-learn xgboost matplotlib joblib
source .venv/bin/activate
```

**Alternative: pip**
```bash
pip install torch nvidia-ml-py pandas numpy scipy scikit-learn xgboost matplotlib joblib
```

> Note: use `nvidia-ml-py` (not `pynvml`) — `pynvml` is deprecated and will trigger a warning from PyTorch.

---

## Pipeline Overview

```
Step 1: Data Collection     collect_gpu_telemetry.py  →  <GPU>_gpu_telemetry.csv
Step 2: Feature Extraction  feature_extraction.py     →  features.csv
Step 3: ML Classification   classifier.py             →  classification_results.csv + model_*.pkl
Step 4: Live Demo           demo.py                   →  real-time confidence scores
```

---

## Step 1 — Data Collection

Run once per GPU device. The script runs a calibrated PyTorch CUDA matmul workload that randomly cycles through 0%, 25%, 50%, 75%, and 100% GPU utilization every 3 seconds, while logging NVML telemetry at 1-second intervals.

```bash
python collect_gpu_telemetry.py --workload --duration 5400 --interval 1
```

| Flag | Description |
|---|---|
| `--workload` | Enables the built-in CUDA matmul stress loop |
| `--duration 5400` | Run for 90 minutes → ~5400 samples per GPU |
| `--interval 1` | Sample every 1 second |
| `--gpu 0` | GPU index (default 0); change if running on a secondary GPU |

Output is saved automatically as `<GPU_Name>_gpu_telemetry.csv`. Repeat for each GPU and keep all CSVs in the same directory.

---

## Step 2 — Feature Extraction

Ingests all telemetry CSVs, applies 30-second sliding windows with 15-second overlap, and computes mean, standard deviation, variance, and skewness per metric column. Each window becomes one labeled feature vector.

```bash
python feature_extraction.py
```

Optional flags:
```bash
python feature_extraction.py --window 30 --step 15 --output features.csv
```

| Flag | Description |
|---|---|
| `--window 30` | Window size in samples (30s at 1s interval) |
| `--step 15` | Step size in samples (15s overlap between windows) |
| `--input a.csv b.csv` | Explicit file list (default: all `*gpu_telemetry.csv` in current dir) |
| `--output` | Output file (default: `features.csv`) |

Output: `features.csv` — each row is one labeled feature vector.

Expected output:
```
NVIDIA_GeForce_RTX_3050_4GB_Laptop_GPU_gpu_telemetry.csv: 5167 rows → 332 windows
NVIDIA_GeForce_RTX_3060_Laptop_GPU_gpu_telemetry.csv:     4988 rows → 320 windows
NVIDIA_GeForce_RTX_3090_gpu_telemetry.csv:                5090 rows → 327 windows
Saved ~950 feature vectors to features.csv
```

---

## Step 3 — ML Classification

Trains Random Forest, SVM, XGBoost, and Decision Tree classifiers using 5-fold stratified cross-validation. Outputs accuracy, precision, recall, F1-score, FP/FN rates, confusion matrix plots, and saves trained models for the live demo.

```bash
python classifier.py
```

Optional flags:
```bash
python classifier.py --input features.csv --folds 5
```

To also train behavioral-only models (VRAM-size features removed):
```bash
python classifier.py --drop-trivial
```

| Flag | Description |
|---|---|
| `--input` | Feature matrix CSV (default: `features.csv`) |
| `--folds` | Number of CV folds (default: 5) |
| `--save-models` | Save trained models as `.pkl` files (default: on, required for demo) |
| `--drop-trivial` | Drop VRAM-size features (mem_free_mib, mem_total_mib, mem_used_mib, mem_used_ratio) to evaluate behavioral-only fingerprinting |

Outputs (run once without `--drop-trivial`, once with, to get all 8 models):
- `classification_results.csv` — metrics table for all classifiers
- `confusion_random_forest.png`, `confusion_svm.png`, `confusion_xgboost.png`, `confusion_decision_tree.png`
- `model_random_forest.pkl`, `model_svm.pkl`, `model_xgboost.pkl`, `model_decision_tree.pkl`
- `model_random_forest_no_mem.pkl`, `model_svm_no_mem.pkl`, `model_xgboost_no_mem.pkl`, `model_decision_tree_no_mem.pkl`
- Top 15 most discriminative features printed to console (Random Forest)

---

## Step 4 — Live Demo

Collects ~30 seconds of live telemetry from the current GPU, extracts one feature window, and runs the trained classifier to output a real-time device identification with confidence scores.

```bash
python demo.py
```

Optional flags:
```bash
python demo.py --model model_random_forest.pkl --gpu 0 --samples 30 --interval 1
```

| Flag | Description |
|---|---|
| `--model` | Trained `.pkl` model file (default: `model_random_forest.pkl`) |
| `--gpu` | GPU index to fingerprint (default: 0) |
| `--samples` | Samples to collect (default: 30 → ~30s) |
| `--interval` | Sample interval in seconds (default: 1.0) |

Example output:
```
Loaded: model_random_forest.pkl
Known GPUs: ['NVIDIA GeForce RTX 3050 4GB Laptop GPU', 'NVIDIA GeForce RTX 3060 Laptop GPU', 'NVIDIA GeForce RTX 3090']

Warming up workload (5s)...
Collecting 30 samples at 1.0s interval...
  [ 1/30] gpu=75%  temp=62C  power=38.2W
  ...

=======================================================
  CLASSIFICATION RESULT
=======================================================
  Predicted GPU: NVIDIA GeForce RTX 3060 Laptop GPU

  Confidence scores:
    NVIDIA GeForce RTX 3060 Laptop GPU      91.5%  ████████████████████████████████████
    NVIDIA GeForce RTX 3050 4GB Laptop GPU   6.0%  ██
    NVIDIA GeForce RTX 3090                  2.5%  █
=======================================================
```

---

## Repository Structure

```
GPU-Fingerprinting-for-Hardware-Security/
├── collect_gpu_telemetry.py                        # Step 1: telemetry collection
├── feature_extraction.py                           # Step 2: sliding window feature extraction
├── classifier.py                                   # Step 3: ML training and evaluation
├── demo.py                                         # Step 4: live classification demo
├── meta_data.csv                                   # Column definitions for telemetry schema
├── NVIDIA_GeForce_RTX_3050_4GB_Laptop_GPU_gpu_telemetry.csv
├── NVIDIA_GeForce_RTX_3060_Laptop_GPU_gpu_telemetry.csv
├── NVIDIA_GeForce_RTX_3090_gpu_telemetry.csv
├── features.csv                                    # Generated by feature_extraction.py
├── classification_results.csv                      # Generated by classifier.py
└── model_*.pkl                                     # Trained models, generated by classifier.py
```

---

## References

[1] Laor et al. 2022. DRAWNAPART: A Device Identification Technique based on Remote GPU Fingerprinting. NDSS 2022. doi:10.14722/ndss.2022.24093
