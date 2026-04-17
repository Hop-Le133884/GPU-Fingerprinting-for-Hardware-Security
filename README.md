# GPU-Fingerprinting-for-Hardware-Security
Classifying Unique Device Features in Embedded Systems

## Project Overview:
This project focuses on classifying unique computer-based features of GPUs to explore hardware security and device identification in embedded-system contexts. The experiments are conducted on laptop and desktop NVIDIA GPUs (such as the RTX 3050 and RTX 3090) to model the same side-channel behavior expected in embedded and edge devices. The system logs performance metrics and applies machine learning to extract hardware-specific signatures, demonstrating the potential privacy risks of silent device tracking by websites and applications through GPU side-channels.

## Prerequisites & Environment Setup:
To build the data ingestion and machine learning pipeline, ensure your Python environment is set up with the following:

 **Deep Learning / Workloads:** torch (PyTorch) for matrix multiplication (matmul) workloads.

 **Data Processing:** pandas and numpy for handling CSV outputs and calculating statistical features.
 
 **Machine Learning:** scikit-learn for training classifiers (Random Forest, SVM, etc.) and generating evaluation metrics.
 
 **Hardware Monitoring:** Tools or Python wrappers (like pynvml) capable of logging NVIDIA GPU telemetry.
 
 ## Step-by-Step Implementation Guide
 
 **Step 1:** Literature Review & Baseline Research
 
 * Review existing research on GPU fingerprinting from ACM, IEEE, and arXiv.
 
 * Analyze DRAWNAPART-style WebGL fingerprinting implementations and reference their open-source GitHub repositories to understand baseline methodologies.
 
 **Step 2:** Workload Execution & Data Acquisition
 
 * Develop a Python script to run intensive workloads, such as PyTorch matmul operations or render loops, directly on the NVIDIA GPU.
 
 * While the workload runs, continuously log hardware performance metrics, including:
 
 * GPU and memory utilization percentages.
 
 * Temperature (°C) and power draw (W).
 
 * Clock speeds (MHz) and execution timing.
 
 * Output the logged telemetry data into CSV format for processing.

 **Data Collection Instructions:**

 Run the following command once per GPU device. The built-in CUDA workload automatically cycles through 0%, 25%, 50%, 75%, and 100% GPU utilization levels (3 seconds each) to produce varied telemetry. The script auto-calibrates the matrix size to achieve consistent timing across different GPU models.

 ```bash
 python collect_gpu_telemetry.py --workload --duration 5400 --interval 0.5
 ```

 * `--duration 5400` — runs for 90 minutes, capturing thermal stabilization and long-term clock/power variance
 * `--interval 0.5` — samples every 0.5 seconds (~10,800 samples per GPU); going below 0.5s is not recommended as NVML's internal measurement window is ~166ms
 * `--workload` — enables the built-in PyTorch matmul stress loop; no external benchmark needed
 * Output is saved automatically as `<GPU_Name>_gpu_telemetry.csv` in the current directory

 Repeat for each GPU. Keep the output CSV files for Step 3 feature extraction. During feature extraction, each CSV is sliced into overlapping 30-second sliding windows — each window becomes one labeled data point for the ML classifier.
 
 **Step 3:** Statistical Feature Extraction
 
 * Ingest the raw CSV metrics and compute statistical features to build out your feature vectors.
 
 Calculate the Mean, Variance, Skewness, and Standard Deviation for the collected metrics.
 
 **Step 4:** ML Classification & Analysis
 
 * Pass the extracted feature vectors into machine learning classifiers.
 Train models such as Random Forest (RF), Support Vector Machines (SVM), Decision Trees, or Logistic Regression to uniquely identify the specific host device.
 
 * Ensure the model effectively demonstrates the risk of silent tracking by distinguishing between different devices (e.g., separating the RTX 3050 data from the 3090 data).
 
 **Step 5:** Evaluation & Reporting
 
 * Since there are no public Kaggle or HuggingFace datasets for GPU performance-based fingerprinting, evaluate the model strictly on your self-collected data.
 
 * Assess the classifiers using standard evaluation metrics: Accuracy, Precision, Recall, F1-score, and False Positive/Negative Ratios.
 
 * Compile the final results, methodologies, and findings into a PDF report.
 
 * Build a live demonstration script that runs a workload and outputs a real-time classification confidence score for the detected device.