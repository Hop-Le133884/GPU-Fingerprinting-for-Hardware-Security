#!/usr/bin/env python3
"""Collect GPU telemetry while an external GPU workload runs.

This script samples NVML metrics at fixed intervals and writes telemetry to CSV.
Use it alongside an external GPU benchmark such as glmark2.
"""

import argparse
import csv
import datetime
import os
import re
import threading
import time

import torch
from pynvml import (
    NVMLError,
    NVML_PCIE_UTIL_RX_BYTES,
    NVML_PCIE_UTIL_TX_BYTES,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetPciInfo,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetPerformanceState,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetTotalEnergyConsumption,
    nvmlDeviceGetUUID,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetCurrPcieLinkGeneration,
    nvmlDeviceGetCurrPcieLinkWidth,
    nvmlDeviceGetMaxPcieLinkGeneration,
    nvmlDeviceGetMaxPcieLinkWidth,
    nvmlDeviceGetPcieThroughput,
    nvmlDeviceGetCurrentClocksThrottleReasons,
    nvmlShutdown,
    nvmlInit,
    nvmlSystemGetDriverVersion,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Collect NVIDIA GPU telemetry while an external GPU benchmark runs.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to sample")
    parser.add_argument("--duration", type=float, default=60.0, help="Total collection duration in seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="Telemetry sample interval in seconds")
    parser.add_argument("--warmup", type=float, default=0.0, help="Seconds to wait before starting telemetry sampling")
    parser.add_argument("--output", type=str, default="gpu_telemetry.csv", help="CSV output file path")
    parser.add_argument("--workload", action="store_true", help="Run built-in CUDA matmul stress workload to generate varying GPU utilization")
    parser.add_argument("--workload-on", type=float, default=3.0, help="Seconds to hold each random utilization level (default 3.0)")
    parser.add_argument("--workload-off", type=float, default=0.0, help="Idle gap between utilization levels in seconds (default 0.0)")
    parser.add_argument("--matrix-size", type=int, default=2048, help="Starting matrix dimension; auto-scaled to ~20ms/call (default 2048)")
    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9._-]", "", value)
    return value[:80]


def initialize_nvml():
    try:
        nvmlInit()
    except NVMLError as err:
        raise RuntimeError(f"Could not initialize NVML: {err}")


def shutdown_nvml():
    try:
        nvmlShutdown()
    except NVMLError:
        pass


def get_gpu_handle(gpu_index: int):
    try:
        return nvmlDeviceGetHandleByIndex(gpu_index)
    except NVMLError as err:
        raise RuntimeError(f"Could not open handle for GPU {gpu_index}: {err}")


def safe_nvml_call(callable_obj, *args):
    try:
        return callable_obj(*args)
    except NVMLError:
        return None


def get_gpu_metadata(handle, gpu_index: int):
    name = safe_nvml_call(nvmlDeviceGetName, handle)
    uuid = safe_nvml_call(nvmlDeviceGetUUID, handle)
    pci = safe_nvml_call(nvmlDeviceGetPciInfo, handle)
    driver = safe_nvml_call(nvmlSystemGetDriverVersion)
    perf_state = safe_nvml_call(nvmlDeviceGetPerformanceState, handle)
    mem = safe_nvml_call(nvmlDeviceGetMemoryInfo, handle)

    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="ignore")
    if isinstance(uuid, bytes):
        uuid = uuid.decode("utf-8", errors="ignore")
    if isinstance(driver, bytes):
        driver = driver.decode("utf-8", errors="ignore")

    pci_bus_id = None
    if pci is not None:
        pci_bus_id = pci.busId.decode("utf-8", errors="ignore") if isinstance(pci.busId, bytes) else str(pci.busId)

    total_memory_mib = None
    if mem is not None:
        total_memory_mib = mem.total // 1024 ** 2

    return {
        "gpu_index": gpu_index,
        "gpu_name": name,
        "gpu_uuid": uuid,
        "driver_version": driver,
        "pci_bus_id": pci_bus_id,
        "total_memory_mib": total_memory_mib,
        "performance_state": perf_state,
    }


def sample_telemetry(handle):
    try:
        util = nvmlDeviceGetUtilizationRates(handle)
        mem = nvmlDeviceGetMemoryInfo(handle)
        temp = nvmlDeviceGetTemperature(handle, 0)
        power_mw = nvmlDeviceGetPowerUsage(handle)
        graphics_clock = nvmlDeviceGetClockInfo(handle, 0)
        sm_clock = nvmlDeviceGetClockInfo(handle, 1)
        memory_clock = nvmlDeviceGetClockInfo(handle, 2)
    except NVMLError as err:
        raise RuntimeError(f"NVML sampling failed: {err}")

    pcie_curr_gen = safe_nvml_call(nvmlDeviceGetCurrPcieLinkGeneration, handle)
    pcie_curr_width = safe_nvml_call(nvmlDeviceGetCurrPcieLinkWidth, handle)
    pcie_max_gen = safe_nvml_call(nvmlDeviceGetMaxPcieLinkGeneration, handle)
    pcie_max_width = safe_nvml_call(nvmlDeviceGetMaxPcieLinkWidth, handle)
    pcie_throughput_tx = safe_nvml_call(nvmlDeviceGetPcieThroughput, handle, NVML_PCIE_UTIL_TX_BYTES)
    pcie_throughput_rx = safe_nvml_call(nvmlDeviceGetPcieThroughput, handle, NVML_PCIE_UTIL_RX_BYTES)
    throttle_reasons = safe_nvml_call(nvmlDeviceGetCurrentClocksThrottleReasons, handle)
    energy_consumption_mj = safe_nvml_call(nvmlDeviceGetTotalEnergyConsumption, handle)

    sample = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_util": util.gpu,
        "mem_util": util.memory,
        "mem_total_mib": mem.total // 1024 ** 2,
        "mem_used_mib": mem.used // 1024 ** 2,
        "mem_free_mib": (mem.total - mem.used) // 1024 ** 2,
        "mem_used_ratio": mem.used / mem.total if mem.total else None,
        "temperature_c": temp,
        "power_w": power_mw / 1000.0,
        "graphics_clock_mhz": graphics_clock,
        "sm_clock_mhz": sm_clock,
        "memory_clock_mhz": memory_clock,
        "pcie_curr_gen": pcie_curr_gen,
        "pcie_curr_width": pcie_curr_width,
        "pcie_max_gen": pcie_max_gen,
        "pcie_max_width": pcie_max_width,
        "pcie_throughput_tx_bytes": pcie_throughput_tx,
        "pcie_throughput_rx_bytes": pcie_throughput_rx,
        "throttle_reasons": throttle_reasons,
        "energy_consumption_mj": energy_consumption_mj,
    }
    return sample


_workload_lock = threading.Lock()
_latest_workload: dict = {"matmul_gflops": None, "workload_time_s": None}


def _calibrate(device, a, b):
    for _ in range(2):
        torch.matmul(a, b)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(3):
        torch.matmul(a, b)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / 3


def run_gpu_workload(device, stop_event: threading.Event, step_seconds: float, off_seconds: float, matrix_size: int):
    """Step through random utilization levels using calibrated duty-cycle control.

    Iteratively scales matrix size until one matmul ≈ TARGET_MS, then inserts
    proportional idle sleep between kernels so NVML time-averages to the target level.
    """
    import random
    global _latest_workload

    TARGET_MS = 20.0
    TOLERANCE = 0.35

    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)

    # Iterative calibration: sqrt-based scaling is conservative and avoids overshooting
    for _ in range(4):
        single_matmul_s = _calibrate(device, a, b)
        actual_ms = single_matmul_s * 1000
        print(f"[workload] {matrix_size}x{matrix_size}: {actual_ms:.1f} ms/call")
        if abs(actual_ms - TARGET_MS) / TARGET_MS <= TOLERANCE:
            break
        scale = (TARGET_MS / actual_ms) ** 0.5
        matrix_size = max(256, int(matrix_size * scale / 64) * 64)
        del a, b
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)

    print(f"[workload] ready: {matrix_size}x{matrix_size} @ {single_matmul_s*1000:.1f} ms/call")

    UTIL_LEVELS = [0, 25, 50, 75, 100]

    while not stop_event.is_set():
        target_util = random.choice(UTIL_LEVELS)
        step_end = time.perf_counter() + step_seconds
        total_iters = 0
        print(f"[workload] targeting {target_util}% utilization for {step_seconds}s")

        if target_util == 0:
            while time.perf_counter() < step_end and not stop_event.is_set():
                stop_event.wait(min(0.05, step_end - time.perf_counter()))
        elif target_util == 100:
            while time.perf_counter() < step_end and not stop_event.is_set():
                torch.matmul(a, b)
                torch.cuda.synchronize(device)  # sync each call to prevent queue flooding
                total_iters += 1
        else:
            # idle_s = time GPU should be idle per matmul to achieve target duty cycle
            # duty = compute_s / (compute_s + idle_s) = target_util/100
            idle_s = single_matmul_s * (100 - target_util) / target_util
            while time.perf_counter() < step_end and not stop_event.is_set():
                torch.matmul(a, b)
                total_iters += 1
                torch.cuda.synchronize(device)
                if idle_s > 0.001:
                    time.sleep(idle_s)

        with _workload_lock:
            if total_iters > 0 and step_seconds > 0:
                gflops = 2 * (matrix_size ** 3) * total_iters / step_seconds / 1e9
                _latest_workload["matmul_gflops"] = round(gflops, 2)
            else:
                _latest_workload["matmul_gflops"] = 0.0
            _latest_workload["workload_time_s"] = round(step_seconds, 4)

        if off_seconds > 0:
            stop_event.wait(off_seconds)


def write_csv_header(path: str, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(path: str, row: dict, fieldnames):
    with open(path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)


def main():
    args = parse_args()
    initialize_nvml()
    handle = get_gpu_handle(args.gpu)

    if not torch.cuda.is_available():
        shutdown_nvml()
        raise RuntimeError("CUDA is not available on this machine.")

    device = torch.device(f"cuda:{args.gpu}")
    if args.gpu >= torch.cuda.device_count():
        shutdown_nvml()
        raise RuntimeError(f"GPU index {args.gpu} is out of range for available devices.")

    print(f"Using GPU #{args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    mode = "built-in CUDA workload" if args.workload else "external workload monitoring"
    print(f"Mode: {mode}, duration: {args.duration}s, interval: {args.interval}s")

    metadata = get_gpu_metadata(handle, args.gpu)
    output_path = args.output
    if args.output == "gpu_telemetry.csv":
        gpu_name = metadata.get("gpu_name") or f"gpu{args.gpu}"
        safe_name = sanitize_filename(gpu_name)
        base = os.path.basename(args.output)
        dirpath = os.path.dirname(args.output) or "."
        output_path = os.path.join(dirpath, f"{safe_name}_{base}")
        print(f"Saving telemetry to: {output_path}")

    fieldnames = [
        "timestamp",
        "gpu_index",
        "gpu_name",
        "gpu_uuid",
        "driver_version",
        "pci_bus_id",
        "total_memory_mib",
        "performance_state",
        "gpu_util",
        "mem_util",
        "mem_total_mib",
        "mem_used_mib",
        "mem_free_mib",
        "mem_used_ratio",
        "temperature_c",
        "power_w",
        "graphics_clock_mhz",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "pcie_curr_gen",
        "pcie_curr_width",
        "pcie_max_gen",
        "pcie_max_width",
        "pcie_throughput_tx_bytes",
        "pcie_throughput_rx_bytes",
        "throttle_reasons",
        "energy_consumption_mj",
        "matmul_gflops",
        "power_per_gpu_util",
        "power_per_gflop",
        "workload_time_s",
    ]
    write_csv_header(output_path, fieldnames)

    stop_event = threading.Event()
    workload_thread = None
    if args.workload:
        print(f"Starting CUDA workload: {args.matrix_size}x{args.matrix_size} matmul, "
              f"step={args.workload_on}s gap={args.workload_off}s, levels=[0,25,50,75,100]%")
        workload_thread = threading.Thread(
            target=run_gpu_workload,
            args=(device, stop_event, args.workload_on, args.workload_off, args.matrix_size),
            daemon=True,
        )
        workload_thread.start()

    if args.warmup > 0:
        print(f"Waiting {args.warmup}s before telemetry collection...")
        time.sleep(args.warmup)

    print("Starting telemetry collection...")
    end_time = time.time() + args.duration
    while time.time() < end_time:
        telemetry = sample_telemetry(handle)
        telemetry.update(metadata)
        with _workload_lock:
            telemetry["matmul_gflops"] = _latest_workload["matmul_gflops"]
            telemetry["workload_time_s"] = _latest_workload["workload_time_s"]
        telemetry["power_per_gpu_util"] = (
            telemetry["power_w"] / telemetry["gpu_util"]
            if telemetry["gpu_util"] and telemetry["power_w"] is not None
            else None
        )
        telemetry["power_per_gflop"] = (
            telemetry["power_w"] / telemetry["matmul_gflops"]
            if telemetry["matmul_gflops"] and telemetry["power_w"] is not None
            else None
        )
        append_csv_row(output_path, telemetry, fieldnames)
        print(
            f"Logged sample: gpu={telemetry['gpu_util']}% mem={telemetry['mem_util']}% "
            f"temp={telemetry['temperature_c']}C power={telemetry['power_w']:.2f}W"
            + (f" gflops={telemetry['matmul_gflops']}" if telemetry["matmul_gflops"] else "")
        )
        time_remaining = end_time - time.time()
        if time_remaining <= 0:
            break
        time.sleep(min(args.interval, time_remaining))

    stop_event.set()
    if workload_thread:
        workload_thread.join(timeout=5)
    shutdown_nvml()
    print(f"Telemetry collection finished. Output saved to: {args.output}")


if __name__ == "__main__":
    main()
