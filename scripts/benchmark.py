#!/usr/bin/env python3
"""Benchmark script for model performance metrics."""

import argparse
import json
import os
import sys
import time
import gc
from pathlib import Path

import numpy as np
import psutil


def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_benchmark(model_path: str, tokenizer_path: str, output_path: str = None):
    """Run comprehensive benchmark and return metrics."""
    results = {
        "model_path": model_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {},
        "memory": {},
        "latency": {},
        "throughput": {},
        "scaling": {},
    }

    # System info
    results["system"]["cpu_count"] = os.cpu_count()
    results["system"]["platform"] = sys.platform

    print("=" * 70)
    print("K8s Log Risk Scorer - Benchmark Report")
    print("=" * 70)

    # Memory baseline
    gc.collect()
    baseline_mem = get_memory_mb()

    # Load model
    print("\n[1/5] Loading model...")
    from src.inference import LogRiskPredictor
    from src.model import InferenceConfig

    config = InferenceConfig(num_threads=1)
    predictor = LogRiskPredictor(model_path, tokenizer_path, config)

    model_mem = get_memory_mb()
    results["memory"]["baseline_mb"] = round(baseline_mem, 1)
    results["memory"]["model_loaded_mb"] = round(model_mem, 1)
    results["memory"]["model_size_mb"] = round(model_mem - baseline_mem, 1)

    # ONNX file size
    onnx_size = os.path.getsize(model_path) / 1024 / 1024
    results["memory"]["onnx_file_mb"] = round(onnx_size, 1)

    print(f"  Model memory: {results['memory']['model_size_mb']} MB")
    print(f"  ONNX file: {results['memory']['onnx_file_mb']} MB")

    # Test data
    test_logs = [
        "INFO kubelet Successfully pulled image nginx:latest",
        "ERROR kernel panic - not syncing: VFS",
        "WARNING connection refused to database",
        "DEBUG processing request id=12345",
        "CRITICAL OOMKilled container exceeded memory limit",
        "INFO nginx GET /health HTTP/1.1 200",
        "ERROR upstream timed out connecting to backend",
        "TRACE entering function processRequest",
    ]

    # Warmup
    print("\n[2/5] Warming up...")
    for _ in range(20):
        predictor.predict(test_logs[:1])

    # Single inference latency
    print("\n[3/5] Measuring single inference latency...")
    times = []
    for _ in range(200):
        start = time.perf_counter()
        predictor.predict_single(test_logs[0])
        times.append((time.perf_counter() - start) * 1000)

    results["latency"]["single"] = {
        "mean_ms": round(np.mean(times), 3),
        "std_ms": round(np.std(times), 3),
        "p50_ms": round(np.percentile(times, 50), 3),
        "p95_ms": round(np.percentile(times, 95), 3),
        "p99_ms": round(np.percentile(times, 99), 3),
        "min_ms": round(np.min(times), 3),
        "max_ms": round(np.max(times), 3),
    }

    print(f"  Mean: {results['latency']['single']['mean_ms']:.2f} ms")
    print(f"  P99:  {results['latency']['single']['p99_ms']:.2f} ms")

    # Batch latency
    print("\n[4/5] Measuring batch performance...")
    results["latency"]["batch"] = {}
    results["throughput"]["batch"] = {}

    for batch_size in [1, 4, 8, 16, 32, 64, 128]:
        batch = (test_logs * ((batch_size // len(test_logs)) + 1))[:batch_size]

        times = []
        for _ in range(50):
            start = time.perf_counter()
            predictor.predict(batch)
            times.append((time.perf_counter() - start) * 1000)

        mean_time = np.mean(times)
        throughput = batch_size / (mean_time / 1000)

        results["latency"]["batch"][str(batch_size)] = {
            "mean_ms": round(mean_time, 2),
            "per_log_ms": round(mean_time / batch_size, 3),
        }
        results["throughput"]["batch"][str(batch_size)] = round(throughput, 1)

        print(f"  Batch {batch_size:>3}: {mean_time:>7.2f} ms | {throughput:>7.1f} logs/sec")

    # Thread scaling
    print("\n[5/5] Measuring thread scaling...")
    batch_32 = (test_logs * 4)[:32]

    for threads in [1, 2, 4, 8]:
        config = InferenceConfig(num_threads=threads)
        predictor = LogRiskPredictor(model_path, tokenizer_path, config)

        # Warmup
        for _ in range(5):
            predictor.predict(batch_32)

        times = []
        for _ in range(30):
            start = time.perf_counter()
            predictor.predict(batch_32)
            times.append((time.perf_counter() - start) * 1000)

        mean_time = np.mean(times)
        throughput = 32 / (mean_time / 1000)

        results["scaling"][str(threads)] = {
            "latency_ms": round(mean_time, 2),
            "throughput": round(throughput, 1),
        }

        print(f"  {threads} thread(s): {mean_time:>6.1f} ms | {throughput:>7.1f} logs/sec")

    # Memory after batch
    gc.collect()
    final_mem = get_memory_mb()
    results["memory"]["peak_mb"] = round(final_mem, 1)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Single inference:  {results['latency']['single']['mean_ms']:.2f} ms (P99: {results['latency']['single']['p99_ms']:.2f} ms)")
    print(f"  Best throughput:   {max(results['throughput']['batch'].values()):.0f} logs/sec")
    print(f"  Memory usage:      {results['memory']['model_size_mb']:.0f} MB")
    print(f"  ONNX model size:   {results['memory']['onnx_file_mb']:.1f} MB")
    print("=" * 70)

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def generate_markdown_report(results: dict, output_path: str):
    """Generate markdown report from benchmark results."""
    report = f"""# Benchmark Report

**Model**: `{results['model_path']}`
**Date**: {results['timestamp']}
**CPU Cores**: {results['system']['cpu_count']}

## Summary

| Metric | Value |
|--------|-------|
| Single Inference (mean) | {results['latency']['single']['mean_ms']:.2f} ms |
| Single Inference (P99) | {results['latency']['single']['p99_ms']:.2f} ms |
| Max Throughput (1 thread) | {results['throughput']['batch']['8']:.0f} logs/sec |
| Max Throughput (8 threads) | {results['scaling']['8']['throughput']:.0f} logs/sec |
| Model Memory | {results['memory']['model_size_mb']:.0f} MB |
| ONNX File Size | {results['memory']['onnx_file_mb']:.1f} MB |

## Latency (Single Inference, 1 Thread)

| Metric | Value |
|--------|-------|
| Mean | {results['latency']['single']['mean_ms']:.3f} ms |
| Std | {results['latency']['single']['std_ms']:.3f} ms |
| P50 | {results['latency']['single']['p50_ms']:.3f} ms |
| P95 | {results['latency']['single']['p95_ms']:.3f} ms |
| P99 | {results['latency']['single']['p99_ms']:.3f} ms |
| Min | {results['latency']['single']['min_ms']:.3f} ms |
| Max | {results['latency']['single']['max_ms']:.3f} ms |

## Batch Performance (1 Thread)

| Batch Size | Latency (ms) | Per Log (ms) | Throughput (logs/sec) |
|------------|--------------|--------------|----------------------|
"""

    for batch_size in ["1", "4", "8", "16", "32", "64", "128"]:
        lat = results["latency"]["batch"][batch_size]
        thr = results["throughput"]["batch"][batch_size]
        report += f"| {batch_size} | {lat['mean_ms']:.2f} | {lat['per_log_ms']:.3f} | {thr:.1f} |\n"

    report += """
## Thread Scaling (Batch Size = 32)

| Threads | Latency (ms) | Throughput (logs/sec) | Speedup |
|---------|--------------|----------------------|---------|
"""

    base_throughput = results["scaling"]["1"]["throughput"]
    for threads in ["1", "2", "4", "8"]:
        s = results["scaling"][threads]
        speedup = s["throughput"] / base_throughput
        report += f"| {threads} | {s['latency_ms']:.1f} | {s['throughput']:.1f} | {speedup:.2f}x |\n"

    report += f"""
## Memory Usage

| Component | Size |
|-----------|------|
| ONNX File | {results['memory']['onnx_file_mb']:.1f} MB |
| Model Loaded | {results['memory']['model_size_mb']:.0f} MB |
| Peak Usage | {results['memory']['peak_mb']:.0f} MB |

## Recommendations

- **Low Latency**: Use batch size 1, single thread → {results['latency']['single']['mean_ms']:.1f}ms per log
- **High Throughput**: Use batch size 8-16, 8 threads → {results['scaling']['8']['throughput']:.0f} logs/sec
- **Memory Constrained**: Model requires ~{results['memory']['model_size_mb']:.0f}MB RAM
"""

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model performance")
    parser.add_argument("-m", "--model", required=True, help="Path to ONNX model")
    parser.add_argument("-t", "--tokenizer", required=True, help="Path to tokenizer")
    parser.add_argument("-o", "--output", default="benchmark_results.json", help="Output JSON file")
    parser.add_argument("--markdown", help="Generate markdown report")
    args = parser.parse_args()

    results = run_benchmark(args.model, args.tokenizer, args.output)

    if args.markdown:
        generate_markdown_report(results, args.markdown)


if __name__ == "__main__":
    main()
