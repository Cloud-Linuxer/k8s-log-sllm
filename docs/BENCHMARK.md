# Benchmark Report

**Model**: `models/v2/model.onnx`
**Date**: 2025-11-25 12:20:55
**CPU Cores**: 32

## Summary

| Metric | Value |
|--------|-------|
| Single Inference (mean) | 3.56 ms |
| Single Inference (P99) | 3.72 ms |
| Max Throughput (1 thread) | 283 logs/sec |
| Max Throughput (8 threads) | 1371 logs/sec |
| Model Memory | 846 MB |
| ONNX File Size | 42.6 MB |

## Latency (Single Inference, 1 Thread)

| Metric | Value |
|--------|-------|
| Mean | 3.560 ms |
| Std | 0.029 ms |
| P50 | 3.552 ms |
| P95 | 3.596 ms |
| P99 | 3.719 ms |
| Min | 3.537 ms |
| Max | 3.784 ms |

## Batch Performance (1 Thread)

| Batch Size | Latency (ms) | Per Log (ms) | Throughput (logs/sec) |
|------------|--------------|--------------|----------------------|
| 1 | 3.55 | 3.555 | 281.3 |
| 4 | 14.05 | 3.513 | 284.7 |
| 8 | 28.29 | 3.536 | 282.8 |
| 16 | 56.88 | 3.555 | 281.3 |
| 32 | 118.37 | 3.699 | 270.3 |
| 64 | 255.66 | 3.995 | 250.3 |
| 128 | 550.13 | 4.298 | 232.7 |

## Thread Scaling (Batch Size = 32)

| Threads | Latency (ms) | Throughput (logs/sec) | Speedup |
|---------|--------------|----------------------|---------|
| 1 | 118.6 | 269.8 | 1.00x |
| 2 | 63.4 | 504.7 | 1.87x |
| 4 | 37.4 | 855.4 | 3.17x |
| 8 | 23.3 | 1370.8 | 5.08x |

## Memory Usage

| Component | Size |
|-----------|------|
| ONNX File | 42.6 MB |
| Model Loaded | 846 MB |
| Peak Usage | 1005 MB |

## Recommendations

- **Low Latency**: Use batch size 1, single thread → 3.6ms per log
- **High Throughput**: Use batch size 8-16, 8 threads → 1371 logs/sec
- **Memory Constrained**: Model requires ~846MB RAM
