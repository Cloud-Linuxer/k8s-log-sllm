# k8s-log-sllm

K8s/System Log Risk Scoring using BERT-mini fine-tuning.

## Overview

This project classifies Kubernetes, system, and application logs into risk levels (0-10) using a fine-tuned `google/bert-mini` (prajjwal1/bert-mini) transformer model.

- **Input**: Single-line log string
- **Output**: Risk score 0-10 (11-class classification)
- **Training**: GPU-enabled fine-tuning
- **Inference**: CPU-optimized via ONNX Runtime

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Preprocess Logs

Convert raw logs to preprocessed format with token replacements:

```bash
python scripts/preprocess.py --input data/sample_logs.txt --output data/processed.csv
```

Token replacements:
- Timestamps → `<TIME>`
- IP addresses → `<IP>`
- UUIDs/hex IDs → `<ID>`
- PIDs → `<PID>`
- Long paths → `<PATH>`

### 2. Generate Labels

Create weak labels based on log level and keywords:

```bash
python scripts/label.py --input data/processed.csv --output data/labeled.csv --details
```

### 3. Train Model

Fine-tune BERT-mini on labeled data:

```bash
python scripts/train.py \
    --data data/labeled.csv \
    --output models/v1 \
    --epochs 5 \
    --batch-size 32 \
    --lr 3e-5
```

### 4. Export to ONNX

Export trained model for CPU inference:

```bash
python scripts/export.py \
    --checkpoint models/v1/best_model.pt \
    --output models/v1/model.onnx
```

### 5. Run Inference

Score logs using the exported model:

```bash
python scripts/predict.py \
    --model models/v1/model.onnx \
    --tokenizer models/v1/tokenizer \
    --input data/sample_logs.txt \
    --preprocess \
    --format simple
```

## Risk Level Guide

| Level | Meaning | Examples |
|-------|---------|----------|
| 0-1 | Trace/Debug | Debug logs, trace information |
| 2-3 | Normal | INFO logs, routine operations |
| 4-5 | Warning | Warnings, minor issues |
| 6-7 | Error | Errors, failures |
| 8-9 | Critical | Critical errors, system issues |
| 10 | Emergency | Panic, fatal errors |

## Labeling Policy

**Base scores by log level:**
- TRACE/DEBUG: 0-1
- INFO: 2
- NOTICE/VERBOSE: 3
- WARN/WARNING: 4-5
- ERROR: 6-7
- CRITICAL/ALERT: 8-9
- FATAL/PANIC/EMERG: 9-10

**Keyword bonuses:**
- K8s: `CrashLoopBackOff`, `ImagePullBackOff`, `node not ready` (+2)
- System: `kernel panic`, `segfault`, `I/O error` (+3)
- Network: `connection refused` (+1), `upstream timed out` (+2)
- Security: `permission denied`, `unauthorized` (+2)

## Project Structure

```
k8s-log-sllm/
├── src/
│   ├── preprocessing/    # Log preprocessing
│   ├── labeling/         # Weak label generation
│   ├── data/             # Dataset utilities
│   ├── model/            # BERT classifier
│   ├── training/         # Training pipeline
│   └── inference/        # ONNX inference
├── scripts/              # CLI tools
├── tests/                # Unit & E2E tests
├── data/                 # Training data
└── models/               # Saved models
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessor.py -v

# Run with coverage
pytest tests/ --cov=src
```

## Model Architecture

```
BERT-mini (prajjwal1/bert-mini)
    ↓
[CLS] token embedding (256 dim)
    ↓
Dense(256 → 128) + ReLU + Dropout
    ↓
Dense(128 → 11) + Softmax
    ↓
11-class probability (risk 0-10)
```

## Inference Output

```json
{
    "risk_label": 7,
    "risk_score": 6.85,
    "probabilities": [0.01, 0.02, ...]
}
```

- `risk_label`: Argmax prediction (0-10)
- `risk_score`: Expected value Σ(k × p[k])
- `probabilities`: 11-dim probability vector

## Performance

- **Model size**: ~11M parameters
- **Inference**: ~5ms per log (CPU, batched)
- **Memory**: ~100MB for ONNX model

## License

MIT
