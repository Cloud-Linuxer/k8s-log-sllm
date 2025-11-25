# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**k8s-log-sllm** is a log risk scoring system that classifies Kubernetes, system, and application logs into risk levels (0-10) using a fine-tuned BERT-mini transformer model.

- **Input**: Single-line log string
- **Output**: Integer risk score 0-10 (11-class classification)
- **Training**: GPU-enabled fine-tuning
- **Inference**: CPU-optimized via ONNX Runtime

## Architecture

```
src/
├── preprocessing/     # Log text preprocessing (token replacement)
├── labeling/          # Weak label generation (level + keywords)
├── data/              # PyTorch Dataset
├── model/             # BERT-mini + classification head
├── training/          # Fine-tuning pipeline
└── inference/         # ONNX export & CPU inference

scripts/               # CLI tools (preprocess, label, train, export, predict)
tests/                 # Unit tests & E2E tests
```

## Commands

```bash
# Install
pip install -r requirements.txt

# Preprocessing
python scripts/preprocess.py -i raw_logs.txt -o processed.csv

# Labeling
python scripts/label.py -i processed.csv -o labeled.csv --details

# Training
python scripts/train.py -d labeled.csv -o models/v1 --epochs 5 --batch-size 32

# Export ONNX
python scripts/export.py -c models/v1/best_model.pt -o models/v1/model.onnx

# Inference
python scripts/predict.py -m models/v1/model.onnx -t models/v1/tokenizer -i logs.txt --preprocess

# Tests
pytest tests/ -v
pytest tests/test_preprocessor.py -v
```

## Model Specifications

- **Base model**: `prajjwal1/bert-mini` (256 hidden, 4 layers, ~11M params)
- **Max sequence length**: 128 tokens
- **Classification head**: Dense(256→128, ReLU) → Dropout → Dense(128→11)
- **Loss**: CrossEntropyLoss
- **Optimizer**: AdamW (lr: 2e-5 to 5e-5, warmup: 10% of steps)

## Labeling Policy

**Base scores by log level:**
| Level | Score |
|-------|-------|
| TRACE/DEBUG | 0-1 |
| INFO | 2 |
| WARN/WARNING | 4-5 |
| ERROR | 6 |
| CRITICAL/ALERT | 8 |
| FATAL/PANIC/EMERG | 9-10 |

**Keyword bonuses (max +5):**
- K8s: CrashLoopBackOff, ImagePullBackOff, node not ready (+2)
- System: kernel panic, segfault, I/O error (+3)
- Network: connection refused (+1), upstream timed out (+2)
- Security: permission denied, unauthorized (+2)

## Preprocessing Tokens

- Timestamps → `<TIME>`
- IP addresses → `<IP>`
- UUIDs/hex IDs → `<ID>`
- PIDs → `<PID>`
- Long paths → `<PATH>`

## Key Files

| File | Purpose |
|------|---------|
| `src/preprocessing/preprocessor.py` | LogPreprocessor class |
| `src/labeling/labeler.py` | RiskLabeler class |
| `src/model/classifier.py` | LogRiskClassifier (BERT + head) |
| `src/model/config.py` | ModelConfig, TrainingConfig, InferenceConfig |
| `src/training/trainer.py` | Trainer class |
| `src/inference/predictor.py` | LogRiskPredictor (ONNX) |
| `src/inference/export.py` | export_to_onnx() |
| `src/data/dataset.py` | LogRiskDataset |

## Key Metrics

- Overall accuracy
- High-risk recall/precision (label ≥ 7)
- Validation loss for early stopping
