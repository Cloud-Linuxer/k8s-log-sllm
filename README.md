# k8s-log-sllm

K8s/System Log Risk Scoring using BERT-mini fine-tuning.

## Overview

This project classifies Kubernetes, system, and application logs into risk levels (0-10) using a fine-tuned `google/bert-mini` (prajjwal1/bert-mini) transformer model.

- **Input**: Single-line log string
- **Output**: Risk score 0-10 (11-class classification)
- **Training**: GPU-enabled fine-tuning (1M samples, 100% accuracy)
- **Inference**: CPU-optimized via ONNX Runtime

## Quick Start

### Docker (Recommended)

```bash
# Build and run
docker build -t k8s-log-scorer .
docker run -d -p 8000:8000 k8s-log-scorer

# Or use docker-compose
docker-compose up -d

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"log": "ERROR kernel panic - not syncing"}'
```

### Local Installation

```bash
pip install -r requirements.txt

# Start API server
python scripts/serve.py --port 8000 --threads 4
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{"log": "ERROR connection refused to database"}

# Response
{"risk_label": 6, "risk_score": 6.0, "level": "error"}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{"logs": ["INFO started", "ERROR failed", "CRITICAL panic"]}

# Response
{"results": [...], "count": 3}
```

### Swagger Docs
```
http://localhost:8000/docs
```

## Python Usage

```python
from src.inference import LogRiskPredictor
from src.preprocessing import LogPreprocessor

# Load model
predictor = LogRiskPredictor(
    model_path='models/v2/model.onnx',
    tokenizer_path='models/v2/tokenizer'
)
preprocessor = LogPreprocessor()

# Predict
log = "ERROR kubelet Failed to create pod sandbox"
processed = preprocessor.preprocess(log)
result = predictor.predict_single(processed['text'])

print(result)
# {'risk_label': 7, 'risk_score': 6.85, 'probabilities': [...]}
```

## CLI Usage

```bash
# Score logs from file
python scripts/predict.py \
  -m models/v2/model.onnx \
  -t models/v2/tokenizer \
  -i logs.txt \
  --preprocess \
  --format simple

# Output:
# [2] INFO nginx GET /health 200
# [10] CRITICAL kernel panic - not syncing
# [7] ERROR connection refused
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

## Performance

| Metric | Value |
|--------|-------|
| Single Inference | 3.56 ms |
| Throughput (1 thread) | 280 logs/sec |
| Throughput (8 threads) | 1,370 logs/sec |
| Model Memory | ~850 MB |
| Docker Image | 556 MB |
| ONNX File | 42.6 MB |

See [docs/BENCHMARK.md](docs/BENCHMARK.md) for detailed benchmarks.

## Training

### Generate Training Data

```bash
python scripts/generate_data.py \
  --count 1000000 \
  --output data/raw/logs.txt \
  --balanced

python scripts/label.py \
  --input data/raw/logs.txt \
  --output data/processed/logs.labeled.csv
```

### Train Model

```bash
python scripts/train.py \
  --data data/processed/logs.labeled.csv \
  --output models/v2 \
  --epochs 5 \
  --batch-size 128 \
  --lr 3e-5
```

### Export to ONNX

```bash
python scripts/export.py \
  --checkpoint models/v2/final_model.pt \
  --output models/v2/model.onnx
```

## Project Structure

```
k8s-log-sllm/
├── src/
│   ├── api/              # FastAPI server
│   ├── preprocessing/    # Log preprocessing
│   ├── labeling/         # Weak label generation
│   ├── data/             # Dataset utilities
│   ├── model/            # BERT classifier
│   ├── training/         # Training pipeline
│   └── inference/        # ONNX inference
├── scripts/              # CLI tools
├── tests/                # Unit & E2E tests
├── data/
│   ├── raw/              # Original log files
│   ├── processed/        # Labeled CSV files
│   └── splits/           # Train/val/test splits
├── models/               # Saved models
├── docs/                 # Documentation
├── Dockerfile
└── docker-compose.yml
```

## Model Architecture

```
BERT-mini (prajjwal1/bert-mini, ~11M params)
    ↓
[CLS] token embedding (256 dim)
    ↓
Dense(256 → 128) + ReLU + Dropout
    ↓
Dense(128 → 11) + Softmax
    ↓
11-class probability (risk 0-10)
```

## Labeling Policy

**Base scores by log level:**
- TRACE/DEBUG: 0-1
- INFO: 2
- NOTICE/VERBOSE: 3
- WARN/WARNING: 4-5
- ERROR: 6-7
- CRITICAL/ALERT: 8-9
- FATAL/PANIC/EMERG: 9-10

**Keyword bonuses (max +5):**
- K8s: `CrashLoopBackOff`, `ImagePullBackOff`, `node not ready` (+2)
- System: `kernel panic`, `segfault`, `I/O error` (+3)
- Network: `connection refused` (+1), `upstream timed out` (+2)
- Security: `permission denied`, `unauthorized` (+2)

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run API tests
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## License

MIT
