# K8s Log Risk Scoring System - Implementation Plan

## Overview

BERT-mini 기반 로그 위험도 분류 시스템 (0-10 스코어, 11-class classification)

```
Raw Log → Preprocessing → Labeling → Training → ONNX Export → CPU Inference
```

---

## Stage 1: 프로젝트 기반 구조

### Task 1.1: 디렉토리 구조 생성
```
k8s-log-sllm/
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── labeling/
│   │   ├── __init__.py
│   │   └── labeler.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── classifier.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── inference/
│       ├── __init__.py
│       ├── predictor.py
│       └── export.py
├── scripts/
│   ├── preprocess.py
│   ├── label.py
│   ├── train.py
│   ├── export.py
│   └── predict.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_labeler.py
│   ├── test_model.py
│   └── test_inference.py
├── data/
│   └── sample_logs.csv
└── models/
    └── .gitkeep
```

### Task 1.2: 의존성 파일 생성

**requirements.txt:**
```
torch>=2.0.0
transformers>=4.30.0
onnx>=1.14.0
onnxruntime>=1.15.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
pytest>=7.4.0
```

---

## Stage 2: 전처리 모듈

### Task 2.1: preprocessor.py

**핵심 기능:**
- 정규표현식 기반 토큰 치환
- 로그 레벨 추출
- 128 토큰 제한

**치환 패턴 (순서 중요):**
1. `TIMESTAMP` → `<TIME>` (ISO8601, syslog 형식)
2. `UUID` → `<ID>` (8-4-4-4-12 hex)
3. `IP` → `<IP>` (IPv4)
4. `LONG_HEX` → `<ID>` (12자리 이상 hex)
5. `PID/TID` → `<PID>`
6. `PATH` → `<PATH>` (3+ depth)

**클래스 설계:**
```python
class LogPreprocessor:
    PATTERNS = {
        'timestamp': r'\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}[.,]?\d*Z?',
        'uuid': r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        'ip': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        'long_hex': r'\b[0-9a-f]{12,}\b',
        'pid': r'\b(?:pid|PID|tid|TID)[=: ]?\d+\b|\[\d+\]',
        'path': r'(?:/[a-zA-Z0-9._-]+){3,}'
    }

    LOG_LEVEL_PATTERN = r'\b(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|CRITICAL|ALERT|FATAL|PANIC|EMERG)\b'

    def preprocess(self, log_line: str) -> dict:
        """Returns {'text': str, 'level_raw': str|None}"""

    def preprocess_batch(self, logs: list[str]) -> list[dict]:
        """Batch processing"""
```

### Task 2.2: test_preprocessor.py

**테스트 케이스:**
```python
def test_timestamp_replacement():
    # "2024-01-15T10:30:45.123Z kubelet ..." → "<TIME> kubelet ..."

def test_ip_replacement():
    # "connection from 192.168.1.100:8080" → "connection from <IP>:8080"

def test_uuid_replacement():
    # "pod-abc123-def456-..." → "pod-<ID>"

def test_log_level_extraction():
    # "[ERROR] failed" → level_raw = "ERROR"

def test_path_truncation():
    # "/var/log/kubernetes/pods/..." → "<PATH>"

def test_preserves_components():
    # "kubelet", "nginx", "systemd" 보존 확인
```

---

## Stage 3: 라벨링 모듈

### Task 3.1: labeler.py

**베이스 스코어 매핑:**
```python
LEVEL_BASE_SCORES = {
    'TRACE': 0, 'DEBUG': 1,
    'INFO': 2,
    'NOTICE': 3, 'VERBOSE': 3,
    'WARN': 4, 'WARNING': 5,
    'ERROR': 6,
    'CRITICAL': 8, 'ALERT': 8,
    'FATAL': 9, 'PANIC': 10, 'EMERG': 10
}
DEFAULT_BASE = 3  # 레벨 없을 때
```

**키워드 보너스:**
```python
KEYWORD_BONUSES = {
    # K8s Critical (+2)
    'crashloopbackoff': 2,
    'imagepullbackoff': 2,
    'back-off restarting': 2,
    'node not ready': 2,
    'notready': 2,
    'evicted': 2,
    'disk pressure': 2,
    'oomkilled': 2,

    # K8s Warning (+1~2)
    'failed to create pod sandbox': 2,
    'failed to create containerd task': 2,
    'context deadline exceeded': 1,
    'failed to pull image': 1,

    # System Critical (+3)
    'kernel panic': 3,
    'bug:': 3,
    'segfault': 3,
    'stack trace': 3,
    'i/o error': 3,
    'read-only file system': 3,
    'disk failure': 3,
    'out of memory': 3,

    # Network (+1~2)
    'connection refused': 1,
    'connection reset by peer': 1,
    'upstream timed out': 2,
    'no live upstreams': 2,
    'connection timed out': 1,

    # Security (+2)
    'permission denied': 2,
    'unauthorized': 2,
    'forbidden': 2,
    'authentication failed': 2,
    'access denied': 2,
}
MAX_BONUS = 5  # 보너스 상한
```

**클래스 설계:**
```python
class RiskLabeler:
    def label(self, text: str, level_raw: str = None) -> int:
        base = self._get_base_score(level_raw)
        bonus = self._calculate_keyword_bonus(text)
        return min(10, max(0, base + bonus))

    def label_batch(self, data: list[dict]) -> list[int]:
        """[{'text': ..., 'level_raw': ...}] → [label, ...]"""
```

### Task 3.2: 샘플 데이터셋 (data/sample_logs.csv)

다양한 로그 레벨과 패턴을 포함한 100-200개 샘플:
- K8s 로그 (kubelet, kube-apiserver, scheduler)
- 시스템 로그 (systemd, kernel, docker)
- 앱 로그 (nginx, application errors)

---

## Stage 4: 모델 아키텍처

### Task 4.1: config.py

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = "google/bert-mini"
    max_seq_length: int = 128
    num_labels: int = 11
    hidden_dropout: float = 0.1
    classifier_dropout: float = 0.1

@dataclass
class TrainingConfig:
    learning_rate: float = 3e-5
    epochs: int = 5
    batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    eval_steps: int = 100
    save_steps: int = 500

@dataclass
class InferenceConfig:
    batch_size: int = 32
    num_threads: int = 4
    streaming_interval: float = 1.0
```

### Task 4.2: classifier.py

```python
import torch
import torch.nn as nn
from transformers import BertModel

class LogRiskClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        hidden_size = self.bert.config.hidden_size  # 256 for bert-mini

        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size // 2, config.num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

    def predict_proba(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)
```

### Task 4.3: dataset.py

```python
from torch.utils.data import Dataset
from transformers import BertTokenizer

class LogRiskDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
```

---

## Stage 5: 학습 파이프라인

### Task 5.1: trainer.py

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        total_steps = len(train_loader) * config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * config.warmup_ratio),
            num_training_steps=total_steps
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train(self):
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch()
            val_metrics = self._evaluate()

            # Early stopping check
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def _evaluate(self):
        """Returns: val_loss, accuracy, high_risk_recall, high_risk_precision"""
```

### Task 5.2: 학습 스크립트 (scripts/train.py)

```python
# CLI interface
# python scripts/train.py \
#   --data data/labeled_logs.csv \
#   --output models/checkpoint \
#   --epochs 5 \
#   --batch-size 32 \
#   --lr 3e-5
```

---

## Stage 6: 추론 파이프라인

### Task 6.1: export.py

```python
def export_to_onnx(model, tokenizer, output_path, max_length=128):
    model.eval()

    dummy_input = tokenizer(
        "sample log",
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14
    )
```

### Task 6.2: predictor.py

```python
class LogRiskPredictor:
    def __init__(self, model_path, tokenizer_path, config: InferenceConfig):
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = config.num_threads

        self.session = InferenceSession(model_path, options)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.config = config

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Returns:
        [{
            'risk_label': int,      # 0-10
            'risk_score': float,    # expected value
            'probabilities': list   # 11-dim
        }, ...]
        """

    def predict_streaming(self, log_queue, callback):
        """
        Batch scoring for streaming logs.
        Collects logs up to batch_size, scores periodically.
        """
```

---

## Stage 7: 통합 및 문서화

### Task 7.1: End-to-end 테스트

```python
def test_full_pipeline():
    # 1. Raw log → Preprocess
    # 2. Preprocessed → Label (weak)
    # 3. Load model → Predict
    # 4. Compare weak label vs prediction
```

### Task 7.2: CLI 사용법

```bash
# Full pipeline
python scripts/preprocess.py --input raw.txt --output processed.csv
python scripts/label.py --input processed.csv --output labeled.csv
python scripts/train.py --data labeled.csv --output models/v1
python scripts/export.py --checkpoint models/v1 --output models/v1.onnx
python scripts/predict.py --model models/v1.onnx --input new_logs.txt
```

---

## Milestones

| Milestone | Deliverable | Criteria |
|-----------|-------------|----------|
| **M1** | 데이터 파이프라인 | 전처리 + 라벨링 → CSV 출력 가능 |
| **M2** | 학습 가능 | 모델 학습 → 체크포인트 저장 |
| **M3** | 추론 가능 | ONNX export → CPU 배치 추론 |

---

## Dependencies Graph

```
requirements.txt
       ↓
   Stage 1 (기반)
       ↓
   Stage 2 (전처리) ──→ Stage 3 (라벨링)
       ↓                    ↓
   Stage 4 (모델) ←─────────┘
       ↓
   Stage 5 (학습)
       ↓
   Stage 6 (추론)
       ↓
   Stage 7 (통합)
```
