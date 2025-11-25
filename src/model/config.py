"""Configuration classes for model, training, and inference."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    model_name: str = "prajjwal1/bert-mini"  # HuggingFace model name
    max_seq_length: int = 128
    num_labels: int = 11  # 0-10 risk scores
    hidden_dropout: float = 0.1
    classifier_dropout: float = 0.1

    def __post_init__(self):
        if self.num_labels < 2:
            raise ValueError("num_labels must be at least 2")
        if self.max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    # Optimizer settings
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8

    # Training settings
    epochs: int = 5
    batch_size: int = 32
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 2
    early_stopping_metric: str = "val_loss"  # val_loss, accuracy, high_risk_recall

    # Logging and saving
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50

    # Data split
    val_split: float = 0.1
    test_split: float = 0.1

    # Device
    device: str = "cuda"  # cuda, cpu, mps

    # Random seed
    seed: int = 42

    # Class weights for imbalanced data
    use_class_weights: bool = False

    def __post_init__(self):
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if self.val_split + self.test_split >= 1:
            raise ValueError("val_split + test_split must be less than 1")


@dataclass
class InferenceConfig:
    """Inference configuration."""

    # Batch processing
    batch_size: int = 32

    # ONNX Runtime settings
    num_threads: int = 4
    graph_optimization_level: str = "all"  # all, basic, extended, disabled

    # Streaming settings
    streaming_interval: float = 1.0  # seconds
    streaming_batch_size: int = 32

    # Output settings
    return_probabilities: bool = True
    return_expected_score: bool = True  # E[score] = sum(k * p[k])

    # Model paths
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive")
