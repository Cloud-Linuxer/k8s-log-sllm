from .config import ModelConfig, TrainingConfig, InferenceConfig

__all__ = ["ModelConfig", "TrainingConfig", "InferenceConfig"]

# Lazy import for classifier (requires torch)
def __getattr__(name):
    if name == "LogRiskClassifier":
        from .classifier import LogRiskClassifier
        return LogRiskClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
