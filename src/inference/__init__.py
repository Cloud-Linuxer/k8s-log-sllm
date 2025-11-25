from .predictor import LogRiskPredictor, AsyncLogRiskPredictor

__all__ = ["LogRiskPredictor", "AsyncLogRiskPredictor"]

# Lazy import for export (requires torch)
def __getattr__(name):
    if name == "export_to_onnx":
        from .export import export_to_onnx
        return export_to_onnx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
