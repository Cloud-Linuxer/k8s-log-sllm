"""Export model to ONNX format for CPU inference."""

from pathlib import Path
from typing import Optional

import torch
import onnx
from transformers import BertTokenizer

from src.model import LogRiskClassifier, ModelConfig


def export_to_onnx(
    model: LogRiskClassifier,
    tokenizer: BertTokenizer,
    output_path: str,
    max_length: int = 128,
    opset_version: int = 14,
    verify: bool = True,
) -> str:
    """
    Export model to ONNX format.

    Args:
        model: Trained LogRiskClassifier
        tokenizer: BERT tokenizer
        output_path: Path to save ONNX model
        max_length: Maximum sequence length
        opset_version: ONNX opset version
        verify: Whether to verify exported model

    Returns:
        Path to exported model
    """
    model.eval()

    # Create dummy input
    dummy_text = "sample log message for export"
    dummy_input = tokenizer(
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"Exported model to {output_path}")

    # Verify model
    if verify:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")

    return str(output_path)


def export_from_checkpoint(
    checkpoint_path: str,
    output_path: str,
    tokenizer_path: Optional[str] = None,
    max_length: int = 128,
) -> str:
    """
    Export model from checkpoint to ONNX.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save ONNX model
        tokenizer_path: Path to tokenizer (defaults to checkpoint dir)
        max_length: Maximum sequence length

    Returns:
        Path to exported model
    """
    checkpoint_path = Path(checkpoint_path)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Reconstruct model config
    model_config_dict = checkpoint["config"]["model_config"]
    model_config = ModelConfig(**model_config_dict)

    # Create and load model
    model = LogRiskClassifier(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = checkpoint_path.parent / "tokenizer"
    tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))

    # Export
    return export_to_onnx(model, tokenizer, output_path, max_length)


def optimize_onnx(input_path: str, output_path: str) -> str:
    """
    Optimize ONNX model for inference.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model

    Returns:
        Path to optimized model
    """
    from onnxruntime.transformers import optimizer

    optimized_model = optimizer.optimize_model(
        input_path,
        model_type="bert",
        num_heads=4,  # bert-mini has 4 attention heads
        hidden_size=256,  # bert-mini hidden size
    )
    optimized_model.save_model_to_file(output_path)
    print(f"Optimized model saved to {output_path}")
    return output_path
