#!/usr/bin/env python3
"""CLI script for exporting model to ONNX."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import export_to_onnx
from src.inference.export import export_from_checkpoint, optimize_onnx


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "--checkpoint", "-c", required=True, help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output path for ONNX model"
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        default=None,
        help="Path to tokenizer (defaults to checkpoint_dir/tokenizer)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max sequence length (default: 128)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply ONNX optimizations for BERT",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX model verification",
    )

    args = parser.parse_args()

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Export to ONNX
    print(f"Exporting checkpoint: {args.checkpoint}")
    output_path = export_from_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        max_length=args.max_length,
    )

    # Optimize if requested
    if args.optimize:
        optimized_path = str(Path(args.output).with_suffix(".optimized.onnx"))
        try:
            optimize_onnx(output_path, optimized_path)
            print(f"Optimized model saved to: {optimized_path}")
        except ImportError:
            print("Warning: onnxruntime-tools not installed, skipping optimization")
        except Exception as e:
            print(f"Warning: Optimization failed: {e}")

    print(f"\nExport complete!")
    print(f"ONNX model: {output_path}")


if __name__ == "__main__":
    main()
