#!/usr/bin/env python3
"""CLI script for running inference."""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import InferenceConfig
from src.inference import LogRiskPredictor
from src.preprocessing import LogPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Run inference on logs")
    parser.add_argument(
        "--model", "-m", required=True, help="Path to ONNX model"
    )
    parser.add_argument(
        "--tokenizer", "-t", required=True, help="Path to tokenizer"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input file (one log per line) or single log string"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output JSON file (default: stdout)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Number of CPU threads (default: 4)"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Apply preprocessing to input"
    )
    parser.add_argument(
        "--no-probs", action="store_true", help="Don't include probabilities in output"
    )
    parser.add_argument(
        "--threshold", type=int, default=None, help="Only output logs with risk >= threshold"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "simple"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    if not Path(args.tokenizer).exists():
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        sys.exit(1)

    # Initialize config
    config = InferenceConfig(
        batch_size=args.batch_size,
        num_threads=args.threads,
        return_probabilities=not args.no_probs,
    )

    # Initialize predictor
    print(f"Loading model from {args.model}...", file=sys.stderr)
    predictor = LogRiskPredictor(args.model, args.tokenizer, config)

    # Initialize preprocessor if needed
    preprocessor = LogPreprocessor() if args.preprocess else None

    # Read input
    input_path = Path(args.input)
    if input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            logs = [line.strip() for line in f if line.strip()]
    else:
        # Treat as single log string
        logs = [args.input]

    print(f"Processing {len(logs)} logs...", file=sys.stderr)

    # Preprocess if needed
    original_logs = logs
    if preprocessor:
        processed = preprocessor.preprocess_batch(logs)
        logs = [p["text"] for p in processed]

    # Run inference
    results = predictor.predict_batch(logs)

    # Add original log to results
    for i, result in enumerate(results):
        result["log"] = original_logs[i]
        if preprocessor:
            result["preprocessed"] = logs[i]

    # Filter by threshold
    if args.threshold is not None:
        results = [r for r in results if r["risk_label"] >= args.threshold]

    # Format output
    if args.format == "json":
        output = json.dumps(results, indent=2)
    elif args.format == "csv":
        if results:
            headers = ["log", "risk_label", "risk_score"]
            lines = [",".join(headers)]
            for r in results:
                line = f'"{r["log"]}",{r["risk_label"]},{r.get("risk_score", "")}'
                lines.append(line)
            output = "\n".join(lines)
        else:
            output = "log,risk_label,risk_score"
    else:  # simple
        lines = []
        for r in results:
            score_str = f" ({r['risk_score']:.2f})" if "risk_score" in r else ""
            lines.append(f"[{r['risk_label']}]{score_str} {r['log'][:80]}")
        output = "\n".join(lines)

    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Print summary
    if results:
        labels = [r["risk_label"] for r in results]
        print(f"\nSummary:", file=sys.stderr)
        print(f"  Total: {len(results)}", file=sys.stderr)
        print(f"  Avg risk: {sum(labels)/len(labels):.2f}", file=sys.stderr)
        print(f"  High risk (>=7): {sum(1 for l in labels if l >= 7)}", file=sys.stderr)


if __name__ == "__main__":
    main()
