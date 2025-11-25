#!/usr/bin/env python3
"""CLI script for preprocessing log files."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import LogPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess log files for training")
    parser.add_argument(
        "--input", "-i", required=True, help="Input file (txt with one log per line)"
    )
    parser.add_argument("--output", "-o", required=True, help="Output CSV file")
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercase conversion",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum character length for truncation",
    )

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = LogPreprocessor(
        lowercase=not args.no_lowercase,
        max_length=args.max_length,
    )

    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        logs = [line.strip() for line in f if line.strip()]

    print(f"Read {len(logs)} log lines from {args.input}")

    # Process logs
    results = preprocessor.preprocess_batch(logs)

    # Create DataFrame
    df = pd.DataFrame(results)
    df["original"] = logs

    # Reorder columns
    df = df[["original", "text", "level_raw"]]

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} preprocessed logs to {args.output}")

    # Print sample
    print("\nSample output:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
