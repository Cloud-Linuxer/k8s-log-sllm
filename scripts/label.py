#!/usr/bin/env python3
"""CLI script for generating weak labels."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labeling import RiskLabeler


def main():
    parser = argparse.ArgumentParser(description="Generate weak labels for preprocessed logs")
    parser.add_argument(
        "--input", "-i", required=True, help="Input CSV file (from preprocess.py)"
    )
    parser.add_argument("--output", "-o", required=True, help="Output CSV file with labels")
    parser.add_argument(
        "--max-bonus",
        type=int,
        default=5,
        help="Maximum keyword bonus (default: 5)",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include detailed label breakdown",
    )

    args = parser.parse_args()

    # Initialize labeler
    labeler = RiskLabeler(max_bonus=args.max_bonus)

    # Read input CSV
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Read {len(df)} entries from {args.input}")

    # Validate required columns
    if "text" not in df.columns:
        print("Error: Input CSV must have 'text' column")
        sys.exit(1)

    # Generate labels
    if args.details:
        details = []
        for _, row in df.iterrows():
            level_raw = row.get("level_raw") if pd.notna(row.get("level_raw")) else None
            detail = labeler.label_with_details(row["text"], level_raw)
            details.append(detail)

        df["label"] = [d["label"] for d in details]
        df["base_score"] = [d["base_score"] for d in details]
        df["bonus"] = [d["capped_bonus"] for d in details]
        df["matched_keywords"] = [
            ",".join([k for k, _ in d["matched_keywords"]]) for d in details
        ]
    else:
        data = [
            {"text": row["text"], "level_raw": row.get("level_raw") if pd.notna(row.get("level_raw")) else None}
            for _, row in df.iterrows()
        ]
        df["label"] = labeler.label_batch(data)

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} labeled entries to {args.output}")

    # Print label distribution
    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    # Print sample
    print("\nSample output:")
    print(df[["text", "label"]].head(5).to_string())


if __name__ == "__main__":
    main()
