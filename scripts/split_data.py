#!/usr/bin/env python3
"""Split labeled data into train/val/test sets."""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Split labeled data into train/val/test")
    parser.add_argument("-i", "--input", required=True, help="Input labeled CSV file")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train set ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stratify", action="store_true", help="Stratified split by label")
    args = parser.parse_args()

    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Total samples: {len(df):,}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratification column
    stratify_col = df["label"] if args.stratify else None

    # First split: train vs (val + test)
    val_test_ratio = args.val_ratio + args.test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=args.seed,
        stratify=stratify_col,
    )

    # Second split: val vs test
    test_ratio_adjusted = args.test_ratio / val_test_ratio
    stratify_col_vt = val_test_df["label"] if args.stratify else None
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_ratio_adjusted,
        random_state=args.seed,
        stratify=stratify_col_vt,
    )

    # Save splits
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSplit complete:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%) → {train_path}")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%) → {val_path}")
    print(f"  Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%) → {test_path}")

    # Print label distribution
    print(f"\nLabel distribution:")
    print(f"  {'Label':<6} {'Train':>10} {'Val':>10} {'Test':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for label in sorted(df["label"].unique()):
        train_cnt = len(train_df[train_df["label"] == label])
        val_cnt = len(val_df[val_df["label"] == label])
        test_cnt = len(test_df[test_df["label"] == label])
        print(f"  {label:<6} {train_cnt:>10,} {val_cnt:>10,} {test_cnt:>10,}")


if __name__ == "__main__":
    main()
