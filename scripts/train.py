#!/usr/bin/env python3
"""CLI script for training the log risk classifier."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ModelConfig, TrainingConfig, LogRiskClassifier
from src.data import LogRiskDataset
from src.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train log risk classifier")
    parser.add_argument(
        "--data", "-d", required=True, help="Input CSV file with 'text' and 'label' columns"
    )
    parser.add_argument("--output", "-o", required=True, help="Output directory for checkpoints")

    # Model arguments
    parser.add_argument(
        "--model-name",
        default="prajjwal1/bert-mini",
        help="HuggingFace model name (default: prajjwal1/bert-mini)",
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="Max sequence length (default: 128)"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-5, help="Learning rate (default: 3e-5)"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1)"
    )
    parser.add_argument(
        "--early-stopping", type=int, default=2, help="Early stopping patience (default: 2)"
    )
    parser.add_argument(
        "--class-weights", action="store_true", help="Use class weights for imbalanced data"
    )

    # Data split arguments
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.1, help="Test split ratio (default: 0.1)"
    )

    # Device arguments
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {args.data}")

    # Validate columns
    if "text" not in df.columns or "label" not in df.columns:
        print("Error: CSV must have 'text' and 'label' columns")
        sys.exit(1)

    # Split data
    train_df, temp_df = train_test_split(
        df, test_size=args.val_split + args.test_split, random_state=args.seed, stratify=df["label"]
    )

    if args.test_split > 0:
        val_ratio = args.val_split / (args.val_split + args.test_split)
        val_df, test_df = train_test_split(
            temp_df, test_size=1 - val_ratio, random_state=args.seed, stratify=temp_df["label"]
        )
        print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    else:
        val_df = temp_df
        test_df = None
        print(f"Split: train={len(train_df)}, val={len(val_df)}")

    # Print label distribution
    print("\nLabel distribution (train):")
    print(train_df["label"].value_counts().sort_index())

    # Create configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        max_seq_length=args.max_length,
    )

    training_config = TrainingConfig(
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping,
        use_class_weights=args.class_weights,
        device=args.device,
        seed=args.seed,
    )

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = LogRiskDataset.from_dataframe(
        train_df,
        tokenizer=args.model_name,
        max_length=args.max_length,
    )
    val_dataset = LogRiskDataset.from_dataframe(
        val_df,
        tokenizer=args.model_name,
        max_length=args.max_length,
    )

    # Create model
    print("\nInitializing model...")
    model = LogRiskClassifier(model_config)
    params = model.get_num_parameters()
    print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
    )
    print(f"Training on device: {trainer.device}")

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    history = trainer.train(output_dir=args.output)

    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    eval_results = trainer.get_detailed_evaluation()
    print(f"Final Accuracy: {eval_results['accuracy']:.4f}")
    print(f"High-Risk Recall: {eval_results['high_risk_recall']:.4f}")
    print(f"High-Risk Precision: {eval_results['high_risk_precision']:.4f}")

    # Save tokenizer for inference
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(Path(args.output) / "tokenizer")
    print(f"\nTokenizer saved to {args.output}/tokenizer")

    print(f"\nTraining complete! Checkpoints saved to {args.output}")


if __name__ == "__main__":
    main()
