#!/usr/bin/env python3
"""Evaluate model accuracy on test dataset."""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import LogRiskPredictor
from src.model import InferenceConfig


def load_test_data(path: str) -> pd.DataFrame:
    """Load test dataset."""
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")
    return df


def evaluate(
    model_path: str,
    tokenizer_path: str,
    test_data_path: str,
    batch_size: int = 64,
    output_path: str = None,
):
    """Run evaluation and compute metrics."""
    print("=" * 70)
    print("Model Accuracy Evaluation")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {model_path}")
    config = InferenceConfig(num_threads=4)
    predictor = LogRiskPredictor(model_path, tokenizer_path, config)

    # Load test data
    print(f"Loading test data: {test_data_path}")
    df = load_test_data(test_data_path)
    texts = df["text"].tolist()
    labels_true = df["label"].tolist()
    print(f"Test samples: {len(texts):,}")

    # Run predictions
    print("\nRunning predictions...")
    labels_pred = []
    scores_pred = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = predictor.predict(batch)
        for r in results:
            labels_pred.append(r["risk_label"])
            scores_pred.append(r.get("risk_score", float(r["risk_label"])))

        if (i + batch_size) % 10000 == 0:
            print(f"  Processed {min(i + batch_size, len(texts)):,} / {len(texts):,}")

    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    scores_pred = np.array(scores_pred)

    # Compute metrics
    results = compute_metrics(labels_true, labels_pred, scores_pred)

    # Print results
    print_results(results, labels_true, labels_pred)

    # Save results
    if output_path:
        save_results(results, output_path)
        print(f"\nResults saved to: {output_path}")

    return results


def compute_metrics(labels_true, labels_pred, scores_pred):
    """Compute all evaluation metrics."""
    results = {}

    # Basic accuracy
    results["accuracy"] = accuracy_score(labels_true, labels_pred)
    results["accuracy_pct"] = results["accuracy"] * 100

    # Per-class metrics
    results["precision_macro"] = precision_score(labels_true, labels_pred, average="macro", zero_division=0)
    results["recall_macro"] = recall_score(labels_true, labels_pred, average="macro", zero_division=0)
    results["f1_macro"] = f1_score(labels_true, labels_pred, average="macro", zero_division=0)

    results["precision_weighted"] = precision_score(labels_true, labels_pred, average="weighted", zero_division=0)
    results["recall_weighted"] = recall_score(labels_true, labels_pred, average="weighted", zero_division=0)
    results["f1_weighted"] = f1_score(labels_true, labels_pred, average="weighted", zero_division=0)

    # Confusion matrix
    results["confusion_matrix"] = confusion_matrix(labels_true, labels_pred).tolist()

    # Per-class report
    results["classification_report"] = classification_report(
        labels_true, labels_pred, output_dict=True, zero_division=0
    )

    # High-risk detection (label >= 7)
    high_risk_true = labels_true >= 7
    high_risk_pred = labels_pred >= 7

    results["high_risk_accuracy"] = accuracy_score(high_risk_true, high_risk_pred)
    results["high_risk_precision"] = precision_score(high_risk_true, high_risk_pred, zero_division=0)
    results["high_risk_recall"] = recall_score(high_risk_true, high_risk_pred, zero_division=0)
    results["high_risk_f1"] = f1_score(high_risk_true, high_risk_pred, zero_division=0)

    # Critical detection (label >= 9)
    critical_true = labels_true >= 9
    critical_pred = labels_pred >= 9

    results["critical_precision"] = precision_score(critical_true, critical_pred, zero_division=0)
    results["critical_recall"] = recall_score(critical_true, critical_pred, zero_division=0)
    results["critical_f1"] = f1_score(critical_true, critical_pred, zero_division=0)

    # Score-based metrics (MAE, RMSE for expected score)
    results["mae"] = float(np.mean(np.abs(labels_true - scores_pred)))
    results["rmse"] = float(np.sqrt(np.mean((labels_true - scores_pred) ** 2)))

    # Off-by-one accuracy (prediction within ±1 of true label)
    within_one = np.abs(labels_true - labels_pred) <= 1
    results["accuracy_within_1"] = float(np.mean(within_one))

    # Off-by-two accuracy
    within_two = np.abs(labels_true - labels_pred) <= 2
    results["accuracy_within_2"] = float(np.mean(within_two))

    # Distribution comparison
    results["label_distribution_true"] = dict(Counter(labels_true.tolist()))
    results["label_distribution_pred"] = dict(Counter(labels_pred.tolist()))

    return results


def print_results(results, labels_true, labels_pred):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n[Overall Accuracy]")
    print(f"  Exact match:    {results['accuracy_pct']:.2f}%")
    print(f"  Within ±1:      {results['accuracy_within_1'] * 100:.2f}%")
    print(f"  Within ±2:      {results['accuracy_within_2'] * 100:.2f}%")

    print("\n[Score Metrics]")
    print(f"  MAE:            {results['mae']:.3f}")
    print(f"  RMSE:           {results['rmse']:.3f}")

    print("\n[Macro Averages]")
    print(f"  Precision:      {results['precision_macro']:.4f}")
    print(f"  Recall:         {results['recall_macro']:.4f}")
    print(f"  F1 Score:       {results['f1_macro']:.4f}")

    print("\n[High-Risk Detection (label >= 7)]")
    print(f"  Accuracy:       {results['high_risk_accuracy'] * 100:.2f}%")
    print(f"  Precision:      {results['high_risk_precision']:.4f}")
    print(f"  Recall:         {results['high_risk_recall']:.4f}")
    print(f"  F1 Score:       {results['high_risk_f1']:.4f}")

    print("\n[Critical Detection (label >= 9)]")
    print(f"  Precision:      {results['critical_precision']:.4f}")
    print(f"  Recall:         {results['critical_recall']:.4f}")
    print(f"  F1 Score:       {results['critical_f1']:.4f}")

    print("\n[Per-Class Report]")
    print(classification_report(labels_true, labels_pred, zero_division=0))

    print("[Confusion Matrix]")
    cm = np.array(results["confusion_matrix"])
    print("     " + " ".join(f"{i:>4}" for i in range(11)))
    print("     " + "-" * 55)
    for i, row in enumerate(cm):
        print(f"{i:>3} |" + " ".join(f"{v:>4}" for v in row))

    print("\n" + "=" * 70)


def save_results(results, output_path: str):
    """Save results to JSON file."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy")
    parser.add_argument("-m", "--model", required=True, help="Path to ONNX model")
    parser.add_argument("-t", "--tokenizer", required=True, help="Path to tokenizer")
    parser.add_argument("-d", "--data", required=True, help="Path to test CSV (text, label columns)")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("-o", "--output", help="Output JSON file for results")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        test_data_path=args.data,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
