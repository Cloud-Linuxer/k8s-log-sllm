"""Training pipeline for log risk classifier."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.model import LogRiskClassifier, ModelConfig, TrainingConfig
from src.data import LogRiskDataset


class Trainer:
    """Trainer for log risk classification model."""

    def __init__(
        self,
        model: LogRiskClassifier,
        train_dataset: LogRiskDataset,
        val_dataset: Optional[LogRiskDataset] = None,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: LogRiskClassifier model
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.model = model

        # Setup device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Class weights for imbalanced data
        class_weights = None
        if self.config.use_class_weights:
            class_weights = train_dataset.get_class_weights().to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_epsilon,
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training state
        self.best_val_loss = float("inf")
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.current_epoch = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_high_risk_recall": [],
            "val_high_risk_precision": [],
        }

    def train(self, output_dir: Optional[str] = None) -> dict:
        """
        Run training loop.

        Args:
            output_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Training epoch
            train_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)

            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            # Validation
            if self.val_loader:
                val_metrics = self._evaluate()
                self.history["val_loss"].append(val_metrics["val_loss"])
                self.history["val_accuracy"].append(val_metrics["accuracy"])
                self.history["val_high_risk_recall"].append(val_metrics["high_risk_recall"])
                self.history["val_high_risk_precision"].append(val_metrics["high_risk_precision"])

                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  High-Risk Recall: {val_metrics['high_risk_recall']:.4f}")
                print(f"  High-Risk Precision: {val_metrics['high_risk_precision']:.4f}")

                # Early stopping check
                if self._check_early_stopping(val_metrics):
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

                # Save best model
                if output_dir and val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self._save_checkpoint(output_path / "best_model.pt")

            # Save checkpoint
            if output_dir and (epoch + 1) % max(1, self.config.epochs // 5) == 0:
                self._save_checkpoint(output_path / f"checkpoint_epoch_{epoch + 1}.pt")

        # Save final model and history
        if output_dir:
            self._save_checkpoint(output_path / "final_model.pt")
            with open(output_path / "training_history.json", "w") as f:
                json.dump(self.history, f, indent=2)

        return self.history

    def _train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch + 1}")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        return total_loss / len(self.train_loader)

    def _evaluate(self) -> dict:
        """Run evaluation on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)

        # High-risk metrics (label >= 7)
        high_risk_mask = all_labels >= 7
        high_risk_pred_mask = all_preds >= 7

        if high_risk_mask.sum() > 0:
            high_risk_recall = recall_score(
                high_risk_mask, high_risk_pred_mask, zero_division=0
            )
            high_risk_precision = precision_score(
                high_risk_mask, high_risk_pred_mask, zero_division=0
            )
        else:
            high_risk_recall = 0.0
            high_risk_precision = 0.0

        return {
            "val_loss": total_loss / len(self.val_loader),
            "accuracy": accuracy,
            "high_risk_recall": high_risk_recall,
            "high_risk_precision": high_risk_precision,
            "predictions": all_preds,
            "labels": all_labels,
        }

    def _check_early_stopping(self, val_metrics: dict) -> bool:
        """Check if early stopping should trigger."""
        metric_name = self.config.early_stopping_metric

        if metric_name == "val_loss":
            current = val_metrics["val_loss"]
            is_better = current < self.best_val_loss
        else:
            current = val_metrics.get(metric_name, val_metrics["accuracy"])
            is_better = current > self.best_val_metric

        if is_better:
            if metric_name == "val_loss":
                self.best_val_loss = current
            else:
                self.best_val_metric = current
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience

    def _save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": {
                "model_config": self.model.config.__dict__,
                "training_config": self.config.__dict__,
            },
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        print(f"Loaded checkpoint from {path}")

    def get_detailed_evaluation(self) -> dict:
        """Get detailed evaluation metrics including classification report."""
        if not self.val_loader:
            raise ValueError("Validation dataset required for detailed evaluation")

        val_metrics = self._evaluate()

        # Generate classification report
        report = classification_report(
            val_metrics["labels"],
            val_metrics["predictions"],
            labels=list(range(11)),
            target_names=[f"Risk {i}" for i in range(11)],
            output_dict=True,
            zero_division=0,
        )

        # Confusion matrix
        cm = confusion_matrix(
            val_metrics["labels"],
            val_metrics["predictions"],
            labels=list(range(11)),
        )

        return {
            **val_metrics,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
