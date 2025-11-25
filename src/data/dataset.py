"""PyTorch Dataset for log risk classification."""

from typing import Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, PreTrainedTokenizer


class LogRiskDataset(Dataset):
    """
    Dataset for log risk classification.

    Handles tokenization and prepares inputs for BERT model.
    """

    def __init__(
        self,
        texts: list[str],
        labels: Optional[list[int]] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        max_length: int = 128,
    ):
        """
        Initialize dataset.

        Args:
            texts: List of preprocessed log strings
            labels: List of risk labels (0-10), None for inference
            tokenizer: Tokenizer or model name string
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        # Initialize tokenizer
        if tokenizer is None:
            tokenizer = "prajjwal1/bert-mini"

        if isinstance(tokenizer, str):
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        # Pre-tokenize all texts for efficiency
        self.encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Convert labels to tensor if provided
        if labels is not None:
            self.label_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            self.label_tensor = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }

        if self.label_tensor is not None:
            item["labels"] = self.label_tensor[idx]

        return item

    @classmethod
    def from_dataframe(
        cls,
        df,
        text_column: str = "text",
        label_column: str = "label",
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        max_length: int = 128,
    ) -> "LogRiskDataset":
        """
        Create dataset from pandas DataFrame.

        Args:
            df: pandas DataFrame
            text_column: Name of text column
            label_column: Name of label column (None for inference)
            tokenizer: Tokenizer or model name
            max_length: Maximum sequence length

        Returns:
            LogRiskDataset instance
        """
        texts = df[text_column].tolist()
        labels = df[label_column].tolist() if label_column and label_column in df.columns else None

        return cls(texts, labels, tokenizer, max_length)

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced data.

        Uses inverse frequency weighting.

        Returns:
            Tensor of class weights [num_classes]
        """
        if self.label_tensor is None:
            raise ValueError("Labels required for class weights")

        # Count occurrences of each class
        num_classes = 11  # 0-10
        counts = torch.bincount(self.label_tensor, minlength=num_classes).float()

        # Avoid division by zero
        counts = torch.clamp(counts, min=1)

        # Inverse frequency weighting
        weights = len(self.label_tensor) / (num_classes * counts)

        # Normalize
        weights = weights / weights.sum() * num_classes

        return weights

    def get_label_distribution(self) -> dict:
        """
        Get distribution of labels.

        Returns:
            Dict mapping label to count
        """
        if self.label_tensor is None:
            return {}

        counts = torch.bincount(self.label_tensor, minlength=11)
        return {i: int(counts[i]) for i in range(11)}
