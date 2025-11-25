"""BERT-mini based log risk classifier."""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from .config import ModelConfig


class LogRiskClassifier(nn.Module):
    """
    Log risk classification model based on BERT-mini.

    Architecture:
        BERT-mini encoder → [CLS] token → Dense(H→H/2) → ReLU → Dropout → Dense(H/2→11)
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize classifier.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Load BERT model
        self.bert = BertModel.from_pretrained(config.model_name)
        hidden_size = self.bert.config.hidden_size  # 256 for bert-mini

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size // 2, config.num_labels),
        )

        # Initialize classifier weights
        self._init_weights()

    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs (optional)

        Returns:
            Logits [batch_size, num_labels]
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Classification
        logits = self.classifier(cls_output)  # [batch_size, num_labels]

        return logits

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Get probability distribution over risk levels.

        Returns:
            Probabilities [batch_size, num_labels]
        """
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.softmax(logits, dim=-1)

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Get predicted risk labels.

        Returns:
            Labels [batch_size]
        """
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.argmax(logits, dim=-1)

    def get_expected_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Get expected risk score as weighted sum of probabilities.

        E[score] = sum(k * p[k]) for k in 0..10

        Returns:
            Expected scores [batch_size]
        """
        probs = self.predict_proba(input_ids, attention_mask, token_type_ids)
        scores = torch.arange(self.config.num_labels, device=probs.device, dtype=probs.dtype)
        return torch.sum(probs * scores, dim=-1)

    def freeze_bert(self):
        """Freeze BERT parameters (for feature extraction mode)."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        """Unfreeze BERT parameters (for fine-tuning mode)."""
        for param in self.bert.parameters():
            param.requires_grad = True

    def get_num_parameters(self) -> dict:
        """Get parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bert_params = sum(p.numel() for p in self.bert.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "bert": bert_params,
            "classifier": classifier_params,
        }
