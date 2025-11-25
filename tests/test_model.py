"""Tests for model architecture."""

import pytest
import torch

from src.model import ModelConfig, LogRiskClassifier
from src.data import LogRiskDataset


@pytest.fixture
def model_config():
    return ModelConfig(
        model_name="prajjwal1/bert-mini",
        max_seq_length=128,
        num_labels=11,
    )


@pytest.fixture
def model(model_config):
    return LogRiskClassifier(model_config)


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.model_name == "prajjwal1/bert-mini"
        assert config.max_seq_length == 128
        assert config.num_labels == 11

    def test_invalid_num_labels(self):
        with pytest.raises(ValueError):
            ModelConfig(num_labels=1)

    def test_invalid_seq_length(self):
        with pytest.raises(ValueError):
            ModelConfig(max_seq_length=0)


class TestLogRiskClassifier:
    def test_model_creation(self, model):
        assert model is not None
        assert hasattr(model, "bert")
        assert hasattr(model, "classifier")

    def test_forward_shape(self, model):
        batch_size = 4
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        logits = model(input_ids, attention_mask)
        assert logits.shape == (batch_size, 11)

    def test_predict_proba_shape(self, model):
        batch_size = 4
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        probs = model.predict_proba(input_ids, attention_mask)
        assert probs.shape == (batch_size, 11)
        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_predict_shape(self, model):
        batch_size = 4
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        labels = model.predict(input_ids, attention_mask)
        assert labels.shape == (batch_size,)
        assert labels.dtype == torch.int64
        assert (labels >= 0).all() and (labels < 11).all()

    def test_expected_score_range(self, model):
        batch_size = 4
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        scores = model.get_expected_score(input_ids, attention_mask)
        assert scores.shape == (batch_size,)
        # Expected score should be between 0 and 10
        assert (scores >= 0).all() and (scores <= 10).all()

    def test_freeze_unfreeze_bert(self, model):
        model.freeze_bert()
        for param in model.bert.parameters():
            assert not param.requires_grad

        model.unfreeze_bert()
        for param in model.bert.parameters():
            assert param.requires_grad

    def test_parameter_count(self, model):
        params = model.get_num_parameters()
        assert "total" in params
        assert "trainable" in params
        assert "bert" in params
        assert "classifier" in params
        assert params["total"] == params["bert"] + params["classifier"]


class TestLogRiskDataset:
    def test_dataset_creation(self):
        texts = ["error log message", "info log message", "warning log"]
        labels = [6, 2, 4]
        dataset = LogRiskDataset(texts, labels)

        assert len(dataset) == 3

    def test_dataset_item_structure(self):
        texts = ["error log message"]
        labels = [6]
        dataset = LogRiskDataset(texts, labels, max_length=64)

        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].shape == (64,)
        assert item["attention_mask"].shape == (64,)

    def test_dataset_without_labels(self):
        texts = ["error log message", "info log message"]
        dataset = LogRiskDataset(texts, labels=None)

        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" not in item

    def test_class_weights(self):
        texts = ["a"] * 10 + ["b"] * 5 + ["c"] * 3
        labels = [0] * 10 + [5] * 5 + [10] * 3
        dataset = LogRiskDataset(texts, labels)

        weights = dataset.get_class_weights()
        assert weights.shape == (11,)
        # Class 0 should have lower weight (more samples)
        # Class 10 should have higher weight (fewer samples)
        assert weights[0] < weights[10]

    def test_label_distribution(self):
        texts = ["a"] * 5 + ["b"] * 3
        labels = [0] * 5 + [5] * 3
        dataset = LogRiskDataset(texts, labels)

        dist = dataset.get_label_distribution()
        assert dist[0] == 5
        assert dist[5] == 3
        assert dist[1] == 0


class TestDatasetModelIntegration:
    def test_dataset_with_model(self, model):
        texts = ["error connection refused", "info started service"]
        labels = [6, 2]
        dataset = LogRiskDataset(texts, labels)

        # Get batch
        item = dataset[0]
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)

        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        assert logits.shape == (1, 11)
