"""End-to-end integration tests."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path


class TestPreprocessingToLabeling:
    """Test preprocessing → labeling pipeline."""

    def test_preprocess_and_label(self):
        from src.preprocessing import LogPreprocessor
        from src.labeling import RiskLabeler

        preprocessor = LogPreprocessor()
        labeler = RiskLabeler()

        logs = [
            "2024-01-15T10:30:45Z ERROR kubelet Failed to create pod sandbox: context deadline exceeded",
            "2024-01-15T10:30:46Z INFO nginx GET /health 200",
            "2024-01-15T10:30:47Z CRITICAL kernel kernel panic - not syncing",
        ]

        # Preprocess
        processed = preprocessor.preprocess_batch(logs)
        assert len(processed) == 3

        # Label
        labels = labeler.label_batch(processed)
        assert len(labels) == 3

        # Verify expected ranges
        assert labels[0] >= 6  # ERROR + deadline exceeded
        assert labels[1] <= 3  # INFO, normal
        assert labels[2] >= 9  # CRITICAL + kernel panic

    def test_full_csv_pipeline(self):
        """Test full pipeline with CSV files."""
        from src.preprocessing import LogPreprocessor
        from src.labeling import RiskLabeler

        preprocessor = LogPreprocessor()
        labeler = RiskLabeler()

        logs = [
            "2024-01-15T10:30:45Z ERROR crashloopbackoff detected",
            "2024-01-15T10:30:46Z WARNING connection refused",
            "2024-01-15T10:30:47Z INFO started successfully",
        ]

        # Preprocess
        processed = preprocessor.preprocess_batch(logs)
        df = pd.DataFrame(processed)
        df["original"] = logs

        # Label
        df["label"] = labeler.label_batch(processed)

        # Verify structure
        assert "text" in df.columns
        assert "level_raw" in df.columns
        assert "label" in df.columns
        assert all(0 <= l <= 10 for l in df["label"])


class TestDatasetCreation:
    """Test dataset creation from processed data."""

    def test_create_dataset_from_dataframe(self):
        from src.data import LogRiskDataset

        df = pd.DataFrame({
            "text": ["error message one", "info message two", "critical message"],
            "label": [6, 2, 9],
        })

        dataset = LogRiskDataset.from_dataframe(
            df, text_column="text", label_column="label"
        )

        assert len(dataset) == 3
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item


class TestModelForward:
    """Test model forward pass."""

    @pytest.fixture
    def model_and_dataset(self):
        from src.model import ModelConfig, LogRiskClassifier
        from src.data import LogRiskDataset

        config = ModelConfig(
            model_name="prajjwal1/bert-mini",
            max_seq_length=64,  # Shorter for testing
        )
        model = LogRiskClassifier(config)

        texts = ["error log", "info log"]
        labels = [6, 2]
        dataset = LogRiskDataset(texts, labels, max_length=64)

        return model, dataset

    def test_forward_pass(self, model_and_dataset):
        import torch

        model, dataset = model_and_dataset
        model.eval()

        item = dataset[0]
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        assert logits.shape == (1, 11)

    def test_predict_methods(self, model_and_dataset):
        import torch

        model, dataset = model_and_dataset
        model.eval()

        item = dataset[0]
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)

        with torch.no_grad():
            # Test predict
            label = model.predict(input_ids, attention_mask)
            assert 0 <= label.item() <= 10

            # Test predict_proba
            probs = model.predict_proba(input_ids, attention_mask)
            assert probs.shape == (1, 11)
            assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

            # Test expected score
            score = model.get_expected_score(input_ids, attention_mask)
            assert 0 <= score.item() <= 10


class TestRealWorldLogs:
    """Test with realistic log samples."""

    @pytest.fixture
    def realistic_logs(self):
        return [
            # K8s logs
            ("kubelet[12345]: Failed to create pod sandbox for pod default/nginx-abc123: rpc error: context deadline exceeded", "ERROR", 7, 9),
            ("kube-apiserver: I0115 10:30:45.123456 12345 handler.go:143] GET /api/v1/pods: (1.234567ms) 200", "INFO", 2, 3),
            ("kube-scheduler: W0115 10:30:45.123456 12345 scheduling.go:456] node node-1 is NotReady", "WARNING", 7, 9),

            # System logs
            ("Jan 15 10:30:45 server1 kernel: BUG: unable to handle page fault for address: 0x0", "ERROR", 9, 10),
            ("Jan 15 10:30:45 server1 systemd[1]: Started nginx.service.", "INFO", 2, 3),
            ("Jan 15 10:30:45 server1 sshd[12345]: Failed password for root from 192.168.1.100 port 22 ssh2", "WARNING", 7, 9),

            # Application logs
            ("2024-01-15T10:30:45.123Z ERROR [app] connection refused while connecting to database", "ERROR", 7, 8),
            ("2024-01-15T10:30:45.123Z INFO [app] Request completed successfully", "INFO", 2, 3),
            ("2024-01-15T10:30:45.123Z FATAL [app] Out of memory, terminating", "FATAL", 10, 10),
        ]

    def test_realistic_preprocessing(self, realistic_logs):
        from src.preprocessing import LogPreprocessor

        preprocessor = LogPreprocessor()

        for log, expected_level, _, _ in realistic_logs:
            result = preprocessor.preprocess(log)

            # Check level extraction
            assert result["level_raw"] == expected_level, f"Failed for: {log}"

            # Check tokens are replaced
            assert "192.168" not in result["text"]  # IP should be replaced
            assert "abc123" not in result["text"] or "<id>" in result["text"]

    def test_realistic_labeling(self, realistic_logs):
        from src.preprocessing import LogPreprocessor
        from src.labeling import RiskLabeler

        preprocessor = LogPreprocessor()
        labeler = RiskLabeler()

        for log, _, min_label, max_label in realistic_logs:
            processed = preprocessor.preprocess(log)
            label = labeler.label(processed["text"], processed["level_raw"])

            assert min_label <= label <= max_label, \
                f"Got {label} for '{log}', expected {min_label}-{max_label}"
