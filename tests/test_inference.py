"""Tests for inference pipeline."""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestSoftmax:
    def test_softmax_basic(self):
        from src.inference.predictor import softmax

        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)

        # Should sum to 1
        assert np.isclose(result.sum(), 1.0)
        # Should be in ascending order
        assert result[0] < result[1] < result[2]

    def test_softmax_batch(self):
        from src.inference.predictor import softmax

        x = np.array([[1.0, 2.0], [3.0, 1.0]])
        result = softmax(x, axis=-1)

        # Each row should sum to 1
        assert np.allclose(result.sum(axis=-1), [1.0, 1.0])


class TestInferenceConfig:
    def test_default_config(self):
        from src.model import InferenceConfig

        config = InferenceConfig()
        assert config.batch_size == 32
        assert config.num_threads == 4
        assert config.return_probabilities is True
        assert config.return_expected_score is True

    def test_invalid_batch_size(self):
        from src.model import InferenceConfig

        with pytest.raises(ValueError):
            InferenceConfig(batch_size=0)

    def test_invalid_threads(self):
        from src.model import InferenceConfig

        with pytest.raises(ValueError):
            InferenceConfig(num_threads=0)


class TestLogRiskPredictorMocked:
    """Test predictor with mocked ONNX session."""

    @pytest.fixture
    def mock_predictor(self):
        """Create predictor with mocked dependencies."""
        from src.model import InferenceConfig

        with patch("src.inference.predictor.InferenceSession") as mock_session, \
             patch("src.inference.predictor.BertTokenizer") as mock_tokenizer:

            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                "input_ids": np.array([[101, 102, 0, 0]]),
                "attention_mask": np.array([[1, 1, 0, 0]]),
            }
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            # Mock session
            mock_session_instance = Mock()
            mock_session_instance.get_inputs.return_value = [
                Mock(name="input_ids"),
                Mock(name="attention_mask"),
            ]
            mock_session_instance.get_outputs.return_value = [Mock(name="logits")]
            # Return logits that would give risk_label = 7
            logits = np.zeros((1, 11))
            logits[0, 7] = 10.0  # High logit for class 7
            mock_session_instance.run.return_value = [logits]
            mock_session.return_value = mock_session_instance

            from src.inference import LogRiskPredictor

            config = InferenceConfig(
                return_probabilities=True,
                return_expected_score=True,
            )
            predictor = LogRiskPredictor("fake_model.onnx", "fake_tokenizer", config)

            return predictor

    def test_predict_single(self, mock_predictor):
        result = mock_predictor.predict_single("test log message")

        assert "risk_label" in result
        assert result["risk_label"] == 7
        assert "risk_score" in result
        assert "probabilities" in result
        assert len(result["probabilities"]) == 11

    def test_predict_batch(self, mock_predictor):
        # Update mock for batch
        mock_predictor.session.run.return_value = [np.zeros((3, 11))]

        results = mock_predictor.predict(["log1", "log2", "log3"])
        assert len(results) == 3

    def test_predict_empty(self, mock_predictor):
        results = mock_predictor.predict([])
        assert results == []


class TestExportModule:
    """Test export functionality."""

    def test_export_functions_exist(self):
        from src.inference.export import export_to_onnx, export_from_checkpoint, optimize_onnx

        assert callable(export_to_onnx)
        assert callable(export_from_checkpoint)
        assert callable(optimize_onnx)


class TestAsyncPredictor:
    """Test async predictor wrapper."""

    def test_async_predictor_context_manager(self):
        from src.inference.predictor import AsyncLogRiskPredictor
        from unittest.mock import patch

        with patch.object(AsyncLogRiskPredictor, "__init__", return_value=None):
            predictor = AsyncLogRiskPredictor.__new__(AsyncLogRiskPredictor)
            predictor._running = False
            predictor._thread = None
            predictor.predictor = Mock()

            # Test context manager doesn't raise
            predictor.start = Mock()
            predictor.stop = Mock()

            with predictor:
                pass

            predictor.start.assert_called_once()
            predictor.stop.assert_called_once()
