"""CPU inference predictor using ONNX Runtime."""

import time
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Optional, Callable, Iterator

import numpy as np
from transformers import BertTokenizer

from src.model import InferenceConfig


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class LogRiskPredictor:
    """
    CPU inference predictor for log risk classification.

    Uses ONNX Runtime for efficient CPU inference.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to ONNX model
            tokenizer_path: Path to tokenizer
            config: Inference configuration
        """
        # Import here to make it optional dependency for non-inference use
        from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

        self.config = config or InferenceConfig()

        # Setup ONNX Runtime session
        options = SessionOptions()

        # Set optimization level
        opt_levels = {
            "all": GraphOptimizationLevel.ORT_ENABLE_ALL,
            "basic": GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "disabled": GraphOptimizationLevel.ORT_DISABLE_ALL,
        }
        options.graph_optimization_level = opt_levels.get(
            self.config.graph_optimization_level, GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Set number of threads
        options.intra_op_num_threads = self.config.num_threads
        options.inter_op_num_threads = self.config.num_threads

        # Create session
        self.session = InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Predict risk scores for log texts.

        Args:
            texts: List of preprocessed log strings

        Returns:
            List of prediction dicts with risk_label, risk_score, probabilities
        """
        if not texts:
            return []

        # Tokenize - must pad to max_length for ONNX fixed shape
        encodings = self.tokenizer(
            texts,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # Run inference
        outputs = self.session.run(
            self.output_names,
            {
                "input_ids": encodings["input_ids"].astype(np.int64),
                "attention_mask": encodings["attention_mask"].astype(np.int64),
            },
        )[0]

        # Convert logits to probabilities
        probs = softmax(outputs, axis=-1)

        # Build results
        results = []
        for i, prob in enumerate(probs):
            risk_label = int(np.argmax(prob))

            result = {"risk_label": risk_label}

            if self.config.return_expected_score:
                # Expected score: E[score] = sum(k * p[k])
                risk_score = float(np.sum(np.arange(11) * prob))
                result["risk_score"] = round(risk_score, 3)

            if self.config.return_probabilities:
                result["probabilities"] = prob.tolist()

            results.append(result)

        return results

    def predict_single(self, text: str) -> dict:
        """
        Predict risk score for a single log.

        Args:
            text: Preprocessed log string

        Returns:
            Prediction dict
        """
        return self.predict([text])[0]

    def predict_batch(self, texts: list[str], batch_size: Optional[int] = None) -> list[dict]:
        """
        Predict with batching for large inputs.

        Args:
            texts: List of preprocessed log strings
            batch_size: Batch size (defaults to config)

        Returns:
            List of prediction dicts
        """
        batch_size = batch_size or self.config.batch_size
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self.predict(batch))

        return results

    def predict_streaming(
        self,
        log_queue: Queue,
        callback: Callable[[list[dict]], None],
        stop_event: Optional[Callable[[], bool]] = None,
    ):
        """
        Stream predictions from a queue of logs.

        Collects logs up to batch_size, then scores them periodically.

        Args:
            log_queue: Queue of log strings
            callback: Function to call with prediction results
            stop_event: Optional function that returns True to stop
        """
        buffer = []
        batch_size = self.config.streaming_batch_size
        interval = self.config.streaming_interval

        while True:
            # Check stop condition
            if stop_event and stop_event():
                break

            # Collect logs from queue
            start_time = time.time()
            while len(buffer) < batch_size:
                try:
                    timeout = max(0.01, interval - (time.time() - start_time))
                    log = log_queue.get(timeout=timeout)
                    buffer.append(log)
                except Empty:
                    break

            # Process buffer if not empty
            if buffer:
                results = self.predict(buffer)
                callback(results)
                buffer = []

            # Wait for remaining interval
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def predict_streaming_iter(
        self,
        log_iterator: Iterator[str],
        batch_size: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Stream predictions from an iterator of logs.

        Args:
            log_iterator: Iterator of log strings
            batch_size: Batch size

        Yields:
            Prediction dicts
        """
        batch_size = batch_size or self.config.streaming_batch_size
        buffer = []

        for log in log_iterator:
            buffer.append(log)
            if len(buffer) >= batch_size:
                yield from self.predict(buffer)
                buffer = []

        # Process remaining
        if buffer:
            yield from self.predict(buffer)


class AsyncLogRiskPredictor:
    """
    Async wrapper for log risk prediction.

    Runs prediction in a background thread.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config: Optional[InferenceConfig] = None,
    ):
        """Initialize async predictor."""
        self.predictor = LogRiskPredictor(model_path, tokenizer_path, config)
        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        self._running = False
        self._thread: Optional[Thread] = None

    def start(self):
        """Start background prediction thread."""
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background prediction thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _process_loop(self):
        """Background processing loop."""
        while self._running:
            self.predictor.predict_streaming(
                self.input_queue,
                lambda results: [self.output_queue.put(r) for r in results],
                stop_event=lambda: not self._running,
            )

    def submit(self, text: str):
        """Submit a log for prediction."""
        self.input_queue.put(text)

    def get_result(self, timeout: float = 1.0) -> Optional[dict]:
        """Get a prediction result."""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
