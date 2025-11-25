"""Weak labeling module for log risk scoring."""

from typing import Optional


class RiskLabeler:
    """Generates risk labels (0-10) based on log level and keyword patterns."""

    # Base scores by log level
    LEVEL_BASE_SCORES = {
        "TRACE": 0,
        "DEBUG": 1,
        "INFO": 2,
        "NOTICE": 3,
        "VERBOSE": 3,
        "WARN": 4,
        "WARNING": 5,
        "ERROR": 6,
        "CRITICAL": 8,
        "ALERT": 8,
        "FATAL": 9,
        "PANIC": 10,
        "EMERG": 10,
    }

    # Default base score when no level is detected
    DEFAULT_BASE_SCORE = 3

    # Keyword patterns and their bonus scores
    # All patterns should be lowercase for matching
    KEYWORD_BONUSES = {
        # K8s Critical (+2)
        "crashloopbackoff": 2,
        "imagepullbackoff": 2,
        "back-off restarting": 2,
        "backoff restarting": 2,
        "node not ready": 2,
        "notready": 2,
        "evicted": 2,
        "disk pressure": 2,
        "diskpressure": 2,
        "memory pressure": 2,
        "memorypressure": 2,
        "oomkilled": 2,
        "oom killed": 2,
        "out of memory": 3,
        # K8s Warning (+1~2)
        "failed to create pod sandbox": 2,
        "failed to create containerd task": 2,
        "context deadline exceeded": 1,
        "failed to pull image": 1,
        "failed to start container": 2,
        "container exited": 1,
        "liveness probe failed": 2,
        "readiness probe failed": 1,
        # System Critical (+3)
        "kernel panic": 3,
        "bug:": 3,
        "segfault": 3,
        "segmentation fault": 3,
        "stack trace": 3,
        "stacktrace": 3,
        "i/o error": 3,
        "io error": 3,
        "read-only file system": 3,
        "readonly file system": 3,
        "disk failure": 3,
        "hardware error": 3,
        "machine check": 3,
        "watchdog": 2,
        # Network (+1~2)
        "connection refused": 1,
        "connection reset by peer": 1,
        "connection reset": 1,
        "upstream timed out": 2,
        "no live upstreams": 2,
        "connection timed out": 1,
        "timeout": 1,
        "unreachable": 1,
        "dns resolution failed": 2,
        "name resolution failed": 2,
        # Security (+2)
        "permission denied": 2,
        "unauthorized": 2,
        "forbidden": 2,
        "authentication failed": 2,
        "auth failed": 2,
        "access denied": 2,
        "certificate expired": 2,
        "certificate error": 2,
        "ssl error": 2,
        "tls error": 2,
        # Database (+2)
        "deadlock": 2,
        "lock timeout": 2,
        "connection pool exhausted": 2,
        "too many connections": 2,
    }

    # Maximum bonus to apply (prevents over-scoring)
    MAX_BONUS = 5

    def __init__(self, max_bonus: int = 5):
        """
        Initialize labeler.

        Args:
            max_bonus: Maximum keyword bonus to apply
        """
        self.max_bonus = max_bonus

    def _get_base_score(self, level_raw: Optional[str]) -> int:
        """Get base score from log level."""
        if level_raw is None:
            return self.DEFAULT_BASE_SCORE
        return self.LEVEL_BASE_SCORES.get(level_raw.upper(), self.DEFAULT_BASE_SCORE)

    def _calculate_keyword_bonus(self, text: str) -> int:
        """Calculate bonus score based on keyword matches."""
        text_lower = text.lower()
        total_bonus = 0

        for keyword, bonus in self.KEYWORD_BONUSES.items():
            if keyword in text_lower:
                total_bonus += bonus

        return min(total_bonus, self.max_bonus)

    def label(self, text: str, level_raw: Optional[str] = None) -> int:
        """
        Generate risk label for a log entry.

        Args:
            text: Preprocessed log text
            level_raw: Extracted log level (optional)

        Returns:
            Risk label from 0-10
        """
        base_score = self._get_base_score(level_raw)
        bonus = self._calculate_keyword_bonus(text)
        final_score = base_score + bonus

        # Clamp to 0-10 range
        return max(0, min(10, final_score))

    def label_with_details(
        self, text: str, level_raw: Optional[str] = None
    ) -> dict:
        """
        Generate risk label with detailed breakdown.

        Args:
            text: Preprocessed log text
            level_raw: Extracted log level (optional)

        Returns:
            Dict with label, base_score, bonus, and matched_keywords
        """
        base_score = self._get_base_score(level_raw)

        # Find matched keywords
        text_lower = text.lower()
        matched_keywords = []
        total_bonus = 0

        for keyword, bonus in self.KEYWORD_BONUSES.items():
            if keyword in text_lower:
                matched_keywords.append((keyword, bonus))
                total_bonus += bonus

        capped_bonus = min(total_bonus, self.max_bonus)
        final_score = max(0, min(10, base_score + capped_bonus))

        return {
            "label": final_score,
            "base_score": base_score,
            "raw_bonus": total_bonus,
            "capped_bonus": capped_bonus,
            "matched_keywords": matched_keywords,
            "level_raw": level_raw,
        }

    def label_batch(self, data: list[dict]) -> list[int]:
        """
        Label multiple preprocessed log entries.

        Args:
            data: List of dicts with 'text' and optionally 'level_raw'

        Returns:
            List of risk labels (0-10)
        """
        return [
            self.label(d["text"], d.get("level_raw"))
            for d in data
        ]
