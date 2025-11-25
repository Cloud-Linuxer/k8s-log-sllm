"""Log preprocessing module for token replacement and normalization."""

import re
from typing import Optional


class LogPreprocessor:
    """Preprocesses log lines by replacing variable tokens with placeholders."""

    # Patterns ordered by specificity (more specific patterns first)
    PATTERNS = {
        # ISO8601 timestamp: 2024-01-15T10:30:45.123Z or 2024/01/15 10:30:45
        "timestamp": re.compile(
            r"\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?"
        ),
        # Syslog timestamp: Jan 15 10:30:45
        "syslog_timestamp": re.compile(
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
        ),
        # UUID: 8-4-4-4-12 hex format
        "uuid": re.compile(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        ),
        # IPv4 address
        "ip": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        # Long hex strings (container IDs, etc.) - 12+ chars
        "long_hex": re.compile(r"\b[0-9a-fA-F]{12,}\b"),
        # PID/TID patterns
        "pid": re.compile(r"\b(?:pid|PID|tid|TID)[=: ]?\d+\b|\[\d+\]"),
        # File paths with 3+ depth
        "path": re.compile(r"(?:/[a-zA-Z0-9._-]+){3,}"),
    }

    # Log level detection pattern
    LOG_LEVEL_PATTERN = re.compile(
        r"\b(TRACE|DEBUG|INFO|NOTICE|VERBOSE|WARN|WARNING|ERROR|CRITICAL|ALERT|FATAL|PANIC|EMERG)\b",
        re.IGNORECASE,
    )

    # Token replacements
    REPLACEMENTS = {
        "timestamp": "<TIME>",
        "syslog_timestamp": "<TIME>",
        "uuid": "<ID>",
        "ip": "<IP>",
        "long_hex": "<ID>",
        "pid": "<PID>",
        "path": "<PATH>",
    }

    # Pattern application order (more specific first to avoid conflicts)
    PATTERN_ORDER = [
        "timestamp",
        "syslog_timestamp",
        "uuid",
        "ip",
        "long_hex",
        "pid",
        "path",
    ]

    def __init__(self, lowercase: bool = True, max_length: Optional[int] = None):
        """
        Initialize preprocessor.

        Args:
            lowercase: Whether to convert text to lowercase (for uncased BERT)
            max_length: Maximum character length (None for no limit)
        """
        self.lowercase = lowercase
        self.max_length = max_length

    def extract_log_level(self, log_line: str) -> Optional[str]:
        """Extract log level from log line."""
        match = self.LOG_LEVEL_PATTERN.search(log_line)
        if match:
            return match.group(1).upper()
        return None

    def preprocess(self, log_line: str) -> dict:
        """
        Preprocess a single log line.

        Args:
            log_line: Raw log string

        Returns:
            dict with 'text' (preprocessed string) and 'level_raw' (extracted log level)
        """
        # Extract log level before any modifications
        level_raw = self.extract_log_level(log_line)

        text = log_line.strip()

        # Apply token replacements in order
        for pattern_name in self.PATTERN_ORDER:
            pattern = self.PATTERNS[pattern_name]
            replacement = self.REPLACEMENTS[pattern_name]
            text = pattern.sub(replacement, text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Apply lowercase if configured (for uncased BERT)
        if self.lowercase:
            text = text.lower()

        # Apply max length truncation
        if self.max_length and len(text) > self.max_length:
            text = text[: self.max_length]

        return {"text": text, "level_raw": level_raw}

    def preprocess_batch(self, logs: list[str]) -> list[dict]:
        """
        Preprocess multiple log lines.

        Args:
            logs: List of raw log strings

        Returns:
            List of dicts with 'text' and 'level_raw'
        """
        return [self.preprocess(log) for log in logs]
