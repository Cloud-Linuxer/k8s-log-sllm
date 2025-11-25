"""Tests for log preprocessor."""

import pytest
from src.preprocessing import LogPreprocessor


@pytest.fixture
def preprocessor():
    return LogPreprocessor(lowercase=True)


@pytest.fixture
def preprocessor_no_lower():
    return LogPreprocessor(lowercase=False)


class TestTimestampReplacement:
    def test_iso8601_timestamp(self, preprocessor):
        log = "2024-01-15T10:30:45.123Z kubelet failed to sync pod"
        result = preprocessor.preprocess(log)
        assert "<time>" in result["text"]
        assert "2024-01-15" not in result["text"]

    def test_syslog_timestamp(self, preprocessor):
        log = "Jan 15 10:30:45 server1 systemd[1]: Started nginx"
        result = preprocessor.preprocess(log)
        assert "<time>" in result["text"]
        assert "Jan 15" not in result["text"]

    def test_timestamp_with_timezone(self, preprocessor):
        log = "2024-01-15T10:30:45Z ERROR connection refused"
        result = preprocessor.preprocess(log)
        assert "<time>" in result["text"]


class TestIPReplacement:
    def test_ipv4_address(self, preprocessor):
        log = "connection from 192.168.1.100:8080 refused"
        result = preprocessor.preprocess(log)
        assert "<ip>" in result["text"]
        assert "192.168.1.100" not in result["text"]

    def test_multiple_ips(self, preprocessor):
        log = "forwarding from 10.0.0.1 to 10.0.0.2"
        result = preprocessor.preprocess(log)
        assert result["text"].count("<ip>") == 2


class TestUUIDReplacement:
    def test_standard_uuid(self, preprocessor):
        log = "pod abc12345-def6-7890-abcd-ef1234567890 created"
        result = preprocessor.preprocess(log)
        assert "<id>" in result["text"]
        assert "abc12345-def6" not in result["text"]


class TestLongHexReplacement:
    def test_container_id(self, preprocessor):
        log = "container abc123def456789 started"
        result = preprocessor.preprocess(log)
        assert "<id>" in result["text"]

    def test_short_hex_preserved(self, preprocessor):
        log = "error code 0x1234"
        result = preprocessor.preprocess(log)
        # Short hex should not be replaced
        assert "0x1234" in result["text"] or "1234" in result["text"]


class TestPIDReplacement:
    def test_pid_format(self, preprocessor):
        log = "process pid=12345 terminated"
        result = preprocessor.preprocess(log)
        assert "<pid>" in result["text"]
        assert "12345" not in result["text"]

    def test_bracket_pid(self, preprocessor):
        log = "systemd[1]: Started service"
        result = preprocessor.preprocess(log)
        assert "<pid>" in result["text"]


class TestPathReplacement:
    def test_long_path(self, preprocessor):
        log = "failed to read /var/log/kubernetes/pods/nginx.log"
        result = preprocessor.preprocess(log)
        assert "<path>" in result["text"]

    def test_short_path_preserved(self, preprocessor):
        log = "reading /etc/hosts"
        result = preprocessor.preprocess(log)
        # Short paths (< 3 depth) should be preserved
        assert "/etc/hosts" in result["text"] or "hosts" in result["text"]


class TestLogLevelExtraction:
    @pytest.mark.parametrize(
        "log,expected_level",
        [
            ("ERROR: connection refused", "ERROR"),
            ("[INFO] Starting service", "INFO"),
            ("level=WARN msg=timeout", "WARN"),
            ("CRITICAL: disk full", "CRITICAL"),
            ("DEBUG checking status", "DEBUG"),
            ("no level here", None),
        ],
    )
    def test_level_extraction(self, preprocessor, log, expected_level):
        result = preprocessor.preprocess(log)
        assert result["level_raw"] == expected_level


class TestComponentPreservation:
    def test_kubelet_preserved(self, preprocessor):
        log = "kubelet failed to create pod sandbox"
        result = preprocessor.preprocess(log)
        assert "kubelet" in result["text"]

    def test_nginx_preserved(self, preprocessor):
        log = "nginx upstream timed out"
        result = preprocessor.preprocess(log)
        assert "nginx" in result["text"]

    def test_systemd_preserved(self, preprocessor):
        log = "systemd started daily cleanup"
        result = preprocessor.preprocess(log)
        assert "systemd" in result["text"]


class TestLowercaseConversion:
    def test_lowercase_enabled(self, preprocessor):
        log = "ERROR: Failed to Connect"
        result = preprocessor.preprocess(log)
        assert result["text"] == result["text"].lower()
        # But level_raw should be uppercase
        assert result["level_raw"] == "ERROR"

    def test_lowercase_disabled(self, preprocessor_no_lower):
        log = "ERROR: Failed to Connect"
        result = preprocessor_no_lower.preprocess(log)
        assert "ERROR" in result["text"]
        assert "Failed" in result["text"]


class TestBatchProcessing:
    def test_batch_processing(self, preprocessor):
        logs = [
            "2024-01-15T10:00:00Z INFO started",
            "ERROR connection from 192.168.1.1 refused",
            "DEBUG checking pod abc12345-def6-7890-abcd-ef1234567890",
        ]
        results = preprocessor.preprocess_batch(logs)
        assert len(results) == 3
        assert all("text" in r and "level_raw" in r for r in results)


class TestEdgeCases:
    def test_empty_string(self, preprocessor):
        result = preprocessor.preprocess("")
        assert result["text"] == ""
        assert result["level_raw"] is None

    def test_whitespace_only(self, preprocessor):
        result = preprocessor.preprocess("   \t\n   ")
        assert result["text"] == ""

    def test_multiple_spaces_collapsed(self, preprocessor):
        log = "error    multiple   spaces   here"
        result = preprocessor.preprocess(log)
        assert "  " not in result["text"]

    def test_max_length_truncation(self):
        preprocessor = LogPreprocessor(max_length=50)
        log = "a" * 100
        result = preprocessor.preprocess(log)
        assert len(result["text"]) == 50
