"""Tests for risk labeler."""

import pytest
from src.labeling import RiskLabeler


@pytest.fixture
def labeler():
    return RiskLabeler()


class TestBaseScores:
    @pytest.mark.parametrize(
        "level,expected_base",
        [
            ("TRACE", 0),
            ("DEBUG", 1),
            ("INFO", 2),
            ("NOTICE", 3),
            ("WARN", 4),
            ("WARNING", 5),
            ("ERROR", 6),
            ("CRITICAL", 8),
            ("ALERT", 8),
            ("FATAL", 9),
            ("PANIC", 10),
            ("EMERG", 10),
        ],
    )
    def test_level_base_scores(self, labeler, level, expected_base):
        # Simple text without keywords
        result = labeler.label("simple log message", level)
        assert result == expected_base

    def test_no_level_default_score(self, labeler):
        result = labeler.label("simple log message", None)
        assert result == 3  # Default base score

    def test_case_insensitive_level(self, labeler):
        result1 = labeler.label("message", "error")
        result2 = labeler.label("message", "ERROR")
        result3 = labeler.label("message", "Error")
        assert result1 == result2 == result3 == 6


class TestKeywordBonuses:
    def test_crashloopbackoff_bonus(self, labeler):
        # ERROR (6) + crashloopbackoff (+2) = 8
        result = labeler.label("pod crashloopbackoff detected", "ERROR")
        assert result == 8

    def test_kernel_panic_bonus(self, labeler):
        # CRITICAL (8) + kernel panic (+3) = 10 (capped at 10)
        result = labeler.label("kernel panic occurred", "CRITICAL")
        assert result == 10

    def test_connection_refused_bonus(self, labeler):
        # WARN (4) + connection refused (+1) = 5
        result = labeler.label("connection refused", "WARN")
        assert result == 5

    def test_multiple_keywords(self, labeler):
        # ERROR (6) + oomkilled (+2) + crashloopbackoff (+2) = 10
        result = labeler.label("pod oomkilled then crashloopbackoff", "ERROR")
        assert result == 10

    def test_security_keywords(self, labeler):
        # WARN (4) + unauthorized (+2) = 6
        result = labeler.label("unauthorized access attempt", "WARN")
        assert result == 6

    def test_keyword_case_insensitive(self, labeler):
        result1 = labeler.label("CrashLoopBackOff", "INFO")
        result2 = labeler.label("crashloopbackoff", "INFO")
        assert result1 == result2


class TestBonusCapping:
    def test_bonus_cap_default(self, labeler):
        # Multiple keywords that would exceed cap
        # INFO (2) + kernel panic (3) + segfault (3) + stack trace (3) = 11
        # But capped at 5 bonus, so 2 + 5 = 7
        text = "kernel panic segfault stack trace"
        result = labeler.label(text, "INFO")
        assert result == 7  # 2 (INFO) + 5 (max bonus)

    def test_custom_max_bonus(self):
        labeler = RiskLabeler(max_bonus=3)
        text = "kernel panic segfault stack trace"
        result = labeler.label(text, "INFO")
        assert result == 5  # 2 (INFO) + 3 (custom max bonus)


class TestLabelWithDetails:
    def test_details_structure(self, labeler):
        result = labeler.label_with_details("crashloopbackoff error", "ERROR")
        assert "label" in result
        assert "base_score" in result
        assert "raw_bonus" in result
        assert "capped_bonus" in result
        assert "matched_keywords" in result
        assert "level_raw" in result

    def test_matched_keywords_list(self, labeler):
        result = labeler.label_with_details("timeout connection refused", "WARN")
        keywords = [k for k, _ in result["matched_keywords"]]
        assert "timeout" in keywords
        assert "connection refused" in keywords

    def test_raw_vs_capped_bonus(self, labeler):
        text = "kernel panic segfault out of memory"
        result = labeler.label_with_details(text, "INFO")
        assert result["raw_bonus"] > result["capped_bonus"]
        assert result["capped_bonus"] == 5


class TestBatchLabeling:
    def test_batch_processing(self, labeler):
        data = [
            {"text": "info message", "level_raw": "INFO"},
            {"text": "crashloopbackoff", "level_raw": "ERROR"},
            {"text": "kernel panic", "level_raw": "CRITICAL"},
        ]
        results = labeler.label_batch(data)
        assert len(results) == 3
        assert results[0] == 2  # INFO
        assert results[1] == 8  # ERROR + crashloopbackoff
        assert results[2] == 10  # CRITICAL + kernel panic (capped)

    def test_batch_without_level(self, labeler):
        data = [
            {"text": "simple message"},
            {"text": "connection refused"},
        ]
        results = labeler.label_batch(data)
        assert len(results) == 2
        assert results[0] == 3  # Default
        assert results[1] == 4  # Default + connection refused


class TestClampingBehavior:
    def test_minimum_clamp(self, labeler):
        # Shouldn't go below 0, but our base scores start at 0
        result = labeler.label("simple debug", "TRACE")
        assert result >= 0

    def test_maximum_clamp(self, labeler):
        # Even with high base + high bonus, should cap at 10
        result = labeler.label("kernel panic segfault oomkilled", "PANIC")
        assert result == 10


class TestRealWorldExamples:
    @pytest.mark.parametrize(
        "log,level,expected_min,expected_max",
        [
            # K8s examples
            ("kubelet failed to create pod sandbox context deadline exceeded", "ERROR", 7, 9),
            ("nginx upstream timed out while reading response header", "ERROR", 8, 10),
            ("systemd started daily apt download activities", "INFO", 2, 3),
            ("kube-apiserver node not ready", "WARNING", 7, 9),
            # System examples
            ("kernel: BUG: unable to handle page fault", "ERROR", 9, 10),
            ("sshd: authentication failed for user root", "WARNING", 7, 9),
            # Normal operations
            ("nginx: GET /health 200", "INFO", 2, 3),
            ("docker: container started successfully", "INFO", 2, 3),
        ],
    )
    def test_real_world_range(self, labeler, log, level, expected_min, expected_max):
        result = labeler.label(log, level)
        assert expected_min <= result <= expected_max, f"Got {result} for '{log}'"
