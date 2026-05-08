from apm.ai_analyzer import APMAIAnalyzer
from logwatch.ai_classifier import LogAIClassifier


def test_apm_parser_handles_missing_sections_with_fallbacks():
    analyzer = APMAIAnalyzer()
    raw = "Fault Type: CPU saturation\nSeverity: critical"
    parsed = analyzer._parse_response(raw)
    assert parsed.fault_type
    assert parsed.root_cause
    assert parsed.immediate_action
    assert parsed.prevention
    assert parsed.severity


def test_apm_parser_handles_empty_response():
    analyzer = APMAIAnalyzer()
    parsed = analyzer._parse_response("")
    assert parsed.fault_type == "Unknown"
    assert "No root-cause section" in parsed.root_cause


def test_log_parser_handles_partial_response():
    classifier = LogAIClassifier()
    raw = "Error Type: Network error\nRecommended Action: Retry connection and verify DNS."
    parsed = classifier._parse_response(raw)
    assert parsed.error_type
    assert parsed.recommended_action
    assert parsed.severity
    assert parsed.recurrence


def test_log_parser_heuristic_fallback_detects_memory_signal():
    classifier = LogAIClassifier()
    raw = "java.lang.OutOfMemoryError: Java heap space"
    parsed = classifier._parse_response(raw)
    assert "memory" in parsed.error_type.lower()
