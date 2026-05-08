from apm.anomaly_detector import AnomalyDetector
from apm.mock_generator import APMSnapshot
from config.settings import (
    APM_CPU_THRESHOLD,
    APM_DB_CONNECTION_THRESHOLD,
    APM_RESPONSE_TIME_THRESHOLD,
)


def _snapshot(**overrides):
    data = {
        "timestamp": "2026-04-23 10:00:00",
        "server": "test_server",
        "cpu_usage": 10.0,
        "memory_usage": 40.0,
        "active_transactions": 5,
        "response_time_ms": 100.0,
        "db_connections": 5,
        "db_connection_max": 50,
        "requests_per_sec": 10.0,
        "error_rate": 0.1,
        "heap_used_mb": 200.0,
        "heap_max_mb": 1024.0,
        "alert_level": "normal",
    }
    data.update(overrides)
    return APMSnapshot(**data)


def test_boundary_values_are_detected_as_anomaly():
    db_connections = int((APM_DB_CONNECTION_THRESHOLD / 100.0) * 50)
    snap = _snapshot(
        cpu_usage=APM_CPU_THRESHOLD,
        response_time_ms=APM_RESPONSE_TIME_THRESHOLD,
        db_connections=db_connections,
    )
    result = AnomalyDetector().analyze(snap)
    assert result.is_anomaly is True
    assert len(result.triggered_rules) >= 3


def test_below_threshold_values_are_not_anomaly():
    db_connections = max(1, int((APM_DB_CONNECTION_THRESHOLD / 100.0) * 50) - 1)
    snap = _snapshot(
        cpu_usage=APM_CPU_THRESHOLD - 0.1,
        response_time_ms=APM_RESPONSE_TIME_THRESHOLD - 0.1,
        db_connections=db_connections,
        memory_usage=60.0,
        heap_used_mb=300.0,
        error_rate=1.0,
    )
    result = AnomalyDetector().analyze(snap)
    assert result.is_anomaly is False
    assert result.triggered_rules == []
