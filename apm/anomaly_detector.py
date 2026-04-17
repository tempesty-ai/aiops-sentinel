"""
APM 이상 감지 모듈
임계값 기반으로 장애 여부를 판단하고 분석용 컨텍스트를 구성
"""
from dataclasses import dataclass
from typing import Optional
from apm.mock_generator import APMSnapshot
from config.settings import (
    APM_CPU_THRESHOLD,
    APM_RESPONSE_TIME_THRESHOLD,
    APM_DB_CONNECTION_THRESHOLD,
)


@dataclass
class AnomalyResult:
    is_anomaly: bool
    severity: str                  # "정상" / "경고" / "심각"
    snapshot: APMSnapshot
    triggered_rules: list[str]     # 어떤 임계값이 걸렸는지
    context_for_ai: str            # AI 분석에 넘길 텍스트 컨텍스트


class AnomalyDetector:
    """
    InterMax와 동일한 임계값 기반 이상 감지 로직
    감지된 이상 상황을 AI 분석용 텍스트로 변환
    """

    def analyze(self, snapshot: APMSnapshot) -> AnomalyResult:
        triggered_rules = []

        # CPU 임계값 체크
        if snapshot.cpu_usage >= APM_CPU_THRESHOLD:
            triggered_rules.append(
                f"CPU 사용률 {snapshot.cpu_usage}% (임계값: {APM_CPU_THRESHOLD}%)"
            )

        # 응답시간 임계값 체크
        if snapshot.response_time_ms >= APM_RESPONSE_TIME_THRESHOLD:
            triggered_rules.append(
                f"응답시간 {snapshot.response_time_ms:.0f}ms (임계값: {APM_RESPONSE_TIME_THRESHOLD}ms)"
            )

        # DB 커넥션 풀 체크 (사용률 %)
        db_usage_pct = (snapshot.db_connections / snapshot.db_connection_max) * 100
        if db_usage_pct >= APM_DB_CONNECTION_THRESHOLD:
            triggered_rules.append(
                f"DB 커넥션 {snapshot.db_connections}/{snapshot.db_connection_max} "
                f"({db_usage_pct:.0f}%, 임계값: {APM_DB_CONNECTION_THRESHOLD}%)"
            )

        # 메모리 체크 (90% 이상)
        if snapshot.memory_usage >= 90:
            triggered_rules.append(
                f"메모리 사용률 {snapshot.memory_usage}% (임계값: 90%)"
            )

        # 힙 메모리 체크 (90% 이상)
        heap_usage_pct = (snapshot.heap_used_mb / snapshot.heap_max_mb) * 100
        if heap_usage_pct >= 90:
            triggered_rules.append(
                f"힙 메모리 {snapshot.heap_used_mb}MB/{snapshot.heap_max_mb}MB "
                f"({heap_usage_pct:.0f}%, 임계값: 90%)"
            )

        # 에러율 체크 (5% 이상)
        if snapshot.error_rate >= 5:
            triggered_rules.append(
                f"에러율 {snapshot.error_rate}% (임계값: 5%)"
            )

        is_anomaly = len(triggered_rules) > 0

        # 심각도 결정 (걸린 룰 수 + 알람 레벨 기반)
        if snapshot.alert_level == "심각" or len(triggered_rules) >= 3:
            severity = "심각"
        elif snapshot.alert_level == "경고" or len(triggered_rules) >= 1:
            severity = "경고"
        else:
            severity = "정상"

        context = self._build_ai_context(snapshot, triggered_rules)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            severity=severity,
            snapshot=snapshot,
            triggered_rules=triggered_rules,
            context_for_ai=context,
        )

    def _build_ai_context(self, s: APMSnapshot, rules: list[str]) -> str:
        db_usage_pct = (s.db_connections / s.db_connection_max) * 100
        heap_usage_pct = (s.heap_used_mb / s.heap_max_mb) * 100

        return f"""
[APM 모니터링 이상 감지 - {s.timestamp}]

서버: {s.server}
감지된 이상 항목:
{chr(10).join(f"  - {r}" for r in rules) if rules else "  - 없음 (정상)"}

전체 지표:
  - CPU 사용률: {s.cpu_usage}%
  - 메모리 사용률: {s.memory_usage}%
  - 힙 메모리: {s.heap_used_mb}MB / {s.heap_max_mb}MB ({heap_usage_pct:.0f}%)
  - 액티브 트랜잭션: {s.active_transactions}건
  - 평균 응답시간: {s.response_time_ms:.0f}ms
  - DB 커넥션: {s.db_connections}/{s.db_connection_max} ({db_usage_pct:.0f}%)
  - 초당 요청: {s.requests_per_sec}건/초
  - 에러율: {s.error_rate}%
  - 현재 알람 레벨: {s.alert_level}
""".strip()
