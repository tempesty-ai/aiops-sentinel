"""
Mock APM 데이터 생성기
InterMax에서 실제로 수집되는 WAS 모니터링 데이터 구조를 모방
"""
import random
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from config.settings import MONITORED_SERVERS


@dataclass
class APMSnapshot:
    timestamp: str
    server: str
    cpu_usage: float           # CPU 사용률 (%)
    memory_usage: float        # MEM 사용률 (%)
    active_transactions: int   # 액티브 트랜잭션 수
    response_time_ms: float    # 평균 응답시간 (ms)
    db_connections: int        # DB 커넥션 수
    db_connection_max: int     # DB 커넥션 최대값
    requests_per_sec: float    # 초당 요청 수
    error_rate: float          # 에러율 (%)
    heap_used_mb: float        # 힙 사용량 (MB)
    heap_max_mb: float         # 힙 최대값 (MB)
    alert_level: str = "정상"  # 정상 / 경고 / 심각


class MockAPMGenerator:
    """
    InterMax WAS 모니터링과 동일한 구조의 Mock 데이터 생성
    장애 시나리오를 주기적으로 주입하여 AI 분석 테스트 가능
    """

    # 장애 시나리오 정의
    FAULT_SCENARIOS = [
        "cpu_spike",           # CPU 폭증
        "memory_leak",         # 메모리 누수
        "db_connection_pool",  # DB 커넥션 풀 고갈
        "slow_response",       # 응답 지연
        "high_error_rate",     # 에러율 급증
        "normal",              # 정상 상태
    ]

    def __init__(self):
        # 서버별 현재 상태 초기화
        self._server_state = {
            server: {"scenario": "normal", "scenario_duration": 0}
            for server in MONITORED_SERVERS
        }

    def _maybe_inject_fault(self, server: str) -> str:
        state = self._server_state[server]

        # 장애 지속 중이면 유지
        if state["scenario_duration"] > 0:
            state["scenario_duration"] -= 1
            return state["scenario"]

        # 10% 확률로 새 장애 주입
        if random.random() < 0.10:
            scenario = random.choice(self.FAULT_SCENARIOS[:-1])  # normal 제외
            state["scenario"] = scenario
            state["scenario_duration"] = random.randint(3, 8)   # 3~8회 지속
            return scenario

        state["scenario"] = "normal"
        return "normal"

    def _generate_snapshot(self, server: str) -> APMSnapshot:
        scenario = self._maybe_inject_fault(server)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 기본 정상 수치
        cpu = random.uniform(10, 40)
        memory = random.uniform(30, 60)
        active_tx = random.randint(1, 10)
        response_ms = random.uniform(100, 500)
        db_conn = random.randint(2, 15)
        db_max = 50
        rps = random.uniform(5, 20)
        error_rate = random.uniform(0, 0.5)
        heap_used = random.uniform(200, 600)
        heap_max = 1024

        # 장애 시나리오별 수치 조작
        if scenario == "cpu_spike":
            cpu = random.uniform(85, 98)
            response_ms = random.uniform(1500, 3000)
            active_tx = random.randint(20, 40)

        elif scenario == "memory_leak":
            memory = random.uniform(88, 97)
            heap_used = random.uniform(900, 1020)
            response_ms = random.uniform(800, 2000)

        elif scenario == "db_connection_pool":
            db_conn = random.randint(45, 50)
            response_ms = random.uniform(3000, 8000)
            active_tx = random.randint(25, 50)
            error_rate = random.uniform(5, 20)

        elif scenario == "slow_response":
            response_ms = random.uniform(4000, 10000)
            active_tx = random.randint(15, 35)
            rps = random.uniform(1, 5)

        elif scenario == "high_error_rate":
            error_rate = random.uniform(15, 40)
            response_ms = random.uniform(1000, 3000)
            active_tx = random.randint(10, 30)

        # 알람 레벨 결정
        alert_level = "정상"
        if cpu > 90 or memory > 90 or db_conn >= db_max - 2 or response_ms > 5000 or error_rate > 20:
            alert_level = "심각"
        elif cpu > 80 or memory > 80 or db_conn >= db_max * 0.8 or response_ms > 2000 or error_rate > 5:
            alert_level = "경고"

        return APMSnapshot(
            timestamp=now,
            server=server,
            cpu_usage=round(cpu, 1),
            memory_usage=round(memory, 1),
            active_transactions=active_tx,
            response_time_ms=round(response_ms, 0),
            db_connections=db_conn,
            db_connection_max=db_max,
            requests_per_sec=round(rps, 1),
            error_rate=round(error_rate, 2),
            heap_used_mb=round(heap_used, 1),
            heap_max_mb=heap_max,
            alert_level=alert_level,
        )

    def get_all_snapshots(self) -> list[APMSnapshot]:
        return [self._generate_snapshot(server) for server in MONITORED_SERVERS]

    def get_snapshot(self, server: str) -> APMSnapshot:
        return self._generate_snapshot(server)
