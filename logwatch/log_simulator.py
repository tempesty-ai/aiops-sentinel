"""
리눅스 수집 모듈 로그 시뮬레이터
실제 수집 서버의 로그 패턴을 모방하여 랜덤으로 ERROR 상황을 생성
"""
import random
import time
import os
import threading
from datetime import datetime
from config.settings import LOG_FILES


# 실제 수집 모듈에서 발생할 수 있는 에러 패턴
ERROR_TEMPLATES = [
    "ERROR [DataCollector] Connection refused to target host: {host}:{port}",
    "ERROR [DataCollector] Socket timeout after {timeout}ms - host: {host}",
    "ERROR [DBConnector] Failed to execute query: ORA-{code}: {msg}",
    "ERROR [MetricForwarder] Send failed - queue overflow ({queue_size} items pending)",
    "ERROR [AgentManager] Agent heartbeat lost: agent_id={agent_id}",
    "ERROR [ConfigLoader] Failed to load config file: {file} - Permission denied",
    "FATAL [DataCollector] Unhandled exception in collection thread",
    "ERROR [HTTPClient] Request failed with status 503: Service Unavailable",
    "ERROR [JVMMonitor] JMX connection failed: {host}:{jmx_port}",
    "ERROR [LogParser] Malformed log entry - skipping {count} lines",
    "Exception in thread 'collector-worker-{n}' java.lang.OutOfMemoryError: Java heap space",
    "ERROR [SSLHandler] SSL handshake failed: certificate expired",
    "Critical [WatchDog] Process {pid} not responding - triggering restart",
    "ERROR [DataBuffer] Buffer flush failed - disk write error: {path}",
]

WARN_TEMPLATES = [
    "WARN [DataCollector] High latency detected: {latency}ms (threshold: 1000ms)",
    "WARN [DBConnector] Connection pool at {pct}% capacity",
    "WARN [MetricForwarder] Retry {n}/3 - target unavailable",
    "WARN [AgentManager] Agent {agent_id} slow response: {ms}ms",
]

INFO_TEMPLATES = [
    "INFO [DataCollector] Collection cycle completed - {n} metrics gathered",
    "INFO [AgentManager] Agent {agent_id} connected successfully",
    "INFO [MetricForwarder] Batch sent: {n} records in {ms}ms",
    "INFO [ConfigLoader] Configuration reloaded successfully",
    "INFO [HealthCheck] All systems operational",
]


def _random_error_log() -> str:
    template = random.choice(ERROR_TEMPLATES)
    return template.format(
        host=f"10.10.{random.randint(1,50)}.{random.randint(1,200)}",
        port=random.choice([8080, 8443, 9090, 3306, 1521, 5432]),
        timeout=random.randint(3000, 30000),
        code=random.randint(1000, 9999),
        msg=random.choice(["table or view does not exist", "insufficient privileges", "deadlock detected"]),
        queue_size=random.randint(500, 5000),
        agent_id=f"agent_{random.randint(1,20):02d}",
        file=random.choice(["/etc/collector/config.yml", "/opt/agent/settings.conf"]),
        host2=f"10.10.{random.randint(1,50)}.{random.randint(1,200)}",
        jmx_port=random.randint(9000, 9999),
        count=random.randint(1, 100),
        n=random.randint(1, 8),
        pid=random.randint(10000, 99999),
        path=f"/data/buffer/{random.randint(1,10)}.buf",
        latency=random.randint(1000, 10000),
        pct=random.randint(80, 99),
        ms=random.randint(100, 5000),
    )


def _random_info_log() -> str:
    template = random.choice(INFO_TEMPLATES + WARN_TEMPLATES)
    return template.format(
        n=random.randint(100, 10000),
        agent_id=f"agent_{random.randint(1,20):02d}",
        ms=random.randint(10, 500),
        pct=random.randint(30, 79),
        latency=random.randint(100, 999),
    )


def _write_log_line(filepath: str, message: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    thread_id = random.randint(100, 999)
    line = f"[{timestamp}] [thread-{thread_id}] {message}\n"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line)


class LogSimulator:
    """
    여러 수집 모듈 로그 파일에 주기적으로 로그를 기록
    10% 확률로 ERROR 상황 발생
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self._running = False
        self._thread: threading.Thread | None = None

    def _simulate_loop(self):
        while self._running:
            for log_file in LOG_FILES:
                filepath = os.path.join(self.base_dir, log_file)

                # 정상 로그 1~3줄
                for _ in range(random.randint(1, 3)):
                    _write_log_line(filepath, _random_info_log())

                # 10% 확률로 에러 발생
                if random.random() < 0.10:
                    _write_log_line(filepath, _random_error_log())

            time.sleep(random.uniform(1.5, 3.0))

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._simulate_loop, daemon=True)
        self._thread.start()
        print("[LogSimulator] 시작됨 - 수집 모듈 로그 생성 중...")

    def stop(self):
        self._running = False
        print("[LogSimulator] 중지됨")
