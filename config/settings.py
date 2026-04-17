import os
from dotenv import load_dotenv

load_dotenv()

# Mattermost
MATTERMOST_WEBHOOK_URL = os.getenv("MATTERMOST_WEBHOOK_URL", "")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:70b")

# APM 임계값
APM_CPU_THRESHOLD = float(os.getenv("APM_CPU_THRESHOLD", "80"))
APM_RESPONSE_TIME_THRESHOLD = float(os.getenv("APM_RESPONSE_TIME_THRESHOLD", "2000"))
APM_DB_CONNECTION_THRESHOLD = float(os.getenv("APM_DB_CONNECTION_THRESHOLD", "80"))
APM_CHECK_INTERVAL_SECONDS = int(os.getenv("APM_CHECK_INTERVAL_SECONDS", "10"))

# 로그 감시
LOG_WATCH_INTERVAL_SECONDS = int(os.getenv("LOG_WATCH_INTERVAL_SECONDS", "5"))
LOG_ERROR_KEYWORDS = os.getenv("LOG_ERROR_KEYWORDS", "ERROR,FATAL,Exception,Critical").split(",")

# 서버 목록 (InterMax에서 보이던 실제 서버명 기반)
MONITORED_SERVERS = [
    "juntion_9100",
    "juntion_9200",
    "juntion_9300",
    "docker_리스크_12000",
    "docker_재무_12001",
    "docker_금융계_13000",
    "JNode_20000",
    "JNode_21000",
    "Was_Agent_1",
    "Was_Agent_2",
]

# 수집 모듈 로그 파일 목록 (mock)
LOG_FILES = [
    "logs/collector_agent_01.log",
    "logs/collector_agent_02.log",
    "logs/collector_agent_03.log",
    "logs/data_forwarder_01.log",
    "logs/data_forwarder_02.log",
]
