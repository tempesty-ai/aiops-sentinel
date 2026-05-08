import os
from dotenv import load_dotenv

load_dotenv()

# Mattermost
MATTERMOST_WEBHOOK_URL = os.getenv("MATTERMOST_WEBHOOK_URL", "").strip()

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:70b").strip()

# APM thresholds
APM_CPU_THRESHOLD = float(os.getenv("APM_CPU_THRESHOLD", "80"))
APM_RESPONSE_TIME_THRESHOLD = float(os.getenv("APM_RESPONSE_TIME_THRESHOLD", "2000"))
APM_DB_CONNECTION_THRESHOLD = float(os.getenv("APM_DB_CONNECTION_THRESHOLD", "80"))
APM_CHECK_INTERVAL_SECONDS = int(os.getenv("APM_CHECK_INTERVAL_SECONDS", "10"))

# Log watch
LOG_WATCH_INTERVAL_SECONDS = int(os.getenv("LOG_WATCH_INTERVAL_SECONDS", "5"))
LOG_ERROR_KEYWORDS = [kw.strip() for kw in os.getenv("LOG_ERROR_KEYWORDS", "ERROR,FATAL,Exception,Critical").split(",") if kw.strip()]

# Monitoring targets
MONITORED_SERVERS = [
    "was_sample_01",
    "was_sample_9200",
    "was_sample_9300",
    "docker_resource_12000",
    "docker_service_12001",
    "docker_finance_13000",
    "JNode_20000",
    "JNode_21000",
    "Was_Agent_1",
    "Was_Agent_2",
]

# Mock log files
LOG_FILES = [
    "logs/agent_sample_c.log",
    "logs/agent_sample_b.log",
    "logs/agent_sample_a.log",
    "logs/data_forwarder_01.log",
    "logs/data_forwarder_02.log",
]


def validate_required_settings(require_mattermost: bool = False) -> None:
    """
    Validate required runtime configuration and raise ValueError with clear guidance.
    """
    missing: list[str] = []

    if not OLLAMA_BASE_URL:
        missing.append("OLLAMA_BASE_URL")
    if not OLLAMA_MODEL:
        missing.append("OLLAMA_MODEL")
    if require_mattermost and not MATTERMOST_WEBHOOK_URL:
        missing.append("MATTERMOST_WEBHOOK_URL")

    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Update your .env file (see .env.example)."
        )
