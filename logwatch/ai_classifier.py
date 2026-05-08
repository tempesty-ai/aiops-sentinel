"""
Log error classifier with retry/fallback behavior.
"""
from dataclasses import dataclass
import time

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


SYSTEM_PROMPT = """You classify operational log errors.
Return short output with these exact sections:
1. Error Type
2. Severity
3. Recurrence
4. Recommended Action"""


@dataclass
class LogAIClassification:
    error_type: str
    severity: str
    recurrence: str
    recommended_action: str
    raw_response: str


class LogAIClassifier:

    def __init__(self, max_retries: int = 2, retry_backoff_seconds: float = 0.5):
        self._llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def classify(self, context: str) -> LogAIClassification:
        raw = self._invoke_with_retry(context)
        return self._parse_response(raw)

    def _invoke_with_retry(self, context: str) -> str:
        last_error = ""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Classify this log error:\n\n{context}"),
        ]
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self._llm.invoke(messages)
                content = str(getattr(response, "content", "")).strip()
                if not content:
                    raise ValueError("LLM returned empty content")
                return content
            except Exception as exc:  # pragma: no cover - runtime integration
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt <= self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
        return self._fallback_response(last_error)

    @staticmethod
    def _clean(text: str) -> str:
        import re

        text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
        return text.strip()

    @staticmethod
    def _fallback_response(reason: str) -> str:
        return (
            "Error Type: Unknown (LLM unavailable)\n"
            "Severity: medium\n"
            "Recurrence: needs confirmation\n"
            f"Recommended Action: Retry after checking Ollama health. reason={reason or 'unknown'}"
        )

    def _parse_response(self, raw: str) -> LogAIClassification:
        error_type = ""
        severity = "medium"
        recurrence = "needs confirmation"
        recommended_action: list[str] = []
        current_section = ""

        for raw_line in raw.strip().split("\n"):
            line = self._clean(raw_line)
            if not line:
                continue

            low = line.lower()
            if "error type" in low or "error_type" in low or "에러 유형" in line:
                current_section = "error_type"
                if ":" in line:
                    error_type = self._clean(line.split(":", 1)[-1])
                continue
            if "severity" in low or "심각도" in line:
                current_section = "severity"
                if ":" in line:
                    severity = self._clean(line.split(":", 1)[-1]) or severity
                continue
            if "recurrence" in low or "반복 가능성" in line:
                current_section = "recurrence"
                if ":" in line:
                    recurrence = self._clean(line.split(":", 1)[-1]) or recurrence
                continue
            if "recommended action" in low or "권고 조치" in line or "조치" in line:
                current_section = "recommended_action"
                if ":" in line:
                    recommended_action.append(self._clean(line.split(":", 1)[-1]))
                continue

            if current_section == "error_type" and not error_type:
                error_type = line
            elif current_section == "severity":
                severity = line
            elif current_section == "recurrence":
                recurrence = line
            elif current_section == "recommended_action":
                recommended_action.append(line)

        if not error_type:
            raw_lower = raw.lower()
            if "connection" in raw_lower or "refused" in raw_lower or "timeout" in raw_lower:
                error_type = "Network error"
            elif "outofmemory" in raw_lower or "heap" in raw_lower or "memory" in raw_lower:
                error_type = "Memory error"
            elif "sql" in raw_lower or "database" in raw_lower or "db" in raw_lower:
                error_type = "Database error"
            else:
                error_type = "Unknown"
        if not recommended_action:
            recommended_action = ["Collect stack trace and verify service dependencies first."]

        return LogAIClassification(
            error_type=error_type,
            severity=severity,
            recurrence=recurrence,
            recommended_action="\n".join(recommended_action[:3]),
            raw_response=raw,
        )
