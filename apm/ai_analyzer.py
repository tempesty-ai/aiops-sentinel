"""
APM anomaly AI analyzer.
Uses LangChain + Ollama with retry/fallback behavior for reliability.
"""
from dataclasses import dataclass
import time

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import APM_PROMPT_VERSION, OLLAMA_BASE_URL, OLLAMA_MODEL


_SECTIONS = """Return short, operational output with these exact sections:
1. Fault Type
2. Root Cause
3. Immediate Action
4. Prevention
5. Severity"""

# Prompt variants kept side by side so an A/B run changes one variable only.
# Select with APM_PROMPT_VERSION; the active version is recorded in eval reports.
APM_PROMPT_VERSIONS = {
    # v1 - baseline. No instruction about grounding.
    "v1": f"""You are a senior APM incident analyst.
{_SECTIONS}

Keep answers practical and concise.""",

    # v2 - forces every claim to cite the metric it rests on, and forbids
    # reasoning about metrics absent from the input.
    "v2": f"""You are a senior APM incident analyst.
{_SECTIONS}

Ground every statement in the metrics you were given:
- For each claim, cite the metric name and value it rests on, e.g. "cpu=92.5%".
- Do not assert a cause that the given metrics do not support.
- If a metric is not present in the input, do not reason about it at all.

Keep answers practical and concise.""",
}


def get_system_prompt(version: str = "") -> str:
    """Resolve a prompt variant, falling back to the baseline for unknown names."""
    return APM_PROMPT_VERSIONS.get(version or APM_PROMPT_VERSION, APM_PROMPT_VERSIONS["v1"])


SYSTEM_PROMPT = get_system_prompt()


@dataclass
class AIAnalysisResult:
    fault_type: str
    root_cause: str
    immediate_action: str
    prevention: str
    severity: str
    raw_response: str
    llm_failed: bool = False          # every retry exhausted -> fallback text was used
    fault_type_missing: bool = False  # no parsable Fault Type section in the response


class APMAIAnalyzer:

    def __init__(self, max_retries: int = 2, retry_backoff_seconds: float = 0.5, prompt_version: str = ""):
        self.prompt_version = prompt_version or APM_PROMPT_VERSION
        self.system_prompt = get_system_prompt(self.prompt_version)
        self._llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def analyze(self, context: str) -> AIAnalysisResult:
        raw, llm_failed = self._invoke_with_retry(context)
        result = self._parse_response(raw)
        result.llm_failed = llm_failed
        return result

    def _invoke_with_retry(self, context: str) -> tuple[str, bool]:
        last_error = ""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Analyze the following APM anomaly context:\n\n{context}"),
        ]

        for attempt in range(1, self.max_retries + 2):
            try:
                response = self._llm.invoke(messages)
                content = str(getattr(response, "content", "")).strip()
                if not content:
                    raise ValueError("LLM returned empty content")
                return content, False
            except Exception as exc:  # pragma: no cover - runtime integration
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt <= self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        return self._fallback_response(last_error), True

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
            "Fault Type: Unknown (LLM unavailable)\n"
            f"Root Cause: LLM call failed or timed out. reason={reason or 'unknown'}\n"
            "Immediate Action: Check Ollama service health and retry analysis.\n"
            "Prevention: Add retry/timeout monitoring and fallback runbook.\n"
            "Severity: warning"
        )

    def _parse_response(self, raw: str) -> AIAnalysisResult:
        fault_type = ""
        root_cause: list[str] = []
        immediate_action: list[str] = []
        prevention: list[str] = []
        severity = ""
        current_section = ""

        for raw_line in raw.strip().split("\n"):
            line = self._clean(raw_line)
            if not line:
                continue

            low = line.lower()
            if "fault type" in low or "fault_type" in low or "장애 유형" in line:
                current_section = "fault_type"
                if ":" in line:
                    fault_type = self._clean(line.split(":", 1)[-1])
                continue
            if "root cause" in low or "근본 원인" in line:
                current_section = "root_cause"
                if ":" in line:
                    root_cause.append(self._clean(line.split(":", 1)[-1]))
                continue
            if "immediate action" in low or "즉각 조치" in line:
                current_section = "immediate_action"
                if ":" in line:
                    immediate_action.append(self._clean(line.split(":", 1)[-1]))
                continue
            if "prevention" in low or "재발 방지" in line:
                current_section = "prevention"
                if ":" in line:
                    prevention.append(self._clean(line.split(":", 1)[-1]))
                continue
            if "severity" in low or "심각도" in line:
                current_section = "severity"
                if ":" in line:
                    severity = self._clean(line.split(":", 1)[-1])
                continue

            if current_section == "fault_type" and not fault_type:
                fault_type = line
            elif current_section == "root_cause":
                root_cause.append(line)
            elif current_section == "immediate_action":
                immediate_action.append(line)
            elif current_section == "prevention":
                prevention.append(line)
            elif current_section == "severity" and not severity:
                severity = line

        fault_type_missing = not fault_type
        if fault_type_missing:
            fault_type = "Unknown"
        if not root_cause:
            root_cause = ["No root-cause section was returned by the model."]
        if not immediate_action:
            immediate_action = ["Review anomaly context and trigger on-call runbook."]
        if not prevention:
            prevention = ["Tune thresholds and add guardrail tests for recurring cases."]
        if not severity:
            severity = "warning"

        return AIAnalysisResult(
            fault_type=fault_type,
            root_cause="\n".join(root_cause),
            immediate_action="\n".join(immediate_action),
            prevention="\n".join(prevention),
            severity=severity,
            raw_response=raw,
            fault_type_missing=fault_type_missing,
        )
