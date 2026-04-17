"""
로그 에러 AI 분류기
수집 모듈 로그에서 감지된 에러를 AI로 분류하고 심각도 판단
"""
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


SYSTEM_PROMPT = """당신은 APM 수집 인프라를 담당하는 시니어 운영 엔지니어입니다.
수집 모듈(collector, forwarder, agent)의 로그에서 발생한 에러를 분석하여 빠른 대응을 지원합니다.

분석 시 다음 항목을 반드시 포함하세요:
1. 에러 유형 (네트워크 오류 / DB 오류 / 메모리 오류 / 설정 오류 / 프로세스 오류 / 기타)
2. 심각도 (높음 / 중간 / 낮음)
3. 반복 가능성 (일시적 / 지속적 / 확인 필요)
4. 권고 조치 (1~2줄로 간결하게)

응답은 한국어로 간결하게 작성하세요."""


@dataclass
class LogAIClassification:
    error_type: str         # 에러 유형
    severity: str           # 높음 / 중간 / 낮음
    recurrence: str         # 일시적 / 지속적 / 확인 필요
    recommended_action: str # 권고 조치
    raw_response: str       # 전체 AI 응답


class LogAIClassifier:

    def __init__(self):
        self._llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )

    def classify(self, context: str) -> LogAIClassification:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"다음 수집 모듈 로그 에러를 분석해주세요:\n\n{context}"),
        ]

        response = self._llm.invoke(messages)
        raw = response.content

        return self._parse_response(raw)

    @staticmethod
    def _clean(text: str) -> str:
        """마크다운 기호 제거"""
        import re
        text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
        return text.strip()

    def _parse_response(self, raw: str) -> LogAIClassification:
        error_type = ""
        severity = "중간"
        recurrence = "확인 필요"
        action_lines = []

        current_section = None

        for line in raw.strip().split("\n"):
            line_stripped = self._clean(line)
            if not line_stripped:
                continue

            if "에러 유형" in line_stripped:
                current_section = "error_type"
                # 같은 줄에 값이 있는 경우 (예: "에러 유형: 네트워크 오류")
                if ":" in line_stripped:
                    val = self._clean(line_stripped.split(":", 1)[-1])
                    if val:
                        error_type = val
            elif "심각도" in line_stripped:
                current_section = "severity"
                if ":" in line_stripped:
                    val = line_stripped.split(":", 1)[-1].strip()
                    if "높음" in val:
                        severity = "높음"
                    elif "낮음" in val:
                        severity = "낮음"
            elif "반복 가능성" in line_stripped:
                current_section = "recurrence"
                if ":" in line_stripped:
                    val = line_stripped.split(":", 1)[-1].strip()
                    if "일시적" in val:
                        recurrence = "일시적"
                    elif "지속적" in val:
                        recurrence = "지속적"
            elif "권고 조치" in line_stripped or "조치" in line_stripped:
                current_section = "action"
            else:
                if current_section == "error_type" and not error_type:
                    error_type = line_stripped
                elif current_section == "severity":
                    if "높음" in line_stripped:
                        severity = "높음"
                    elif "낮음" in line_stripped:
                        severity = "낮음"
                elif current_section == "recurrence":
                    if "일시적" in line_stripped:
                        recurrence = "일시적"
                    elif "지속적" in line_stripped:
                        recurrence = "지속적"
                elif current_section == "action":
                    action_lines.append(line_stripped)

        # fallback: 전체 응답에서 키워드로 에러 유형 추출
        if not error_type:
            raw_lower = raw.lower()
            if "네트워크" in raw or "connection" in raw_lower or "refused" in raw_lower:
                error_type = "네트워크 오류"
            elif "메모리" in raw or "outofmemory" in raw_lower or "heap" in raw_lower:
                error_type = "메모리 오류"
            elif "db" in raw_lower or "database" in raw_lower or "sql" in raw_lower:
                error_type = "DB 오류"
            elif "ssl" in raw_lower or "certificate" in raw_lower or "인증서" in raw:
                error_type = "설정 오류 (SSL)"
            elif "timeout" in raw_lower or "타임아웃" in raw:
                error_type = "네트워크 오류 (타임아웃)"
            else:
                error_type = "기타"

        if not action_lines:
            action_lines = [self._clean(raw.strip()[:300])]

        return LogAIClassification(
            error_type=error_type,
            severity=severity,
            recurrence=recurrence,
            recommended_action="\n".join(action_lines[:3]),
            raw_response=raw,
        )
