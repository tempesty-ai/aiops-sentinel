"""
APM 장애 AI 분석기
LangChain + Ollama (llama3.1:70b) 를 사용하여
장애 원인 분석 및 조치 방안을 생성
"""
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


SYSTEM_PROMPT = """당신은 10년 이상의 경험을 가진 APM(Application Performance Monitoring) 전문 엔지니어입니다.
WAS 서버의 모니터링 지표를 분석하여 장애 원인을 정확하게 진단하고 구체적인 조치 방안을 제시합니다.

분석 시 다음 항목을 반드시 포함하세요:
1. 장애 유형 분류 (CPU 폭증 / 메모리 누수 / DB 커넥션 풀 고갈 / 응답 지연 / 에러율 급증 / 복합)
2. 근본 원인 (가장 가능성 높은 원인 1~2가지)
3. 즉각 조치 방안 (지금 당장 해야 할 것)
4. 재발 방지 방안 (중장기 개선 방안)
5. 심각도 평가 (심각 / 경고 / 주의)

응답은 반드시 한국어로, 간결하고 실무적으로 작성하세요.
각 항목은 명확하게 구분하여 작성하세요."""


@dataclass
class AIAnalysisResult:
    fault_type: str       # 장애 유형
    root_cause: str       # 근본 원인
    immediate_action: str # 즉각 조치
    prevention: str       # 재발 방지
    severity: str         # 심각도
    raw_response: str     # 전체 AI 응답


class APMAIAnalyzer:

    def __init__(self):
        self._llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,  # 일관성 있는 분석을 위해 낮게 설정
        )

    def analyze(self, context: str) -> AIAnalysisResult:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"다음 APM 모니터링 데이터를 분석해주세요:\n\n{context}"),
        ]

        response = self._llm.invoke(messages)
        raw = response.content

        return self._parse_response(raw)

    @staticmethod
    def _clean(text: str) -> str:
        """마크다운 기호 제거"""
        import re
        text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)  # **bold**, *italic*
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # ## 헤더
        text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)  # 리스트 기호
        return text.strip()

    def _parse_response(self, raw: str) -> AIAnalysisResult:
        lines = raw.strip().split("\n")

        fault_type = ""
        root_cause = []
        immediate_action = []
        prevention = []
        severity = ""

        current_section = None

        for line in lines:
            line_lower = line.lower()
            if "장애 유형" in line or "fault type" in line_lower:
                current_section = "fault_type"
                # 같은 줄에 값이 있는 경우 처리 (예: "장애 유형: CPU 폭증")
                if ":" in line:
                    val = self._clean(line.split(":", 1)[-1])
                    if val:
                        fault_type = val
            elif "근본 원인" in line or "root cause" in line_lower:
                current_section = "root_cause"
            elif "즉각 조치" in line or "immediate" in line_lower:
                current_section = "immediate"
            elif "재발 방지" in line or "prevention" in line_lower:
                current_section = "prevention"
            elif "심각도" in line or "severity" in line_lower:
                current_section = "severity"
                if ":" in line:
                    val = self._clean(line.split(":", 1)[-1])
                    if val and not severity:
                        severity = val
            else:
                content = self._clean(line)
                if not content:
                    continue
                if current_section == "fault_type" and not fault_type:
                    fault_type = content
                elif current_section == "root_cause":
                    root_cause.append(content)
                elif current_section == "immediate":
                    immediate_action.append(content)
                elif current_section == "prevention":
                    prevention.append(content)
                elif current_section == "severity" and not severity:
                    severity = content

        # 파싱 실패 시 fallback
        if not fault_type:
            fault_type = "복합 장애"
        if not root_cause:
            root_cause = [self._clean(raw[:300])]
        if not immediate_action:
            immediate_action = ["즉각 조치 확인 필요"]
        if not severity:
            if "심각" in raw:
                severity = "심각"
            elif "경고" in raw:
                severity = "경고"
            else:
                severity = "주의"

        return AIAnalysisResult(
            fault_type=fault_type,
            root_cause="\n".join(root_cause),
            immediate_action="\n".join(immediate_action),
            prevention="\n".join(prevention),
            severity=severity,
            raw_response=raw,
        )
