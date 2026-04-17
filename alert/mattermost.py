"""
Mattermost Incoming Webhook 알람 모듈
APM 장애 알람 / 로그 에러 알람을 Mattermost 채널로 전송
"""
import requests
from apm.anomaly_detector import AnomalyResult
from apm.ai_analyzer import AIAnalysisResult
from logwatch.log_tailer import LogErrorEvent
from logwatch.ai_classifier import LogAIClassification
from config.settings import MATTERMOST_WEBHOOK_URL


SEVERITY_ICON = {
    "심각": ":red_circle:",
    "경고": ":large_orange_circle:",
    "주의": ":large_yellow_circle:",
    "높음": ":red_circle:",
    "중간": ":large_orange_circle:",
    "낮음": ":large_yellow_circle:",
    "정상": ":white_check_mark:",
}


def _send(payload: dict) -> bool:
    if not MATTERMOST_WEBHOOK_URL:
        print("[Mattermost] WEBHOOK URL 미설정 - 콘솔 출력으로 대체")
        print(payload.get("text", ""))
        for att in payload.get("attachments", []):
            print(att.get("text", ""))
        return True

    try:
        resp = requests.post(MATTERMOST_WEBHOOK_URL, json=payload, timeout=5)
        return resp.status_code == 200
    except requests.RequestException as e:
        print(f"[Mattermost] 전송 실패: {e}")
        return False


def send_apm_alert(anomaly: AnomalyResult, analysis: AIAnalysisResult) -> bool:
    """APM 장애 알람 전송"""
    s = anomaly.snapshot
    icon = SEVERITY_ICON.get(anomaly.severity, ":large_yellow_circle:")
    color = "#FF0000" if anomaly.severity == "심각" else "#FF8C00"

    triggered = "\n".join(f"• {r}" for r in anomaly.triggered_rules)

    text = f"{icon} **[{anomaly.severity}] APM 장애 감지 - {s.server}**"

    attachment_text = f"""**📊 이상 지표**
{triggered}

**🤖 AI 분석 결과**
**장애 유형:** {analysis.fault_type}
**근본 원인:** {analysis.root_cause}
**즉각 조치:** {analysis.immediate_action}
**재발 방지:** {analysis.prevention}

**⏰ 감지 시각:** {s.timestamp}
**📈 주요 수치:** CPU {s.cpu_usage}% | 응답 {s.response_time_ms:.0f}ms | DB커넥션 {s.db_connections}/{s.db_connection_max} | 에러율 {s.error_rate}%"""

    payload = {
        "text": text,
        "attachments": [
            {
                "color": color,
                "text": attachment_text,
            }
        ],
    }

    return _send(payload)


def send_log_alert(event: LogErrorEvent, classification: LogAIClassification) -> bool:
    """로그 에러 알람 전송"""
    icon = SEVERITY_ICON.get(classification.severity, ":large_yellow_circle:")
    color = "#FF0000" if classification.severity == "높음" else "#FF8C00"

    # 로그 라인이 너무 길면 자르기
    raw_line_short = event.raw_line[:200] + "..." if len(event.raw_line) > 200 else event.raw_line

    text = f"{icon} **[{classification.severity}] 수집 모듈 오류 - {event.module_name}**"

    attachment_text = f"""**📄 오류 로그**
```
{raw_line_short}
```

**🤖 AI 분류 결과**
**에러 유형:** {classification.error_type}
**반복 가능성:** {classification.recurrence}
**권고 조치:** {classification.recommended_action}

**⏰ 감지 시각:** {event.timestamp}
**📁 파일:** {event.log_file}
**🔑 감지 키워드:** `{event.keyword_matched}`"""

    payload = {
        "text": text,
        "attachments": [
            {
                "color": color,
                "text": attachment_text,
            }
        ],
    }

    return _send(payload)
