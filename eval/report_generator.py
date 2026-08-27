"""
HTML report generator for runtime alerts + eval outcomes.

Written for a 102-case run: the reader's first question is "what failed", so
failures sort to the top, per-case detail stays collapsed, and every metric is
shown next to its threshold and verdict rather than as a comma-separated line.
"""
import os
from dataclasses import dataclass
from datetime import datetime

from config.settings import OLLAMA_MODEL

# Display labels. The JSON keeps the English keys as identifiers.
METRIC_LABELS = {
    "overall_score": "종합 점수",
    "apm_fault_type_accuracy": "APM 장애유형 정확도",
    "log_error_type_accuracy": "로그 에러유형 정확도",
    "cause_grounding": "원인 정합성",
    "action_grounding": "조치 정합성",
    "relevancy": "과제 정합성 (심판)",
    "faithfulness": "근거 충실성 (심판)",
}

VERDICT_OK = "충족"
VERDICT_BELOW = "임계값 미달"
VERDICT_UNMEASURED = "측정 불가"
VERDICT_THIN = "표본 부족"

JUDGE_AXES = (
    ("과제 정합성", "relevancy_score", "relevancy_reason"),
    ("근거 충실성", "faithfulness_score", "faithfulness_reason"),
)

GROUNDING_AXES = (
    ("원인 정합성", "cause_grounding_score", "cause_grounding_unsupported", "cause_grounding_reason"),
    ("조치 정합성", "action_grounding_score", "action_grounding_unsupported", "action_grounding_reason"),
)


@dataclass
class AlertRecord:
    timestamp: str
    alert_type: str
    source: str
    severity: str
    fault_type: str
    root_cause: str
    action: str
    raw_ai_response: str


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _severity_badge(severity: str) -> str:
    low = str(severity).lower()
    if severity in ("심각", "높음", "critical") or "critical" in low:
        return f'<span class="badge critical">{_escape(severity)}</span>'
    if severity in ("경고", "중간", "warning") or "warn" in low:
        return f'<span class="badge warning">{_escape(severity)}</span>'
    return f'<span class="badge normal">{_escape(severity)}</span>'


def _pct(value, digits: int = 0) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def _metric_names(keys) -> str:
    """Show rule names the way the metric table shows them."""
    return ", ".join(METRIC_LABELS.get(key, key) for key in (keys or [])) or "-"


def _card(value: str, label: str, cls: str = "") -> str:
    return f'<div class="card {cls}"><div class="value">{value}</div><div class="label">{_escape(label)}</div></div>'


def _alert_section(records: list[AlertRecord]) -> str:
    """Omitted entirely when empty - an eval-only run has no alerts, and a table
    of zeroes at the top of the page only pushes the findings down."""
    if not records:
        return ""

    rows = ""
    for r in sorted(records, key=lambda x: x.timestamp, reverse=True):
        rows += (
            "<tr>"
            f"<td>{_escape(r.timestamp)}</td><td>{_escape(r.alert_type)}</td>"
            f"<td>{_escape(r.source)}</td><td>{_severity_badge(r.severity)}</td>"
            f"<td>{_escape(r.fault_type)}</td>"
            f"<td>{_escape(r.root_cause).replace(chr(10), '<br>')}</td>"
            f"<td>{_escape(r.action).replace(chr(10), '<br>')}</td>"
            "</tr>"
        )
    critical = sum(1 for r in records if str(r.severity).lower() in ("critical", "높음", "심각"))
    apm = sum(1 for r in records if r.alert_type == "APM")
    return f"""
        <section>
            <h2>알람</h2>
            <div class="cards">
                {_card(str(len(records)), "전체")}
                {_card(str(critical), "심각", "fail" if critical else "")}
                {_card(str(apm), "APM")}
                {_card(str(len(records) - apm), "로그")}
            </div>
            <div class="scroll">
            <table>
                <thead><tr><th>시각</th><th>구분</th><th>대상</th><th>심각도</th>
                <th>장애유형</th><th>원인</th><th>조치</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            </div>
        </section>
    """


def _metric_rows(gate: dict) -> str:
    metrics = gate.get("metrics") or {}
    counts = gate.get("sample_counts") or {}
    coverage = gate.get("coverage") or {}
    thresholds = gate.get("thresholds") or {}
    failed = set(gate.get("failed_rules") or [])
    unmeasured = set(gate.get("unmeasured_rules") or [])
    thin = set(gate.get("insufficient_sample_rules") or [])

    rows = ""
    for name, value in metrics.items():
        threshold = thresholds.get(f"{name}_min")
        if name in unmeasured:
            verdict, cls = VERDICT_UNMEASURED, "warn"
        elif name in failed:
            verdict, cls = VERDICT_BELOW, "fail"
        elif name in thin:
            verdict, cls = VERDICT_THIN, "warn"
        else:
            verdict, cls = VERDICT_OK, "pass"
        margin = "-" if (value is None or threshold is None) else f"{value - threshold:+.2f}"
        rows += (
            f'<tr class="{cls}">'
            f"<td>{_escape(METRIC_LABELS.get(name, name))}</td>"
            f'<td class="num">{"-" if value is None else f"{value:.2f}"}</td>'
            f'<td class="num">{"-" if threshold is None else f"{threshold:.2f}"}</td>'
            f'<td class="num">{margin}</td>'
            f'<td class="num">{counts.get(name, "?")}</td>'
            f'<td class="num">{_pct(coverage.get(name))}</td>'
            f"<td>{verdict}</td>"
            "</tr>"
        )
    return rows or '<tr><td colspan="7" class="muted">지표 없음</td></tr>'


def _grounding_block(case: dict) -> str:
    rows = ""
    for label, score_key, unsupported_key, reason_key in GROUNDING_AXES:
        score = case.get(score_key)
        if score is None:
            continue
        cls = "jreason ungrounded" if (case.get(unsupported_key) or []) else "jreason"
        rows += (
            f'<div class="judge"><span class="jname">{label}</span>'
            f'<span class="jscore">{score}</span>'
            f'<span class="{cls}">{_escape(case.get(reason_key) or "")}</span></div>'
        )
    return rows


def _judge_block(case: dict) -> str:
    rows = ""
    for label, score_key, reason_key in JUDGE_AXES:
        score = case.get(score_key, -1)
        if not isinstance(score, (int, float)) or score < 0:
            rows += (
                f'<div class="judge"><span class="jname">{label}</span>'
                f'<span class="jscore muted">{VERDICT_UNMEASURED}</span></div>'
            )
            continue
        rows += (
            f'<div class="judge"><span class="jname">{label}</span>'
            f'<span class="jscore">{score}</span>'
            f'<span class="jreason">{_escape(case.get(reason_key) or "")}</span></div>'
        )
    return rows


def _case_rows(cases: list[dict], type_key: str, correct_key: str) -> str:
    """Worst first: at 102 cases the reader is looking for what failed."""
    def sort_key(case: dict):
        return (
            float(case.get("overall_score") or 0),
            1 if case.get(correct_key) else 0,
        )

    rows = ""
    for case in sorted(cases, key=sort_key):
        unmeasured = case.get("classification_unmeasured")
        mark = "-" if unmeasured else ("Y" if case.get(correct_key) else "N")
        cls = "" if (unmeasured or case.get(correct_key)) else "fail"
        detail = _grounding_block(case) + _judge_block(case)
        excerpt = case.get("answer_excerpt")
        if excerpt:
            detail += f"<pre>{_escape(excerpt)}</pre>"
        rows += (
            f'<tr class="{cls}">'
            f'<td>{_escape(case.get("case_id") or "")}</td>'
            f'<td>{_escape(case.get("scenario") or "")}</td>'
            f'<td>{_escape(case.get(type_key) or "")}</td>'
            f"<td>{mark}</td>"
            f'<td class="num">{_pct(case.get("overall_score"))}</td>'
            "</tr>"
            f'<tr class="detail"><td colspan="5">'
            f"<details><summary>상세</summary>{detail}</details>"
            "</td></tr>"
        )
    return rows or '<tr><td colspan="5" class="muted">케이스 없음</td></tr>'


def _scenario_rows(by_scenario: dict) -> str:
    rows = ""
    for key, entry in (by_scenario or {}).items():
        accuracy = entry.get("accuracy")
        variants = entry.get("by_variant") or {}
        variant_text = ", ".join(
            f"{name} {_pct(v.get('accuracy'))}" for name, v in sorted(variants.items())
        ) or "-"
        cls = "fail" if accuracy is not None and accuracy < 0.7 else ""
        rows += (
            f'<tr class="{cls}"><td>{_escape(key)}</td>'
            f'<td class="num">{entry.get("cases", 0)}</td>'
            f'<td class="num">{_pct(accuracy)}</td>'
            f"<td>{_escape(variant_text)}</td></tr>"
        )
    return rows or '<tr><td colspan="4" class="muted">시나리오별 데이터 없음</td></tr>'


def _eval_section(eval_data: dict) -> str:
    gate = eval_data.get("quality_gate") or {}
    meta = eval_data.get("run_metadata") or {}
    halluc = eval_data.get("hallucination") or {}

    status = str(gate.get("status") or ("PASS" if gate.get("passed") else "FAIL"))
    status_cls = {"PASS": "pass", "INCONCLUSIVE": "warn"}.get(status, "fail")

    evaluated = meta.get("cases_evaluated")
    available = meta.get("cases_available")
    cases_text = f"{evaluated}/{available}" if evaluated and available else "?"

    rate = halluc.get("rate")
    axis = halluc.get("by_axis") or {}
    common = ", ".join(f"{k} x{v}" for k, v in (halluc.get("most_common_unsupported") or {}).items())
    if rate is None:
        halluc_detail = VERDICT_UNMEASURED
    else:
        halluc_detail = (
            f"측정 가능한 {halluc.get('cases_measurable')}건 중 "
            f"<strong>{halluc.get('cases_with_hallucination')}건</strong>이 "
            f"관측 지표가 뒷받침하지 않는 내용을 단정했습니다 "
            f"(원인 단계 {axis.get('cause', 0)}건, 조치 단계 {axis.get('action', 0)}건)"
            + (f" &middot; 가장 자주 지목된 리소스: {_escape(common)}" if common else "")
        )

    meta_bits = " &middot; ".join(
        _escape(x) for x in (
            f"데이터셋 {meta.get('dataset_version', '?')}",
            f"프롬프트 {meta.get('prompt_version', '?')}",
            f"분석 모델 {meta.get('analysis_model', '?')}",
            f"심판 모델 {meta.get('judge_model', '?')} ({meta.get('judge_mode', '?')} 모드)",
            f"Ollama {meta.get('ollama_version', '?')}",
        )
    )
    if meta.get("self_grading"):
        meta_bits += ' &middot; <span class="badge warning">자기채점</span>'

    failed = _metric_names(gate.get("failed_rules"))
    unmeasured = _metric_names(gate.get("unmeasured_rules"))
    thin = _metric_names(gate.get("insufficient_sample_rules"))

    return f"""
        <section>
            <h2>AI 품질 평가</h2>
            <div class="cards">
                {_card(_pct(eval_data.get("overall_score")), "종합 점수")}
                {_card(status, "품질 게이트", status_cls)}
                {_card(_pct(rate), "환각률", "fail" if (rate or 0) > 0.3 else "")}
                {_card(cases_text, "평가 케이스")}
            </div>
            <p class="muted">{meta_bits}</p>
            <p class="muted">스키마 {_escape(eval_data.get("schema_version", "?"))}
               &middot; 생성 {_escape(eval_data.get("timestamp", "?"))}</p>

            <h3>게이트 지표</h3>
            <div class="scroll">
            <table>
                <thead><tr><th>지표</th><th>값</th><th>임계값</th><th>여유</th>
                <th>표본</th><th>커버리지</th><th>판정</th></tr></thead>
                <tbody>{_metric_rows(gate)}</tbody>
            </table>
            </div>
            <p><strong>미달:</strong> {_escape(failed)}
               &nbsp;<strong>측정 불가:</strong> {_escape(unmeasured)}
               &nbsp;<strong>표본 부족:</strong> {_escape(thin)}</p>

            <h3>환각률</h3>
            <p>{halluc_detail}</p>
            <p class="muted">원인·조치 정합성에서 파생한 집계 &mdash; 게이트 지표는 아님</p>

            <h3>시나리오별</h3>
            <p class="muted">약한 순. 같은 유형인데 문구별로 정확도가 갈리면 노이즈가 아니라
               표현에 취약한 것입니다.</p>
            <div class="scroll">
            <table>
                <thead><tr><th>시나리오</th><th>케이스</th><th>정확도</th><th>문구별</th></tr></thead>
                <tbody>{_scenario_rows(eval_data.get("by_scenario"))}</tbody>
            </table>
            </div>

            <h3>APM 케이스</h3>
            <p class="muted">점수 낮은 순. <em>상세</em>를 펼치면 정합성·심판 판정과 AI 응답 원문이 나옵니다.</p>
            <div class="scroll">
            <table>
                <thead><tr><th>케이스</th><th>시나리오</th><th>장애유형</th><th>정답</th><th>점수</th></tr></thead>
                <tbody>{_case_rows(eval_data.get("apm_eval") or [], "fault_type", "fault_type_correct")}</tbody>
            </table>
            </div>

            <h3>로그 케이스</h3>
            <div class="scroll">
            <table>
                <thead><tr><th>케이스</th><th>시나리오</th><th>에러유형</th><th>정답</th><th>점수</th></tr></thead>
                <tbody>{_case_rows(eval_data.get("log_eval") or [], "error_type", "error_type_correct")}</tbody>
            </table>
            </div>
        </section>
    """


STYLE = """
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 28px 20px 60px; background: #0f172a; color: #e2e8f0;
           font: 14px/1.6 -apple-system, "Segoe UI", Roboto, "Malgun Gothic", sans-serif; }
    main { max-width: 1180px; margin: 0 auto; }
    h1 { font-size: 20px; margin: 0 0 4px; }
    h2 { font-size: 17px; margin: 34px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #334155; }
    h3 { font-size: 14px; margin: 26px 0 8px; color: #93c5fd; }
    p { margin: 6px 0; }
    .muted { color: #94a3b8; font-size: 12px; }
    .cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 14px 0 18px; }
    .card { background: #1e293b; padding: 12px 18px; border-radius: 10px;
            border: 1px solid #334155; min-width: 150px; }
    .card .value { font-size: 22px; font-weight: 600; }
    .card .label { font-size: 12px; color: #94a3b8; margin-top: 2px; }
    .card.pass { border-color: #10b981; }
    .card.fail { border-color: #ef4444; }
    .card.warn { border-color: #f59e0b; }
    .scroll { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #263449; vertical-align: top; }
    th { background: #172033; color: #93c5fd; font-weight: 600; white-space: nowrap; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    tr.pass td:last-child { color: #6ee7b7; }
    tr.fail td { color: #fca5a5; }
    tr.warn td:last-child { color: #fcd34d; }
    tr.detail td { background: #131c2e; border-bottom: 1px solid #263449; padding: 4px 10px 10px; }
    tr.detail summary { cursor: pointer; color: #64748b; font-size: 12px; padding: 4px 0; }
    .judge { display: grid; grid-template-columns: 130px 54px 1fr; gap: 8px;
             align-items: start; padding: 3px 0; font-size: 12px; }
    .jname { color: #93c5fd; }
    .jscore { color: #e2e8f0; font-variant-numeric: tabular-nums; }
    .jreason { color: #94a3b8; }
    .jreason.ungrounded { color: #fca5a5; }
    pre { white-space: pre-wrap; margin: 8px 0 0; padding: 9px; background: #0b1220;
          border-radius: 6px; overflow-x: auto; font-size: 12px; color: #94a3b8; }
    .badge { display: inline-block; padding: 1px 8px; border-radius: 999px;
             border: 1px solid #475569; font-size: 11px; }
    .badge.critical { border-color: #ef4444; color: #fca5a5; }
    .badge.warning { border-color: #f59e0b; color: #fcd34d; }
"""


def generate_html_report(
    alert_records: list[AlertRecord],
    eval_data: dict | None = None,
    output_path: str = "reports/aiops_report.html",
) -> str:
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIOps Sentinel Report</title>
<style>{STYLE}</style>
</head>
<body>
<main>
    <h1>AIOps Sentinel 리포트</h1>
    <p class="muted">생성 {now} &middot; 분석 모델 {_escape(OLLAMA_MODEL)}</p>
    {_alert_section(alert_records)}
    {_eval_section(eval_data) if eval_data else ""}
</main>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(html)
    print(f"[Report] HTML report saved: {output_path}")
    return output_path
