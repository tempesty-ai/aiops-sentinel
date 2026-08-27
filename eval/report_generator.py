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

METRIC_LABELS = {
    "overall_score": "Overall score",
    "apm_fault_type_accuracy": "APM fault-type accuracy",
    "log_error_type_accuracy": "Log error-type accuracy",
    "cause_grounding": "Cause grounding",
    "action_grounding": "Action grounding",
    "relevancy": "Relevancy",
    "faithfulness": "Faithfulness",
}

JUDGE_AXES = (
    ("Relevancy", "relevancy_score", "relevancy_reason"),
    ("Faithfulness", "faithfulness_score", "faithfulness_reason"),
)

GROUNDING_AXES = (
    ("Cause grounding", "cause_grounding_score", "cause_grounding_unsupported", "cause_grounding_reason"),
    ("Action grounding", "action_grounding_score", "action_grounding_unsupported", "action_grounding_reason"),
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
            <h2>Alerts</h2>
            <div class="cards">
                {_card(str(len(records)), "Total")}
                {_card(str(critical), "Critical", "fail" if critical else "")}
                {_card(str(apm), "APM")}
                {_card(str(len(records) - apm), "Log")}
            </div>
            <div class="scroll">
            <table>
                <thead><tr><th>Time</th><th>Type</th><th>Source</th><th>Severity</th>
                <th>Fault</th><th>Root cause</th><th>Action</th></tr></thead>
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
            verdict, cls = "not measured", "warn"
        elif name in failed:
            verdict, cls = "below threshold", "fail"
        elif name in thin:
            verdict, cls = "thin sample", "warn"
        else:
            verdict, cls = "ok", "pass"
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
    return rows or '<tr><td colspan="7" class="muted">No metrics</td></tr>'


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
                f'<span class="jscore muted">not measured</span></div>'
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
            f"<details><summary>detail</summary>{detail}</details>"
            "</td></tr>"
        )
    return rows or '<tr><td colspan="5" class="muted">No cases</td></tr>'


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
    return rows or '<tr><td colspan="4" class="muted">No per-scenario data</td></tr>'


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
        halluc_detail = "not measured"
    else:
        halluc_detail = (
            f"{halluc.get('cases_with_hallucination')} of {halluc.get('cases_measurable')} cases "
            f"asserted something the metrics do not support "
            f"(diagnosis {axis.get('cause', 0)}, action {axis.get('action', 0)})"
            + (f" &middot; most often {_escape(common)}" if common else "")
        )

    meta_bits = " &middot; ".join(
        _escape(x) for x in (
            f"dataset {meta.get('dataset_version', '?')}",
            f"prompt {meta.get('prompt_version', '?')}",
            f"analysis {meta.get('analysis_model', '?')}",
            f"judge {meta.get('judge_model', '?')} ({meta.get('judge_mode', '?')})",
            f"ollama {meta.get('ollama_version', '?')}",
        )
    )
    if meta.get("self_grading"):
        meta_bits += ' &middot; <span class="badge warning">SELF-GRADING</span>'

    failed = ", ".join(gate.get("failed_rules") or []) or "-"
    unmeasured = ", ".join(gate.get("unmeasured_rules") or []) or "-"
    thin = ", ".join(gate.get("insufficient_sample_rules") or []) or "-"

    return f"""
        <section>
            <h2>AI evaluation</h2>
            <div class="cards">
                {_card(_pct(eval_data.get("overall_score")), "Overall score")}
                {_card(status, "Quality gate", status_cls)}
                {_card(_pct(rate), "Hallucination rate", "fail" if (rate or 0) > 0.3 else "")}
                {_card(cases_text, "Cases evaluated")}
            </div>
            <p class="muted">{meta_bits}</p>
            <p class="muted">schema {_escape(eval_data.get("schema_version", "?"))}
               &middot; generated {_escape(eval_data.get("timestamp", "?"))}</p>

            <h3>Gate metrics</h3>
            <div class="scroll">
            <table>
                <thead><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Margin</th>
                <th>n</th><th>Coverage</th><th>Verdict</th></tr></thead>
                <tbody>{_metric_rows(gate)}</tbody>
            </table>
            </div>
            <p><strong>Failed:</strong> {_escape(failed)}
               &nbsp;<strong>Unmeasured:</strong> {_escape(unmeasured)}
               &nbsp;<strong>Thin sample:</strong> {_escape(thin)}</p>

            <h3>Hallucination</h3>
            <p>{halluc_detail}</p>
            <p class="muted">derived from cause/action grounding &mdash; reported, not gated</p>

            <h3>By scenario</h3>
            <p class="muted">Weakest first. Accuracy splitting across phrasings means the
               classifier is sensitive to wording, not merely noisy.</p>
            <div class="scroll">
            <table>
                <thead><tr><th>Scenario</th><th>Cases</th><th>Accuracy</th><th>By phrasing</th></tr></thead>
                <tbody>{_scenario_rows(eval_data.get("by_scenario"))}</tbody>
            </table>
            </div>

            <h3>APM cases</h3>
            <p class="muted">Lowest score first. Open <em>detail</em> for the grounding and judge
               verdicts and the answer excerpt.</p>
            <div class="scroll">
            <table>
                <thead><tr><th>Case</th><th>Scenario</th><th>Fault type</th><th>Correct</th><th>Score</th></tr></thead>
                <tbody>{_case_rows(eval_data.get("apm_eval") or [], "fault_type", "fault_type_correct")}</tbody>
            </table>
            </div>

            <h3>Log cases</h3>
            <div class="scroll">
            <table>
                <thead><tr><th>Case</th><th>Scenario</th><th>Error type</th><th>Correct</th><th>Score</th></tr></thead>
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
    <h1>AIOps Sentinel Report</h1>
    <p class="muted">Generated {now} &middot; analysis model {_escape(OLLAMA_MODEL)}</p>
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
