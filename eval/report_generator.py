"""
HTML report generator for runtime alerts + eval outcomes.
"""
import os
from dataclasses import dataclass
from datetime import datetime

from config.settings import OLLAMA_MODEL


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
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _severity_badge(severity: str) -> str:
    low = severity.lower()
    if severity in ("심각", "높음", "critical") or "critical" in low:
        return f'<span class="badge critical">{_escape(severity)}</span>'
    if severity in ("경고", "중간", "warning") or "warn" in low:
        return f'<span class="badge warning">{_escape(severity)}</span>'
    return f'<span class="badge normal">{_escape(severity)}</span>'


def generate_html_report(
    alert_records: list[AlertRecord],
    eval_data: dict | None = None,
    output_path: str = "reports/aiops_report.html",
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total = len(alert_records)
    apm_count = sum(1 for r in alert_records if r.alert_type == "APM")
    log_count = sum(1 for r in alert_records if r.alert_type == "LOG")
    critical_count = sum(1 for r in alert_records if str(r.severity).lower() in ("critical", "높음", "심각"))

    alert_rows = ""
    for r in sorted(alert_records, key=lambda x: x.timestamp, reverse=True):
        alert_rows += (
            "<tr>"
            f"<td>{_escape(r.timestamp)}</td>"
            f"<td>{_escape(r.alert_type)}</td>"
            f"<td>{_escape(r.source)}</td>"
            f"<td>{_severity_badge(r.severity)}</td>"
            f"<td>{_escape(r.fault_type)}</td>"
            f"<td>{_escape(r.root_cause).replace(chr(10), '<br>')}</td>"
            f"<td>{_escape(r.action).replace(chr(10), '<br>')}</td>"
            "</tr>"
        )
    if not alert_rows:
        alert_rows = '<tr><td colspan="7" class="muted">No alert records</td></tr>'

    eval_section = ""
    if eval_data:
        schema_version = _escape(str(eval_data.get("schema_version", "n/a")))
        overall = int(float(eval_data.get("overall_score", 0.0)) * 100)

        gate = eval_data.get("quality_gate") or {}
        gate_passed = bool(gate.get("passed"))
        gate_class = "pass" if gate_passed else "fail"
        gate_label = "PASS" if gate_passed else "FAIL"
        failed_rules = gate.get("failed_rules", [])
        failed_text = ", ".join(failed_rules) if failed_rules else "-"
        metrics = gate.get("metrics", {})

        apm_rows = ""
        for r in eval_data.get("apm_eval", []):
            apm_rows += (
                "<tr>"
                f"<td>{_escape(str(r.get('scenario', '')))}</td>"
                f"<td>{_escape(str(r.get('fault_type', '')))}</td>"
                f"<td>{'Y' if r.get('fault_type_correct') else 'N'}</td>"
                f"<td>{int(float(r.get('overall_score', 0.0)) * 100)}%</td>"
                "</tr>"
            )
        if not apm_rows:
            apm_rows = '<tr><td colspan="4" class="muted">No APM eval rows</td></tr>'

        log_rows = ""
        for r in eval_data.get("log_eval", []):
            log_rows += (
                "<tr>"
                f"<td>{_escape(str(r.get('scenario', '')))}</td>"
                f"<td>{_escape(str(r.get('error_type', '')))}</td>"
                f"<td>{'Y' if r.get('error_type_correct') else 'N'}</td>"
                f"<td>{int(float(r.get('overall_score', 0.0)) * 100)}%</td>"
                "</tr>"
            )
        if not log_rows:
            log_rows = '<tr><td colspan="4" class="muted">No log eval rows</td></tr>'

        eval_section = f"""
        <section>
            <h2>AI Evaluation</h2>
            <div class="cards">
                <div class="card"><div class="value">{overall}%</div><div class="label">Overall Score</div></div>
                <div class="card {gate_class}"><div class="value">{gate_label}</div><div class="label">Quality Gate</div></div>
            </div>
            <p class="muted">schema_version: {schema_version}</p>
            <p><strong>Failed Rules:</strong> {_escape(failed_text)}</p>
            <p class="muted">
                Metrics:
                overall={metrics.get('overall_score', 0)},
                apm_acc={metrics.get('apm_fault_type_accuracy', 0)},
                log_acc={metrics.get('log_error_type_accuracy', 0)},
                hallucination={metrics.get('hallucination', 0)},
                relevancy={metrics.get('relevancy', 0)},
                faithfulness={metrics.get('faithfulness', 0)}
            </p>

            <h3>APM Eval</h3>
            <table>
                <thead><tr><th>Scenario</th><th>Fault Type</th><th>Correct</th><th>Overall</th></tr></thead>
                <tbody>{apm_rows}</tbody>
            </table>

            <h3>Log Eval</h3>
            <table>
                <thead><tr><th>Scenario</th><th>Error Type</th><th>Correct</th><th>Overall</th></tr></thead>
                <tbody>{log_rows}</tbody>
            </table>
        </section>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AIOps Sentinel Report</title>
  <style>
    body {{ font-family: Segoe UI, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    header {{ padding: 20px 28px; border-bottom: 1px solid #1e293b; background: #111827; }}
    h1 {{ margin: 0; font-size: 22px; }}
    h2 {{ margin-top: 24px; }}
    section {{ padding: 18px 28px; }}
    .muted {{ color: #94a3b8; }}
    .cards {{ display: flex; gap: 12px; margin: 10px 0 12px; }}
    .card {{ background: #1e293b; padding: 12px 16px; border-radius: 10px; border: 1px solid #334155; min-width: 160px; }}
    .card.pass {{ border-color: #10b981; }}
    .card.fail {{ border-color: #ef4444; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    .label {{ color: #94a3b8; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; background: #111827; border: 1px solid #1f2937; }}
    th, td {{ text-align: left; border-bottom: 1px solid #1f2937; padding: 10px; font-size: 13px; }}
    th {{ color: #93c5fd; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; border: 1px solid; }}
    .badge.critical {{ border-color: #ef4444; color: #fca5a5; }}
    .badge.warning {{ border-color: #f59e0b; color: #fcd34d; }}
    .badge.normal {{ border-color: #64748b; color: #cbd5e1; }}
  </style>
</head>
<body>
  <header>
    <h1>AIOps Sentinel Monitoring Report</h1>
    <p class="muted">Generated: {now} | Model: {OLLAMA_MODEL}</p>
  </header>

  <section>
    <h2>Alert Summary</h2>
    <p>Total={total}, Critical={critical_count}, APM={apm_count}, LOG={log_count}</p>
    <table>
      <thead><tr><th>Time</th><th>Type</th><th>Source</th><th>Severity</th><th>Fault</th><th>Root Cause</th><th>Action</th></tr></thead>
      <tbody>{alert_rows}</tbody>
    </table>
  </section>

  {eval_section}
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(html)

    print(f"[Report] HTML report saved: {output_path}")
    return output_path
