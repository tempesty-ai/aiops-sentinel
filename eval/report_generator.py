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
        gate_label = str(gate.get("status") or ("PASS" if gate.get("passed") else "FAIL"))
        gate_class = {"PASS": "pass", "INCONCLUSIVE": "warn"}.get(gate_label, "fail")
        failed_rules = gate.get("failed_rules", [])
        failed_text = ", ".join(failed_rules) if failed_rules else "-"
        unmeasured_rules = gate.get("unmeasured_rules", [])
        unmeasured_text = ", ".join(unmeasured_rules) if unmeasured_rules else "-"
        thin_rules = gate.get("insufficient_sample_rules", [])
        thin_text = ", ".join(thin_rules) if thin_rules else "-"
        metrics = gate.get("metrics", {})
        sample_counts = gate.get("sample_counts", {})

        def metric_text(key: str) -> str:
            value = metrics.get(key)
            if value is None:
                return "not measured"
            return f"{value} (n={sample_counts.get(key, '?')})"

        meta = eval_data.get("run_metadata", {})
        meta_bits = [
            f"dataset={meta.get('dataset_version', 'n/a')}",
            f"analysis={meta.get('analysis_model', 'n/a')}",
            f"judge={meta.get('judge_model', 'n/a')}",
            f"ollama={meta.get('ollama_version', 'n/a')}",
        ]
        if meta.get("self_grading"):
            meta_bits.append("SELF-GRADING")
        meta_text = " | ".join(meta_bits)

        def judge_block(r: dict) -> str:
            """Per-metric judge score together with the reason the judge gave."""
            parts = []
            for name, score_key, reason_key in (
                ("Relevancy", "relevancy_score", "relevancy_reason"),
                ("Faithfulness", "faithfulness_score", "faithfulness_reason"),
            ):
                score = r.get(score_key, -1)
                if not isinstance(score, (int, float)) or score < 0:
                    parts.append(
                        f"<div class=\"judge\"><span class=\"jname\">{name}</span>"
                        f"<span class=\"jscore muted\">not measured</span></div>"
                    )
                    continue
                reason = _escape(str(r.get(reason_key) or "(no reason returned)"))
                parts.append(
                    f"<div class=\"judge\"><span class=\"jname\">{name}</span>"
                    f"<span class=\"jscore\">{score}</span>"
                    f"<span class=\"jreason\">{reason}</span></div>"
                )
            return "".join(parts)

        def grounding_block(r: dict) -> str:
            rows = ""
            for label, score_key, unsupported_key, reason_key in (
                ("Cause grounding", "cause_grounding_score",
                 "cause_grounding_unsupported", "cause_grounding_reason"),
                ("Action grounding", "action_grounding_score",
                 "action_grounding_unsupported", "action_grounding_reason"),
            ):
                score = r.get(score_key)
                if score is None:
                    continue
                cls = "jreason ungrounded" if (r.get(unsupported_key) or []) else "jreason"
                rows += (
                    f'<div class="judge"><span class="jname">{label}</span>'
                    f'<span class="jscore">{score}</span>'
                    f'<span class="{cls}">{_escape(str(r.get(reason_key) or ""))}</span></div>'
                )
            return rows

        def answer_block(r: dict) -> str:
            excerpt = r.get("answer_excerpt")
            if not excerpt:
                return ""
            return f"<details><summary>AI answer excerpt</summary><pre>{_escape(str(excerpt))}</pre></details>"

        apm_rows = ""
        for r in eval_data.get("apm_eval", []):
            apm_rows += (
                "<tr>"
                f"<td>{_escape(str(r.get('scenario', '')))}</td>"
                f"<td>{_escape(str(r.get('fault_type', '')))}</td>"
                f"<td>{'-' if r.get('classification_unmeasured') else ('Y' if r.get('fault_type_correct') else 'N')}</td>"
                f"<td>{int(float(r.get('overall_score', 0.0)) * 100)}%</td>"
                "</tr>"
                f'<tr class="detail"><td colspan="4">{grounding_block(r)}{judge_block(r)}{answer_block(r)}</td></tr>'
            )
        if not apm_rows:
            apm_rows = '<tr><td colspan="4" class="muted">No APM eval rows</td></tr>'

        log_rows = ""
        for r in eval_data.get("log_eval", []):
            log_rows += (
                "<tr>"
                f"<td>{_escape(str(r.get('scenario', '')))}</td>"
                f"<td>{_escape(str(r.get('error_type', '')))}</td>"
                f"<td>{'-' if r.get('classification_unmeasured') else ('Y' if r.get('error_type_correct') else 'N')}</td>"
                f"<td>{int(float(r.get('overall_score', 0.0)) * 100)}%</td>"
                "</tr>"
                f'<tr class="detail"><td colspan="4">{judge_block(r)}{answer_block(r)}</td></tr>'
            )
        if not log_rows:
            log_rows = '<tr><td colspan="4" class="muted">No log eval rows</td></tr>'

        halluc = eval_data.get("hallucination") or {}
        if halluc.get("rate") is None:
            halluc_line = "not measured"
        else:
            axis = halluc.get("by_axis", {})
            common = ", ".join(f"{k} x{v}" for k, v in (halluc.get("most_common_unsupported") or {}).items())
            halluc_line = (
                f"{halluc['rate']:.0%} - {halluc['cases_with_hallucination']} of "
                f"{halluc['cases_measurable']} cases asserted something the metrics do not support "
                f"(diagnosis {axis.get('cause', 0)}, action {axis.get('action', 0)})"
                + (f". Most often: {common}" if common else "")
            )

        scenario_rows = ""
        for key, entry in (eval_data.get("by_scenario") or {}).items():
            acc = entry.get("accuracy")
            acc_text = "not measured" if acc is None else f"{int(acc * 100)}%"
            variants = entry.get("by_variant") or {}
            variant_text = ", ".join(
                f"{name} {int(v['accuracy'] * 100)}%" if v.get("accuracy") is not None else f"{name} -"
                for name, v in sorted(variants.items())
            ) or "-"
            weak = ' class="fail"' if acc is not None and acc < 0.7 else ""
            scenario_rows += (
                f"<tr{weak}>"
                f"<td>{_escape(str(key))}</td>"
                f"<td>{entry.get('cases', 0)}</td>"
                f"<td>{acc_text}</td>"
                f"<td>{_escape(variant_text)}</td>"
                "</tr>"
            )
        if not scenario_rows:
            scenario_rows = '<tr><td colspan="4" class="muted">No per-scenario data</td></tr>'

        eval_section = f"""
        <section>
            <h2>AI Evaluation</h2>
            <div class="cards">
                <div class="card"><div class="value">{overall}%</div><div class="label">Overall Score</div></div>
                <div class="card {gate_class}"><div class="value">{gate_label}</div><div class="label">Quality Gate</div></div>
            </div>
            <p class="muted">schema_version: {schema_version}</p>
            <p class="muted">{_escape(meta_text)}</p>
            <p><strong>Hallucination rate:</strong> {_escape(halluc_line)}</p>
            <p class="muted">derived from cause/action grounding - reported, not gated</p>
            <p><strong>Failed Rules:</strong> {_escape(failed_text)}</p>
            <p><strong>Unmeasured:</strong> {_escape(unmeasured_text)}</p>
            <p><strong>Thin sample:</strong> {_escape(thin_text)}</p>
            <p class="muted">
                Metrics:
                overall={_escape(metric_text('overall_score'))},
                apm_acc={_escape(metric_text('apm_fault_type_accuracy'))},
                log_acc={_escape(metric_text('log_error_type_accuracy'))},
                relevancy={_escape(metric_text('relevancy'))},
                faithfulness={_escape(metric_text('faithfulness'))},
                cause_grounding={_escape(metric_text('cause_grounding'))},
                action_grounding={_escape(metric_text('action_grounding'))}
            </p>

            <h3>By Scenario</h3>
            <p class="muted">가장 약한 유형부터. 문구별 정확도가 갈리면 표현에 취약한 것입니다.</p>
            <table>
                <thead><tr><th>Scenario</th><th>Cases</th><th>Accuracy</th><th>By phrasing</th></tr></thead>
                <tbody>{scenario_rows}</tbody>
            </table>

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
    .card.warn {{ border-color: #f59e0b; }}
    tr.fail td {{ color: #fca5a5; }}
    tr.detail td {{ background: #172033; border-top: none; padding: 8px 12px 12px; }}
    .judge {{ display: grid; grid-template-columns: 110px 52px 1fr; gap: 8px; align-items: start; padding: 3px 0; font-size: 12px; }}
    .jname {{ color: #93c5fd; }}
    .jscore {{ color: #e2e8f0; font-variant-numeric: tabular-nums; }}
    .jreason {{ color: #94a3b8; line-height: 1.5; }}
    .jreason.ungrounded {{ color: #fca5a5; }}
    tr.detail details {{ margin-top: 6px; font-size: 12px; color: #94a3b8; }}
    tr.detail pre {{ white-space: pre-wrap; margin: 6px 0 0; padding: 8px; background: #0f172a; border-radius: 6px; overflow-x: auto; }}
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
