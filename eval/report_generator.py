"""
HTML 리포트 생성기
실시간 감지 이력 + AI Eval 결과를 하나의 HTML 파일로 생성
"""
import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from config.settings import OLLAMA_MODEL


@dataclass
class AlertRecord:
    timestamp: str
    alert_type: str      # "APM" / "LOG"
    source: str          # 서버명 or 모듈명
    severity: str
    fault_type: str
    root_cause: str
    action: str
    raw_ai_response: str


def generate_html_report(
    alert_records: list[AlertRecord],
    eval_data: dict | None = None,
    output_path: str = "reports/aiops_report.html"
) -> str:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    severity_badge = {
        "심각": '<span class="badge badge-critical">심각</span>',
        "경고": '<span class="badge badge-warning">경고</span>',
        "높음": '<span class="badge badge-critical">높음</span>',
        "중간": '<span class="badge badge-warning">중간</span>',
        "낮음": '<span class="badge badge-low">낮음</span>',
        "주의": '<span class="badge badge-low">주의</span>',
    }

    # 통계 집계
    total = len(alert_records)
    apm_count = sum(1 for r in alert_records if r.alert_type == "APM")
    log_count = sum(1 for r in alert_records if r.alert_type == "LOG")
    critical_count = sum(1 for r in alert_records if r.severity in ["심각", "높음"])

    # 알람 테이블 행 생성
    rows = ""
    for r in sorted(alert_records, key=lambda x: x.timestamp, reverse=True):
        badge = severity_badge.get(r.severity, f'<span class="badge">{r.severity}</span>')
        type_icon = "📊" if r.alert_type == "APM" else "📄"
        safe_cause = r.root_cause.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        safe_action = r.action.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        rows += f"""
        <tr>
            <td>{r.timestamp}</td>
            <td>{type_icon} {r.alert_type}</td>
            <td><strong>{r.source}</strong></td>
            <td>{badge}</td>
            <td>{r.fault_type}</td>
            <td class="text-small">{safe_cause}</td>
            <td class="text-small">{safe_action}</td>
        </tr>"""

    # Eval 결과 섹션
    eval_section = ""
    if eval_data:
        overall_pct = int(eval_data.get("overall_score", 0) * 100)
        apm_rows = ""
        for r in eval_data.get("apm_eval", []):
            correct = "✅" if r.get("fault_type_correct") else "❌"
            apm_rows += f"""
            <tr>
                <td>{r['scenario']}</td>
                <td>{r['fault_type']}</td>
                <td>{correct}</td>
                <td>{int(r['keyword_score']*100)}%</td>
                <td>{int(r['completeness_score']*100)}%</td>
                <td><strong>{int(r['overall_score']*100)}%</strong></td>
            </tr>"""

        log_rows = ""
        for r in eval_data.get("log_eval", []):
            correct = "✅" if r.get("error_type_correct") else "❌"
            log_rows += f"""
            <tr>
                <td>{r['scenario']}</td>
                <td>{r['error_type']}</td>
                <td>{correct}</td>
                <td>{r['severity']}</td>
                <td><strong>{int(r['overall_score']*100)}%</strong></td>
            </tr>"""

        # DeepEval 메트릭 평균 계산
        all_results = eval_data.get("apm_eval", []) + eval_data.get("log_eval", [])
        valid_h = [r["hallucination_score"] for r in all_results if r.get("hallucination_score", -1) >= 0]
        valid_r = [r["relevancy_score"] for r in all_results if r.get("relevancy_score", -1) >= 0]
        valid_f = [r["faithfulness_score"] for r in all_results if r.get("faithfulness_score", -1) >= 0]
        avg_h = int(sum(valid_h) / len(valid_h) * 100) if valid_h else "N/A"
        avg_r = int(sum(valid_r) / len(valid_r) * 100) if valid_r else "N/A"
        avg_f = int(sum(valid_f) / len(valid_f) * 100) if valid_f else "N/A"

        # APM 테이블 행 (DeepEval 메트릭 포함)
        apm_rows = ""
        for r in eval_data.get("apm_eval", []):
            correct = "✅" if r.get("fault_type_correct") else "❌"
            h = f"{int(r['hallucination_score']*100)}%" if r.get("hallucination_score", -1) >= 0 else "N/A"
            rv = f"{int(r['relevancy_score']*100)}%" if r.get("relevancy_score", -1) >= 0 else "N/A"
            fa = f"{int(r['faithfulness_score']*100)}%" if r.get("faithfulness_score", -1) >= 0 else "N/A"
            apm_rows += f"""
            <tr>
                <td>{r['scenario']}</td>
                <td>{r['fault_type']}</td>
                <td>{correct}</td>
                <td>{int(r['keyword_score']*100)}%</td>
                <td>{int(r['completeness_score']*100)}%</td>
                <td class="metric-cell">{h}</td>
                <td class="metric-cell">{rv}</td>
                <td class="metric-cell">{fa}</td>
                <td><strong>{int(r['overall_score']*100)}%</strong></td>
            </tr>"""

        # 로그 테이블 행 (DeepEval 메트릭 포함)
        log_rows = ""
        for r in eval_data.get("log_eval", []):
            correct = "✅" if r.get("error_type_correct") else "❌"
            h = f"{int(r['hallucination_score']*100)}%" if r.get("hallucination_score", -1) >= 0 else "N/A"
            rv = f"{int(r['relevancy_score']*100)}%" if r.get("relevancy_score", -1) >= 0 else "N/A"
            fa = f"{int(r['faithfulness_score']*100)}%" if r.get("faithfulness_score", -1) >= 0 else "N/A"
            log_rows += f"""
            <tr>
                <td>{r['scenario']}</td>
                <td>{r['error_type']}</td>
                <td>{correct}</td>
                <td>{r['severity']}</td>
                <td class="metric-cell">{h}</td>
                <td class="metric-cell">{rv}</td>
                <td class="metric-cell">{fa}</td>
                <td><strong>{int(r['overall_score']*100)}%</strong></td>
            </tr>"""

        eval_section = f"""
        <section class="section">
            <h2>🧪 AI 품질 평가 결과 (Eval Suite)</h2>

            <div style="display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap;">
                <div class="score-card">
                    <div class="score-value">{overall_pct}%</div>
                    <div class="score-label">전체 AI 품질 점수</div>
                </div>
                <div class="score-card">
                    <div class="score-value" style="font-size:2rem; color:#58a6ff;">{avg_h}%</div>
                    <div class="score-label">Hallucination 방지율</div>
                    <div style="font-size:0.75rem;color:#8b949e;margin-top:4px;">환각 없이 답변한 비율</div>
                </div>
                <div class="score-card">
                    <div class="score-value" style="font-size:2rem; color:#58a6ff;">{avg_r}%</div>
                    <div class="score-label">Answer Relevancy</div>
                    <div style="font-size:0.75rem;color:#8b949e;margin-top:4px;">질문과 관련성 있는 답변</div>
                </div>
                <div class="score-card">
                    <div class="score-value" style="font-size:2rem; color:#58a6ff;">{avg_f}%</div>
                    <div class="score-label">Faithfulness</div>
                    <div style="font-size:0.75rem;color:#8b949e;margin-top:4px;">데이터 기반 근거 있는 답변</div>
                </div>
            </div>

            <h3>APM 분석 AI</h3>
            <table>
                <thead>
                    <tr>
                        <th>시나리오</th><th>분류된 장애유형</th><th>정확도</th>
                        <th>키워드</th><th>완결성</th>
                        <th>Hallucination↑</th><th>Relevancy↑</th><th>Faithfulness↑</th>
                        <th>종합</th>
                    </tr>
                </thead>
                <tbody>{apm_rows}</tbody>
            </table>

            <h3>로그 분류 AI</h3>
            <table>
                <thead>
                    <tr>
                        <th>시나리오</th><th>분류된 에러유형</th><th>정확도</th><th>심각도</th>
                        <th>Hallucination↑</th><th>Relevancy↑</th><th>Faithfulness↑</th>
                        <th>종합</th>
                    </tr>
                </thead>
                <tbody>{log_rows}</tbody>
            </table>
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIOps Sentinel - 모니터링 리포트</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; }}
        header {{ background: #161b22; padding: 20px 40px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }}
        header h1 {{ font-size: 1.4rem; color: #58a6ff; }}
        header .meta {{ font-size: 0.85rem; color: #8b949e; }}
        .stats {{ display: flex; gap: 16px; padding: 24px 40px; background: #161b22; border-bottom: 1px solid #30363d; }}
        .stat-card {{ background: #21262d; border-radius: 8px; padding: 16px 24px; flex: 1; text-align: center; border: 1px solid #30363d; }}
        .stat-card .value {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
        .stat-card .label {{ font-size: 0.8rem; color: #8b949e; margin-top: 4px; }}
        .stat-card.critical .value {{ color: #f85149; }}
        .section {{ padding: 24px 40px; }}
        .section h2 {{ font-size: 1.1rem; color: #e6edf3; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #30363d; }}
        .section h3 {{ font-size: 0.95rem; color: #8b949e; margin: 20px 0 12px 0; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        th {{ background: #21262d; padding: 10px 12px; text-align: left; color: #8b949e; font-weight: 600; border-bottom: 1px solid #30363d; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; vertical-align: top; }}
        tr:hover td {{ background: #161b22; }}
        .text-small {{ font-size: 0.78rem; color: #8b949e; max-width: 300px; }}
        .badge {{ padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
        .badge-critical {{ background: #3d1a1a; color: #f85149; border: 1px solid #f85149; }}
        .badge-warning {{ background: #3d2a1a; color: #d29922; border: 1px solid #d29922; }}
        .badge-low {{ background: #1a2a3d; color: #58a6ff; border: 1px solid #58a6ff; }}
        .score-card {{ text-align: center; padding: 24px; background: #21262d; border-radius: 12px; display: inline-block; margin-bottom: 20px; border: 1px solid #30363d; }}
        .score-value {{ font-size: 3rem; font-weight: bold; color: #3fb950; }}
        .score-label {{ color: #8b949e; margin-top: 4px; }}
        .metric-cell {{ color: #3fb950; font-weight: 600; }}
        .metric-cell:contains("N/A") {{ color: #8b949e; }}
    </style>
</head>
<body>
    <header>
        <h1>🛡️ AIOps Sentinel — 모니터링 리포트</h1>
        <div class="meta">생성: {now} | Model: {OLLAMA_MODEL} (Ollama)</div>
    </header>

    <div class="stats">
        <div class="stat-card">
            <div class="value">{total}</div>
            <div class="label">전체 알람</div>
        </div>
        <div class="stat-card critical">
            <div class="value">{critical_count}</div>
            <div class="label">심각/높음</div>
        </div>
        <div class="stat-card">
            <div class="value">{apm_count}</div>
            <div class="label">APM 장애</div>
        </div>
        <div class="stat-card">
            <div class="value">{log_count}</div>
            <div class="label">로그 에러</div>
        </div>
    </div>

    <section class="section">
        <h2>📋 감지 이력</h2>
        <table>
            <thead>
                <tr>
                    <th>시각</th><th>유형</th><th>서버/모듈</th><th>심각도</th>
                    <th>장애유형</th><th>AI 원인 분석</th><th>권고 조치</th>
                </tr>
            </thead>
            <tbody>
                {rows if rows else '<tr><td colspan="7" style="text-align:center;color:#8b949e;">감지된 이벤트 없음</td></tr>'}
            </tbody>
        </table>
    </section>

    {eval_section}

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[Report] HTML 리포트 저장: {output_path}")
    return output_path
