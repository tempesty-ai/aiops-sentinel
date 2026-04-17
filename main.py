"""
AIOps Sentinel - 메인 실행 진입점

실행 모드:
  python main.py          → 실시간 모니터링 (APM + 로그 감시)
  python main.py --eval   → AI 품질 평가만 실행
  python main.py --report → 누적 알람 HTML 리포트 생성
"""
import sys
import io

# Windows 한글 출력 인코딩 강제 설정
if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import time
import argparse
import threading
from datetime import datetime

from apm.mock_generator import MockAPMGenerator
from apm.anomaly_detector import AnomalyDetector
from apm.ai_analyzer import APMAIAnalyzer
from logwatch.log_simulator import LogSimulator
from logwatch.log_tailer import LogTailer, LogErrorEvent
from logwatch.ai_classifier import LogAIClassifier
from alert.mattermost import send_apm_alert, send_log_alert
from eval.eval_suite import AIQualityEvaluator, save_eval_report_json
from eval.report_generator import generate_html_report, AlertRecord
from config.settings import APM_CHECK_INTERVAL_SECONDS


# 전역 알람 이력 (리포트 생성용)
_alert_records: list[AlertRecord] = []
_alert_lock = threading.Lock()


def _handle_log_error(event: LogErrorEvent):
    """로그 에러 감지 시 호출되는 콜백"""
    print(f"\n[LogWatch] ERROR 감지: {event.module_name} - {event.keyword_matched}")

    classifier = LogAIClassifier()
    classification = classifier.classify(event.context_for_ai)

    print(f"  → AI 분류: {classification.error_type} | 심각도: {classification.severity}")

    # Mattermost 알람 전송
    send_log_alert(event, classification)

    # 이력 저장
    with _alert_lock:
        _alert_records.append(AlertRecord(
            timestamp=event.timestamp,
            alert_type="LOG",
            source=event.module_name,
            severity=classification.severity,
            fault_type=classification.error_type,
            root_cause=classification.recommended_action,
            action=classification.recommended_action,
            raw_ai_response=classification.raw_response,
        ))


def run_monitoring():
    """실시간 APM + 로그 모니터링"""
    print("=" * 60)
    print("  AIOps Sentinel - 실시간 모니터링 시작")
    print("=" * 60)
    print(f"  APM 체크 주기: {APM_CHECK_INTERVAL_SECONDS}초")
    print(f"  종료: Ctrl+C")
    print("=" * 60)

    # 로그 시뮬레이터 시작 (mock)
    log_simulator = LogSimulator(base_dir=".")
    log_simulator.start()

    # 로그 감시 시작
    log_tailer = LogTailer(base_dir=".", on_error=_handle_log_error)
    log_tailer.start()

    # APM 모니터링
    apm_generator = MockAPMGenerator()
    apm_detector = AnomalyDetector()
    apm_analyzer = APMAIAnalyzer()

    try:
        while True:
            snapshots = apm_generator.get_all_snapshots()

            for snapshot in snapshots:
                result = apm_detector.analyze(snapshot)

                if result.is_anomaly:
                    print(f"\n[APM] 이상 감지: {snapshot.server} | {result.severity}")
                    for rule in result.triggered_rules:
                        print(f"  ↳ {rule}")

                    # AI 분석
                    analysis = apm_analyzer.analyze(result.context_for_ai)
                    print(f"  → AI 분석: {analysis.fault_type} | {analysis.severity}")

                    # Mattermost 알람
                    send_apm_alert(result, analysis)

                    # 이력 저장
                    with _alert_lock:
                        _alert_records.append(AlertRecord(
                            timestamp=snapshot.timestamp,
                            alert_type="APM",
                            source=snapshot.server,
                            severity=result.severity,
                            fault_type=analysis.fault_type,
                            root_cause=analysis.root_cause,
                            action=analysis.immediate_action,
                            raw_ai_response=analysis.raw_response,
                        ))

            time.sleep(APM_CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\n[Sentinel] 모니터링 종료...")
        log_simulator.stop()
        log_tailer.stop()

        # 종료 시 리포트 자동 생성
        if _alert_records:
            print(f"[Sentinel] 총 {len(_alert_records)}건 감지 → 리포트 생성 중...")
            path = generate_html_report(_alert_records)
            print(f"[Sentinel] 리포트: {path}")


def run_eval():
    """AI 품질 평가 실행"""
    print("=" * 60)
    print("  AIOps Sentinel - AI 품질 평가")
    print("=" * 60)

    evaluator = AIQualityEvaluator()
    report = evaluator.run_full_eval()

    eval_data = save_eval_report_json(report)
    generate_html_report([], eval_data=eval_data, output_path="reports/eval_report.html")

    print(f"\n결과:")
    print(f"  전체 품질 점수: {report.overall_score:.0%}")
    for r in report.apm_results:
        print(f"  APM [{r['scenario']}]: {r['overall_score']:.0%}")
    for r in report.log_results:
        print(f"  LOG [{r['scenario']}]: {r['overall_score']:.0%}")


def run_report():
    """저장된 이력으로 리포트 생성"""
    import json, os
    eval_data = None
    if os.path.exists("reports/eval_result.json"):
        with open("reports/eval_result.json", encoding="utf-8") as f:
            eval_data = json.load(f)

    path = generate_html_report(_alert_records, eval_data=eval_data)
    print(f"리포트 생성: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIOps Sentinel")
    parser.add_argument("--eval", action="store_true", help="AI 품질 평가 실행")
    parser.add_argument("--report", action="store_true", help="HTML 리포트 생성")
    args = parser.parse_args()

    if args.eval:
        run_eval()
    elif args.report:
        run_report()
    else:
        run_monitoring()
