"""
AIOps Sentinel entrypoint

Usage:
  py -3 main.py
  py -3 main.py --eval
  py -3 main.py --eval --report
  py -3 main.py --eval --gate
  py -3 main.py --eval --sample 20

Exit codes: 0 pass, 2 quality gate failed, 3 config error, 4 gate inconclusive.
"""
import argparse
import io
import os
import sys
import threading
import time

from alert.mattermost import send_apm_alert, send_log_alert
from apm.ai_analyzer import APMAIAnalyzer
from apm.anomaly_detector import AnomalyDetector
from apm.mock_generator import MockAPMGenerator
from config.settings import APM_CHECK_INTERVAL_SECONDS, validate_required_settings
from eval.eval_suite import AIQualityEvaluator, save_eval_report_json
from eval.report_generator import AlertRecord, generate_html_report
from logwatch.ai_classifier import LogAIClassifier
from logwatch.log_simulator import LogSimulator
from logwatch.log_tailer import LogErrorEvent, LogTailer

if hasattr(sys.stdout, "buffer") and (sys.stdout.encoding or "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

_alert_records: list[AlertRecord] = []
_alert_lock = threading.Lock()


def _handle_log_error(event: LogErrorEvent):
    print(f"\n[LogWatch] Error detected: {event.module_name} - {event.keyword_matched}")

    classifier = LogAIClassifier()
    classification = classifier.classify(event.context_for_ai)
    print(f"  -> AI classification: {classification.error_type} | severity={classification.severity}")

    send_log_alert(event, classification)
    with _alert_lock:
        _alert_records.append(
            AlertRecord(
                timestamp=event.timestamp,
                alert_type="LOG",
                source=event.module_name,
                severity=classification.severity,
                fault_type=classification.error_type,
                root_cause=classification.recommended_action,
                action=classification.recommended_action,
                raw_ai_response=classification.raw_response,
            )
        )


def run_monitoring() -> int:
    validate_required_settings(require_mattermost=False)

    print("=" * 60)
    print("AIOps Sentinel - Monitoring mode")
    print("=" * 60)
    print(f"APM interval: {APM_CHECK_INTERVAL_SECONDS}s")
    print("Stop with Ctrl+C")

    log_simulator = LogSimulator(base_dir=".")
    log_simulator.start()

    log_tailer = LogTailer(base_dir=".", on_error=_handle_log_error)
    log_tailer.start()

    apm_generator = MockAPMGenerator()
    apm_detector = AnomalyDetector()
    apm_analyzer = APMAIAnalyzer()

    try:
        while True:
            snapshots = apm_generator.get_all_snapshots()
            for snapshot in snapshots:
                result = apm_detector.analyze(snapshot)
                if not result.is_anomaly:
                    continue

                print(f"\n[APM] anomaly: {snapshot.server} | severity={result.severity}")
                for rule in result.triggered_rules:
                    print(f"  - {rule}")

                analysis = apm_analyzer.analyze(result.context_for_ai)
                print(f"  -> AI analysis: {analysis.fault_type} | severity={analysis.severity}")

                send_apm_alert(result, analysis)
                with _alert_lock:
                    _alert_records.append(
                        AlertRecord(
                            timestamp=snapshot.timestamp,
                            alert_type="APM",
                            source=snapshot.server,
                            severity=result.severity,
                            fault_type=analysis.fault_type,
                            root_cause=analysis.root_cause,
                            action=analysis.immediate_action,
                            raw_ai_response=analysis.raw_response,
                        )
                    )
            time.sleep(APM_CHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[Sentinel] Stopping monitoring...")
        log_simulator.stop()
        log_tailer.stop()
        if _alert_records:
            path = generate_html_report(_alert_records)
            print(f"[Sentinel] Saved runtime report: {path}")
        return 0


def run_eval(generate_report: bool = False, enforce_gate: bool = False,
             sample: int = 0, sample_seed: int = 0) -> int:
    validate_required_settings(require_mattermost=False)

    print("=" * 60)
    print("AIOps Sentinel - Evaluation mode")
    print("=" * 60)

    evaluator = AIQualityEvaluator(sample_size=sample, sample_seed=sample_seed)
    report = evaluator.run_full_eval()
    eval_data = save_eval_report_json(report)

    if generate_report:
        path = generate_html_report([], eval_data=eval_data, output_path="reports/eval_report.html")
        print(f"[Eval] Saved HTML report: {path}")

    print("\n[Eval] Summary")
    print(f"  overall_score: {report.overall_score:.0%}")
    evaluated = len(report.apm_results) + len(report.log_results)
    print(f"  dataset: {report.dataset_version} | prompt: {report.prompt_version}"
          f" | analysis: {report.analysis_model} | judge: {report.judge_model}")
    print(f"  cases: {evaluated}/{report.cases_available}"
          + (f" (sampled, seed {report.sample_seed})" if evaluated < report.cases_available else ""))
    if report.quality_gate:
        gate = report.quality_gate
        print(f"  quality_gate: {gate.status} ({gate.score:.0%} of checks certified)")
        if gate.failed_rules:
            print(f"  failed_rules: {', '.join(gate.failed_rules)}")
        if gate.unmeasured_rules:
            print(f"  unmeasured_rules: {', '.join(gate.unmeasured_rules)}")
        if gate.insufficient_sample_rules:
            thin = ", ".join(f"{r} (coverage {gate.coverage[r]:.0%})" for r in gate.insufficient_sample_rules)
            print(f"  insufficient_sample_rules: {thin}")
        if gate.status == "INCONCLUSIVE":
            print("  -> INCONCLUSIVE means the metrics could not be certified, not that quality was bad.")

    if enforce_gate and report.quality_gate and not report.quality_gate.passed:
        # 2 = a measured metric is below threshold, 4 = could not measure at all
        return 2 if report.quality_gate.failed_rules else 4
    return 0


def run_report() -> int:
    import json

    eval_data = None
    eval_path = "reports/eval_result.json"
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as fp:
            eval_data = json.load(fp)
    path = generate_html_report(_alert_records, eval_data=eval_data)
    print(f"Report generated: {path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIOps Sentinel")
    parser.add_argument("--eval", action="store_true", help="Run AI evaluation suite")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--gate", action="store_true", help="Fail process when quality gate does not pass")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Evaluate a stratified subset of N cases instead of the whole dataset (0 = all). "
             "A full run costs about 4 LLM calls per case.",
    )
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for --sample, so a subset is reproducible")
    args = parser.parse_args()

    try:
        if args.eval:
            raise_code = run_eval(
                generate_report=args.report,
                enforce_gate=args.gate,
                sample=args.sample,
                sample_seed=args.sample_seed,
            )
        elif args.report:
            raise_code = run_report()
        else:
            raise_code = run_monitoring()
        raise SystemExit(raise_code)
    except ValueError as exc:
        print(f"[ConfigError] {exc}")
        raise SystemExit(3)
