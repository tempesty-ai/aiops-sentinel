import json
from pathlib import Path

from eval.eval_suite import EvalReport, AIQualityEvaluator, evaluate_quality_gate, save_eval_report_json


def test_quality_gate_fails_when_scores_are_low():
    report = EvalReport(
        apm_results=[
            {"fault_type_correct": False, "hallucination_score": 0.2, "relevancy_score": 0.2, "faithfulness_score": 0.2}
        ],
        log_results=[
            {"error_type_correct": False, "hallucination_score": 0.2, "relevancy_score": 0.2, "faithfulness_score": 0.2}
        ],
        overall_score=0.2,
    )
    gate = evaluate_quality_gate(report)
    assert gate.passed is False
    assert "overall_score" in gate.failed_rules


def test_save_eval_report_has_schema_and_gate(tmp_path):
    report = EvalReport(apm_results=[], log_results=[], overall_score=0.8)
    report.quality_gate = evaluate_quality_gate(report)
    out = tmp_path / "eval_result.json"
    data = save_eval_report_json(report, output_path=str(out))
    assert data["schema_version"] == "1.1.0"
    assert "quality_gate" in data
    assert out.exists()


def test_eval_smoke_with_mocks_and_regression_fixture():
    fixture_path = Path("tests/fixtures/eval_regression_baseline.json")
    baseline = json.loads(fixture_path.read_text(encoding="utf-8"))

    evaluator = AIQualityEvaluator()

    evaluator.run_apm_eval = lambda: [
        {
            "scenario": "mock_apm",
            "fault_type": "CPU",
            "fault_type_correct": True,
            "keyword_score": 0.8,
            "completeness_score": 0.9,
            "custom_score": 0.85,
            "hallucination_score": 0.9,
            "relevancy_score": 0.9,
            "faithfulness_score": 0.9,
            "overall_score": 0.85,
            "severity": "warning",
        }
    ]
    evaluator.run_log_eval = lambda: [
        {
            "scenario": "mock_log",
            "error_type": "Network",
            "error_type_correct": True,
            "severity": "medium",
            "action_present": True,
            "custom_score": 0.8,
            "hallucination_score": 0.9,
            "relevancy_score": 0.8,
            "faithfulness_score": 0.8,
            "overall_score": 0.8,
        }
    ]

    report = evaluator.run_full_eval()
    assert report.overall_score >= baseline["minimum_overall_score"]
    assert report.quality_gate is not None
    assert report.quality_gate.metrics["apm_fault_type_accuracy"] >= baseline["minimum_apm_accuracy"]
    assert report.quality_gate.metrics["log_error_type_accuracy"] >= baseline["minimum_log_accuracy"]
