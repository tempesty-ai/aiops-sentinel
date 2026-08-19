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
    assert data["schema_version"] == "1.2.0"
    assert "quality_gate" in data
    assert out.exists()
    meta = data["run_metadata"]
    assert meta["dataset_version"]
    assert meta["analysis_model"]
    assert meta["judge_model"]


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
    assert report.dataset_version == baseline["dataset_version"]


def test_missing_judge_scores_are_inconclusive_not_failed():
    """A judge outage must not be reported as a quality failure."""
    case = {
        "fault_type_correct": True,
        "hallucination_score": 0.9,
        "relevancy_score": -1.0,      # judge returned invalid JSON
        "faithfulness_score": -1.0,   # judge returned invalid JSON
    }
    report = EvalReport(apm_results=[case], log_results=[dict(case, error_type_correct=True)], overall_score=0.9)
    gate = evaluate_quality_gate(report)

    assert gate.status == "INCONCLUSIVE"
    assert gate.passed is False
    assert "faithfulness" not in gate.failed_rules
    assert "relevancy" not in gate.failed_rules
    assert set(gate.unmeasured_rules) == {"relevancy", "faithfulness"}
    assert gate.metrics["faithfulness"] is None
    assert gate.sample_counts["hallucination"] == 2


def test_unmeasurable_classification_is_not_counted_as_correct():
    """A fallback answer carries no evidence, so it must not inflate accuracy."""
    measured_wrong = {"fault_type_correct": False, "classification_unmeasured": False}
    fallback = {"fault_type_correct": True, "classification_unmeasured": True}
    report = EvalReport(apm_results=[measured_wrong, fallback], log_results=[], overall_score=0.5)
    gate = evaluate_quality_gate(report)

    # only the one measurable case counts, and it was wrong
    assert gate.sample_counts["apm_fault_type_accuracy"] == 1
    assert gate.metrics["apm_fault_type_accuracy"] == 0.0
    assert "apm_fault_type_accuracy" in gate.failed_rules


def test_all_metrics_measured_and_above_threshold_passes():
    case = {
        "fault_type_correct": True,
        "error_type_correct": True,
        "hallucination_score": 0.9,
        "relevancy_score": 0.8,
        "faithfulness_score": 0.8,
    }
    report = EvalReport(apm_results=[case], log_results=[case], overall_score=0.85)
    gate = evaluate_quality_gate(report)

    assert gate.status == "PASS"
    assert gate.passed is True
    assert gate.unmeasured_rules == []
    assert gate.score == 1.0


def test_thin_sample_cannot_certify_a_pass():
    """One surviving judge call must not certify a metric as passing."""
    cases = [
        {"fault_type_correct": True, "hallucination_score": 0.9, "relevancy_score": 0.9},
        {"fault_type_correct": True, "hallucination_score": 0.9, "relevancy_score": -1.0},
        {"fault_type_correct": True, "hallucination_score": 0.9, "relevancy_score": -1.0},
        {"fault_type_correct": True, "hallucination_score": 0.9, "relevancy_score": -1.0},
        {"fault_type_correct": True, "hallucination_score": 0.9, "relevancy_score": -1.0},
    ]
    report = EvalReport(apm_results=cases, log_results=[], overall_score=0.9)
    gate = evaluate_quality_gate(report)

    # relevancy is above threshold but rests on 1 of 5 cases
    assert gate.metrics["relevancy"] == 0.9
    assert gate.coverage["relevancy"] == 0.2
    assert "relevancy" in gate.insufficient_sample_rules
    assert "relevancy" not in gate.failed_rules
    assert gate.status == "INCONCLUSIVE"

    # hallucination covers every case, so it still certifies
    assert gate.coverage["hallucination"] == 1.0
    assert "hallucination" not in gate.insufficient_sample_rules


def test_below_threshold_still_fails_even_on_thin_sample():
    """Thin data is not an excuse to downgrade a real below-threshold reading."""
    cases = [
        {"fault_type_correct": True, "hallucination_score": 0.1},
        {"fault_type_correct": True, "hallucination_score": -1.0},
        {"fault_type_correct": True, "hallucination_score": -1.0},
    ]
    report = EvalReport(apm_results=cases, log_results=[], overall_score=0.9)
    gate = evaluate_quality_gate(report)

    assert "hallucination" in gate.failed_rules
    assert "hallucination" not in gate.insufficient_sample_rules
    assert gate.status == "FAIL"


def test_fallback_answers_are_excluded_from_judge_metrics():
    """Judging fallback boilerplate measures nothing about the model."""
    cases = [
        {"fault_type_correct": True, "hallucination_score": 0.9},
        {"fault_type_correct": False, "llm_failed": True, "classification_unmeasured": True, "hallucination_score": 0.9},
    ]
    report = EvalReport(apm_results=cases, log_results=[], overall_score=0.9)
    gate = evaluate_quality_gate(report)

    assert gate.sample_counts["hallucination"] == 1
    assert gate.sample_counts["apm_fault_type_accuracy"] == 1

