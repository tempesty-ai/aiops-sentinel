"""Report layout: what a reader of a 102-case run needs to find first."""
import json
import re

from eval.report_generator import AlertRecord, generate_html_report

GATE = {
    "status": "FAIL",
    "failed_rules": ["cause_grounding"],
    "unmeasured_rules": ["faithfulness"],
    "insufficient_sample_rules": ["relevancy"],
    "metrics": {"overall_score": 0.78, "cause_grounding": 0.61, "faithfulness": None, "relevancy": 0.9},
    "sample_counts": {"overall_score": 102, "cause_grounding": 57, "faithfulness": 0, "relevancy": 4},
    "coverage": {"overall_score": 1.0, "cause_grounding": 0.95, "faithfulness": 0.0, "relevancy": 0.1},
    "thresholds": {"overall_score_min": 0.7, "cause_grounding_min": 0.8,
                   "faithfulness_min": 0.6, "relevancy_min": 0.6},
}

EVAL = {
    "schema_version": "1.2.0",
    "timestamp": "2026-08-25 19:49:45",
    "overall_score": 0.78,
    "run_metadata": {"dataset_version": "2.2.0", "prompt_version": "v1", "analysis_model": "8b",
                     "judge_model": "8b", "judge_mode": "domain", "ollama_version": "0.32.9",
                     "self_grading": True, "cases_evaluated": 102, "cases_available": 102},
    "quality_gate": GATE,
    "hallucination": {"rate": 0.65, "cases_with_hallucination": 39, "cases_measurable": 60,
                      "by_axis": {"cause": 37, "action": 25},
                      "most_common_unsupported": {"memory": 61, "cpu": 59}},
    "by_scenario": {"malformed_log": {"cases": 3, "accuracy": 0.0,
                                      "by_variant": {"v1": {"accuracy": 0.0}, "v2": {"accuracy": 0.0}}}},
    "apm_eval": [
        {"case_id": "apm-050", "scenario": "good", "fault_type": "CPU", "fault_type_correct": True,
         "overall_score": 1.0, "cause_grounding_score": 1.0, "action_grounding_score": 1.0,
         "relevancy_score": 0.9, "faithfulness_score": 0.9, "answer_excerpt": "fine"},
        {"case_id": "apm-002", "scenario": "worst", "fault_type": "Memory", "fault_type_correct": False,
         "overall_score": 0.53, "cause_grounding_score": 0.0,
         "cause_grounding_unsupported": ["cpu"], "cause_grounding_reason": "points at cpu",
         "action_grounding_score": None, "relevancy_score": -1.0, "faithfulness_score": 0.4,
         "answer_excerpt": "bad"},
    ],
    "log_eval": [],
}


def _render(tmp_path, alerts=None, data=EVAL):
    out = tmp_path / "r.html"
    generate_html_report(alerts or [], eval_data=data, output_path=str(out))
    return out.read_text(encoding="utf-8")


def test_alert_section_is_omitted_when_there_are_no_alerts(tmp_path):
    """An eval-only run has none; a table of zeroes just pushes findings down."""
    assert "Alerts</h2>" not in _render(tmp_path)


def test_alert_section_appears_when_alerts_exist(tmp_path):
    record = AlertRecord("2026-01-01 00:00:00", "APM", "was_1", "심각", "CPU", "cause", "action", "raw")
    assert "Alerts</h2>" in _render(tmp_path, alerts=[record])


def test_worst_case_is_listed_first(tmp_path):
    html = _render(tmp_path)
    assert html.index("apm-002") < html.index("apm-050")


def test_every_case_detail_is_collapsed(tmp_path):
    """102 expanded blocks is unreadable; detail belongs behind a summary."""
    html = _render(tmp_path)
    assert html.count("<details>") == len(EVAL["apm_eval"])


def test_metrics_are_a_table_with_threshold_and_verdict(tmp_path):
    html = _render(tmp_path)
    for header in ("Threshold", "Margin", "Coverage", "Verdict"):
        assert header in html
    assert "below threshold" in html      # cause_grounding 0.61 vs 0.80
    assert "not measured" in html         # faithfulness has no sample
    assert "thin sample" in html          # relevancy coverage 0.10
    assert "-0.19" in html                # 0.61 - 0.80, signed margin


def test_run_conditions_and_self_grading_are_visible(tmp_path):
    html = _render(tmp_path)
    for token in ("dataset 2.2.0", "prompt v1", "judge 8b (domain)", "SELF-GRADING", "102/102"):
        assert token in html, token


def test_hallucination_is_labelled_as_derived_and_not_gated(tmp_path):
    html = _render(tmp_path)
    assert "65%" in html
    assert "39 of 60 cases" in html
    assert "reported, not gated" in html


def test_a_weak_scenario_row_is_marked(tmp_path):
    html = _render(tmp_path)
    scenario_section = html.split("By scenario")[1].split("APM cases")[0]
    assert 'class="fail"' in scenario_section


def test_ungrounded_reason_is_highlighted(tmp_path):
    assert "jreason ungrounded" in _render(tmp_path)


def test_report_without_eval_data_still_renders(tmp_path):
    out = tmp_path / "r.html"
    generate_html_report([], eval_data=None, output_path=str(out))
    html = out.read_text(encoding="utf-8")
    assert "AIOps Sentinel Report" in html
    assert "AI evaluation" not in html


def test_old_schema_report_does_not_crash(tmp_path):
    """A pre-1.2.0 result has no run_metadata, hallucination, or by_scenario."""
    legacy = {"timestamp": "2026-04-16", "overall_score": 0.69,
              "apm_eval": [{"scenario": "x", "fault_type": "y", "overall_score": 0.7}],
              "log_eval": []}
    html = _render(tmp_path, data=legacy)
    assert "AI evaluation" in html
    assert "No metrics" in html
