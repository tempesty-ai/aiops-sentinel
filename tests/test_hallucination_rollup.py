"""The reported hallucination figure: derived from the checks that work, not gated."""
from eval.eval_suite import EvalReport, QUALITY_GATE_THRESHOLDS, summarize_hallucination


def _case(**kwargs):
    base = {"case_id": "apm-001", "cause_grounding_score": 1.0, "action_grounding_score": 1.0}
    return {**base, **kwargs}


def test_a_clean_run_reports_a_zero_rate():
    summary = summarize_hallucination(EvalReport(apm_results=[_case(), _case()], log_results=[]))
    assert summary["rate"] == 0.0
    assert summary["cases_with_hallucination"] == 0
    assert summary["cases_measurable"] == 2


def test_an_unsupported_diagnosis_counts_once_per_case():
    """A case is one hallucination even when both axes point at the same resource."""
    bad = _case(
        case_id="apm-008",
        cause_grounding_score=0.5, cause_grounding_unsupported=["cpu"],
        action_grounding_score=0.0, action_grounding_unsupported=["cpu", "memory"],
    )
    summary = summarize_hallucination(EvalReport(apm_results=[bad, _case()], log_results=[]))

    assert summary["cases_with_hallucination"] == 1
    assert summary["rate"] == 0.5
    assert summary["by_axis"] == {"cause": 1, "action": 1}
    assert summary["most_common_unsupported"]["cpu"] == 2


def test_the_axes_are_counted_separately():
    """Wrong diagnosis and wrong remedy are different failures."""
    cause_only = _case(cause_grounding_score=0.0, cause_grounding_unsupported=["memory"])
    action_only = _case(action_grounding_score=0.0, action_grounding_unsupported=["cpu"])
    summary = summarize_hallucination(EvalReport(apm_results=[cause_only, action_only], log_results=[]))

    assert summary["by_axis"] == {"cause": 1, "action": 1}
    assert summary["cases_with_hallucination"] == 2


def test_cases_with_nothing_checkable_are_not_in_the_denominator():
    """An answer naming no resource is not evidence of honesty."""
    unmeasurable = _case(cause_grounding_score=None, action_grounding_score=None)
    summary = summarize_hallucination(EvalReport(apm_results=[unmeasurable, _case()], log_results=[]))
    assert summary["cases_measurable"] == 1
    assert summary["rate"] == 0.0


def test_no_measurable_case_yields_no_rate_rather_than_zero():
    summary = summarize_hallucination(EvalReport(apm_results=[], log_results=[]))
    assert summary["rate"] is None


def test_the_figure_is_reported_not_gated():
    """Gating it would count the grounding evidence twice."""
    summary = summarize_hallucination(EvalReport(apm_results=[_case()], log_results=[]))
    assert summary["gated"] is False
    assert summary["derived_from"] == ["cause_grounding", "action_grounding"]
    assert "hallucination_min" not in QUALITY_GATE_THRESHOLDS


def test_examples_name_the_case_and_the_resource():
    bad = _case(case_id="apm-042", cause_grounding_score=0.0, cause_grounding_unsupported=["db_connection"])
    summary = summarize_hallucination(EvalReport(apm_results=[bad], log_results=[]))
    assert summary["examples"][0]["case_id"] == "apm-042"
    assert summary["examples"][0]["unsupported"]["cause"] == ["db_connection"]
