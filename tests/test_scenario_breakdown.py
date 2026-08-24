"""Per-class and per-phrasing breakdown: the aggregate hides brittleness."""
from eval.datasets.build_golden import LOG_TEMPLATES
from eval.eval_suite import EvalReport, LOG_TEST_SCENARIOS, summarize_by_scenario


def test_every_error_class_carries_multiple_phrasings():
    for template in LOG_TEMPLATES:
        variants = template["variants"]
        assert len(variants) >= 2, template["key"]
        assert len(set(variants)) == len(variants), f"{template['key']} has duplicate phrasings"


def test_generated_log_cases_are_all_distinct_wordings():
    lines = [c["context"].split("line=", 1)[1] for c in LOG_TEST_SCENARIOS]
    assert len(set(lines)) == len(lines)


def test_all_phrasings_of_a_class_are_exercised():
    variants_seen: dict[str, set] = {}
    for case in LOG_TEST_SCENARIOS:
        variants_seen.setdefault(case["scenario_key"], set()).add(case["variant"])
    for template in LOG_TEMPLATES:
        assert variants_seen[template["key"]] == set(range(len(template["variants"]))), template["key"]


def test_breakdown_exposes_a_class_that_only_fails_on_one_wording():
    """67% overall, but one phrasing fails every time - that is the signal."""
    log = [
        {"scenario_key": "connection_refused", "variant": 0, "error_type_correct": True},
        {"scenario_key": "connection_refused", "variant": 1, "error_type_correct": True},
        {"scenario_key": "connection_refused", "variant": 2, "error_type_correct": False},
    ]
    summary = summarize_by_scenario(EvalReport(apm_results=[], log_results=log, overall_score=0.9))

    entry = summary["connection_refused"]
    assert entry["accuracy"] == 0.67
    assert entry["by_variant"]["v3"]["accuracy"] == 0.0
    assert entry["by_variant"]["v1"]["accuracy"] == 1.0


def test_breakdown_orders_weakest_first():
    apm = [
        {"scenario_key": "cpu_spike", "fault_type_correct": True},
        {"scenario_key": "db_connection_pool", "fault_type_correct": False},
    ]
    summary = summarize_by_scenario(EvalReport(apm_results=apm, log_results=[], overall_score=0.5))
    assert list(summary) == ["db_connection_pool", "cpu_spike"]


def test_unmeasurable_cases_do_not_count_against_a_class():
    apm = [
        {"scenario_key": "cpu_spike", "fault_type_correct": True},
        {"scenario_key": "cpu_spike", "fault_type_correct": True, "classification_unmeasured": True},
    ]
    entry = summarize_by_scenario(EvalReport(apm_results=apm, log_results=[], overall_score=0.9))["cpu_spike"]
    assert entry["cases"] == 2
    assert entry["unmeasured"] == 1
    assert entry["accuracy"] == 1.0   # measured on the one usable case


def test_class_with_no_measurable_case_has_no_accuracy():
    apm = [{"scenario_key": "cpu_spike", "fault_type_correct": True, "classification_unmeasured": True}]
    entry = summarize_by_scenario(EvalReport(apm_results=apm, log_results=[], overall_score=0.9))["cpu_spike"]
    assert entry["accuracy"] is None
