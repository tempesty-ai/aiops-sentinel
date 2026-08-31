"""Golden-set construction: label validity, balance, reproducibility, sampling."""
import json
from pathlib import Path

from eval.datasets.build_golden import APM_SCENARIO_LABELS, LOG_TEMPLATES, build
from eval.datasets.build_golden import DATASET_VERSION
from eval.eval_suite import APM_TEST_SCENARIOS, LOG_TEST_SCENARIOS, stratified_sample


def test_dataset_is_large_enough_to_move_a_metric_less_than_a_case():
    total = len(APM_TEST_SCENARIOS) + len(LOG_TEST_SCENARIOS)
    # With 5 cases one case moved the score by 20 points; 100+ keeps it under 1.
    assert total >= 100, f"only {total} cases"


def test_every_case_carries_a_label_and_a_traceable_id():
    for case in APM_TEST_SCENARIOS:
        assert case["case_id"] and case["context"]
        assert case["expected_fault_types"], case["case_id"]
        assert case["expected_keywords"], case["case_id"]
    for case in LOG_TEST_SCENARIOS:
        assert case["case_id"] and case["context"]
        assert case["expected_error_types"], case["case_id"]

    ids = [c["case_id"] for c in APM_TEST_SCENARIOS + LOG_TEST_SCENARIOS]
    assert len(ids) == len(set(ids)), "case ids must be unique"


def test_labels_never_include_the_parser_fallback_value():
    """'Unknown' is what the parser emits on failure, so it cannot be a correct answer."""
    for case in APM_TEST_SCENARIOS:
        assert "unknown" not in [t.lower() for t in case["expected_fault_types"]]
    for case in LOG_TEST_SCENARIOS:
        assert "unknown" not in [t.lower() for t in case["expected_error_types"]]


def test_memory_scenario_does_not_expect_an_uninferable_label():
    """A single snapshot cannot separate a leak from legitimate high usage."""
    labels = APM_SCENARIO_LABELS["memory_leak"]["fault_types"]
    assert "leak" not in labels
    assert "memory" in labels


def test_every_scenario_class_is_represented():
    apm_keys = {c["scenario_key"] for c in APM_TEST_SCENARIOS}
    log_keys = {c["scenario_key"] for c in LOG_TEST_SCENARIOS}
    assert apm_keys == set(APM_SCENARIO_LABELS)
    assert log_keys == {t["key"] for t in LOG_TEMPLATES}


def test_apm_cases_are_all_anomalies_the_detector_would_forward():
    """The runtime pipeline only asks the AI about detected anomalies."""
    for case in APM_TEST_SCENARIOS:
        assert case["triggered_rules"], f"{case['case_id']} has no triggered rule"


def test_same_seed_rebuilds_an_identical_dataset():
    a = build(apm_count=10, log_count=14, seed=42)
    b = build(apm_count=10, log_count=14, seed=42)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_different_seed_changes_the_dataset():
    a = build(apm_count=10, log_count=14, seed=1)
    b = build(apm_count=10, log_count=14, seed=2)
    assert json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)


def test_sampling_keeps_every_scenario_represented():
    picked = stratified_sample(APM_TEST_SCENARIOS, 5, seed=0)
    assert len(picked) == 5
    assert len({c["scenario_key"] for c in picked}) == len(APM_SCENARIO_LABELS)


def test_sampling_is_reproducible_and_bounded():
    assert stratified_sample(APM_TEST_SCENARIOS, 7, seed=3) == stratified_sample(APM_TEST_SCENARIOS, 7, seed=3)
    assert stratified_sample(APM_TEST_SCENARIOS, 0, seed=0) == APM_TEST_SCENARIOS
    assert stratified_sample(APM_TEST_SCENARIOS, 9999, seed=0) == APM_TEST_SCENARIOS


def test_only_the_active_dataset_ships():
    """
    The retired 5-case set carried a label a single snapshot cannot support
    ("leak"), so leaving the file around invited an eval that looks valid and is
    not. Git holds the history; the builder reproduces any size from a seed.
    """
    shipped = sorted(p.name for p in Path("eval/datasets").glob("*.json"))
    assert shipped == ["golden_v2.json"], shipped


def test_the_builder_can_reproduce_a_small_set_without_a_stored_file():
    small = build(apm_count=3, log_count=2, seed=1)
    assert len(small["apm_cases"]) + len(small["log_cases"]) == 5
    assert small["dataset_version"] == DATASET_VERSION
