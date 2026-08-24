"""Action grounding: verify a recommendation against what the detector flagged."""
from eval.action_grounding import check_action_grounding, find_targeted_resources


def test_catches_the_observed_defect():
    """
    Real answer for a snapshot with cpu=92.5% and db_connections=12/50.
    HallucinationMetric scored this 1.0; the DB pool was never flagged.
    """
    action = ("Increase database connections on was_sample_01 server to 75 (50% buffer) "
              "to reduce connection overhead and improve response time.")
    verdict = check_action_grounding(action, ["cpu", "response_time"])

    assert "db_connection" in verdict.unsupported
    assert verdict.score is not None and verdict.score < 1.0
    assert "db_connection" in verdict.reason


def test_action_on_a_flagged_resource_is_grounded():
    verdict = check_action_grounding("Scale out the WAS instances to add CPU capacity.", ["cpu"])
    assert verdict.score == 1.0
    assert verdict.unsupported == []


def test_heap_and_memory_are_treated_as_the_same_pressure():
    """The detector may flag heap while the answer says memory, or the reverse."""
    assert check_action_grounding("Increase the JVM heap allocation.", ["memory"]).score == 1.0
    assert check_action_grounding("Add more memory to the container.", ["heap"]).score == 1.0


def test_action_without_a_resource_is_unmeasured():
    """An action can be valid without naming a metric, so it must not score 0."""
    verdict = check_action_grounding("Collect a thread dump and escalate to the owner.", ["cpu"])
    assert verdict.score is None
    assert verdict.targeted == []


def test_mentioning_a_resource_is_not_the_same_as_targeting_it():
    """"CPU is high, so increase the DB pool" targets the pool, not the CPU."""
    targeted = find_targeted_resources("CPU usage is high; increase the database connection pool.")
    assert targeted == ["db_connection"]


def test_partially_grounded_action_scores_between():
    verdict = check_action_grounding("Increase the heap size and expand the DB connection pool.", ["heap"])
    assert verdict.score == 0.5
    assert verdict.unsupported == ["db_connection"]


def test_korean_remediation_verbs_are_recognized():
    verdict = check_action_grounding("DB 커넥션 풀을 75로 증설하십시오.", ["cpu"])
    assert verdict.unsupported == ["db_connection"]


def test_empty_action_is_unmeasured():
    assert check_action_grounding("", ["cpu"]).score is None
