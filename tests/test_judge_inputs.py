"""What the judge is given: separable facts, and the task the model really had."""
from eval.eval_suite import (
    APM_JUDGE_TASK,
    APM_TEST_SCENARIOS,
    LOG_JUDGE_TASK,
    split_context_into_facts,
)
from apm.ai_analyzer import APM_PROMPT_VERSIONS
from logwatch.ai_classifier import SYSTEM_PROMPT as LOG_SYSTEM_PROMPT


def test_a_snapshot_becomes_many_separately_judgeable_facts():
    """
    HallucinationMetric emits one verdict per context document. With a single
    blob the score can only be 0.0 or 1.0 - which is why it read 1.0 on every
    case, including one that contradicted the metrics.
    """
    facts = split_context_into_facts(APM_TEST_SCENARIOS[0]["context"])
    assert len(facts) > 1
    assert all(fact.strip() for fact in facts)
    assert not any(fact.startswith("-") for fact in facts), "bullet marker should be stripped"


def test_the_metric_that_catches_the_known_defect_is_its_own_fact():
    """The db-connection reading has to be checkable on its own."""
    facts = split_context_into_facts(APM_TEST_SCENARIOS[0]["context"])
    assert any("DB 커넥션" in fact for fact in facts)


def test_free_text_context_falls_back_to_one_document():
    assert split_context_into_facts("no bullet lines here") == ["no bullet lines here"]
    assert split_context_into_facts("") == [""]


def test_apm_judge_question_matches_the_sections_the_model_must_produce():
    """
    Relevancy is scored against this question. Asking for less than the system
    prompt requires penalises the model for following instructions.
    """
    for section in ("Fault Type", "Root Cause", "Immediate Action", "Prevention", "Severity"):
        assert section in APM_PROMPT_VERSIONS["v1"], f"{section} missing from the system prompt"
        assert section in APM_JUDGE_TASK, f"{section} missing from the judge question"


def test_log_judge_question_matches_the_classifier_prompt():
    for section in ("Error Type", "Severity", "Recurrence", "Recommended Action"):
        assert section in LOG_SYSTEM_PROMPT
        assert section in LOG_JUDGE_TASK
