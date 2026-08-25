"""Single-call domain judge: two axes, ratio scoring, and refusing to guess."""
from eval.domain_judge import DomainJudgeVerdict, judge, score_verdict
from eval.eval_suite import ABNORMAL, NORMAL, APM_TEST_SCENARIOS, label_facts


def test_labels_follow_the_detector_not_the_numbers():
    """
    An 8B judge read a 94.1% CPU reading against an 80% threshold as "within
    normal range". The detector already decided, so the label is handed over.
    """
    case = next(c for c in APM_TEST_SCENARIOS if c["scenario_key"] == "cpu_spike")
    labelled = label_facts(case["context"])

    cpu = [f for f in labelled if "CPU" in f]
    assert cpu and all(f.startswith(ABNORMAL) for f in cpu)
    db = [f for f in labelled if "DB 커넥션" in f]
    assert db and all(f.startswith(NORMAL) for f in db)


def test_every_fact_is_labelled_exactly_once():
    for case in APM_TEST_SCENARIOS[:5]:
        for fact in label_facts(case["context"]):
            assert fact.startswith((ABNORMAL, NORMAL)), fact


def test_memory_case_labels_memory_abnormal_and_cpu_normal():
    case = next(c for c in APM_TEST_SCENARIOS if c["scenario_key"] == "memory_leak")
    labelled = label_facts(case["context"])
    assert any(f.startswith(ABNORMAL) and "메모리" in f for f in labelled)
    assert any(f.startswith(NORMAL) and "CPU" in f for f in labelled)


def test_free_text_context_is_left_alone():
    assert label_facts("no bullet lines") == ["no bullet lines"]


def test_scores_are_ratios_of_the_returned_lists():
    verdict = DomainJudgeVerdict(
        supported_claims=["a", "b", "c"],
        unsupported_claims=["d"],
        relevant_statements=["x", "y"],
        irrelevant_statements=["z"],
        reason="note",
    )
    scores = score_verdict(verdict)
    assert scores.faithfulness == 0.75       # 3 supported of 4 claims
    assert scores.answer_relevancy == 0.67   # 2 relevant of 3 statements
    assert "note" in scores.faithfulness_reason


def test_a_clean_answer_scores_one_and_says_so():
    scores = score_verdict(DomainJudgeVerdict(supported_claims=["a"], relevant_statements=["b"]))
    assert scores.faithfulness == 1.0
    assert scores.answer_relevancy == 1.0
    assert "Every claim is supported" in scores.faithfulness_reason


def test_empty_lists_are_unmeasured_rather_than_zero():
    """Nothing to divide is not evidence of a perfect or a failing answer."""
    scores = score_verdict(DomainJudgeVerdict())
    assert scores.faithfulness == -1.0
    assert scores.answer_relevancy == -1.0


def test_a_judge_that_breaks_yields_unmeasured_not_a_guess():
    class Broken:
        def generate(self, prompt, schema=None):
            raise RuntimeError("model down")

    scores = judge(Broken(), task="t", answer="a", facts=["f"])
    assert scores.faithfulness == -1.0
    assert "Judge call failed" in scores.judge_error


def test_a_judge_ignoring_the_schema_yields_unmeasured():
    class Freeform:
        def generate(self, prompt, schema=None):
            return "I think it looks fine"

    scores = judge(Freeform(), task="t", answer="a", facts=["f"])
    assert scores.faithfulness == -1.0
    assert "did not return the required structure" in scores.judge_error


def test_the_judge_is_no_longer_asked_about_contradictions():
    """
    That axis moved to the deterministic grounding checks; leaving it here would
    reintroduce the 2-of-3 misjudgement that motivated the move.
    """
    seen = {}

    class Capture:
        def generate(self, prompt, schema=None):
            seen["prompt"] = prompt
            return DomainJudgeVerdict(supported_claims=["ok"], relevant_statements=["ok"])

    judge(Capture(), task="t", answer="a", facts=["line"])
    assert "contradict" not in seen["prompt"].lower()
    assert "contradicted_facts" not in DomainJudgeVerdict.model_fields
