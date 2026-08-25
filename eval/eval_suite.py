"""
DeepEval-based AI quality evaluation suite.
Includes a quality gate with pass/fail criteria for portfolio-friendly QA evidence.
"""
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from apm.ai_analyzer import APMAIAnalyzer
from eval.action_grounding import check_action_grounding, check_cause_grounding
from eval.domain_judge import judge as domain_judge
from config.settings import (
    APM_PROMPT_VERSION,
    EVAL_JUDGE_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    get_model_digest,
    get_ollama_version,
)
from logwatch.ai_classifier import LogAIClassifier

SCHEMA_VERSION = "1.2.0"

# Minimum share of applicable cases that must yield a valid measurement before a
# metric is allowed to certify a PASS. Guards against a metric that looks fine but
# rests on one surviving judge call.
QUALITY_GATE_MIN_COVERAGE = 0.6

QUALITY_GATE_THRESHOLDS = {
    "overall_score_min": 0.70,
    "apm_fault_type_accuracy_min": 0.67,
    "log_error_type_accuracy_min": 0.50,
    "relevancy_min": 0.60,
    "faithfulness_min": 0.60,
    # An action may only target a resource the detector flagged. Judgement call,
    # set to allow occasional defensible over-reach but not systematic drift.
    "action_grounding_min": 0.80,
    # The diagnosis is held to the same bar as the remedy.
    "cause_grounding_min": 0.80,
}


# The judge scores relevancy against this question, so it has to describe the
# task the model was actually given. Asking only for "root cause and action"
# made AnswerRelevancy penalise the Prevention and Severity sections that the
# system prompt requires - the better the model followed instructions, the lower
# it scored.
# deepeval: three library metrics, ~9 judge calls per case, ~280s on CPU.
# domain:   the same three questions in one structured call, ~20s on CPU.
# Recorded per run and treated as a changed variable by compare_runs, because a
# score from one mode is not evidence about the other.
JUDGE_MODE_DEEPEVAL = "deepeval"
JUDGE_MODE_DOMAIN = "domain"

APM_JUDGE_TASK = (
    "As a senior APM incident analyst, analyse this anomaly and report all five "
    "sections - Fault Type, Root Cause, Immediate Action, Prevention, Severity - "
    "for:"
)
LOG_JUDGE_TASK = (
    "Classify this collector log error and report all four sections - Error Type, "
    "Severity, Recurrence, Recommended Action - for:"
)

class OllamaEvalModel(DeepEvalBaseLLM):
    """
    DeepEval judge. Uses EVAL_JUDGE_MODEL so the model under test is not its own
    judge; falls back to OLLAMA_MODEL (self-grading) when the judge is unset.
    """

    def __init__(self, model: str = ""):
        self.model_name = model or EVAL_JUDGE_MODEL
        self._llm = ChatOllama(
            model=self.model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
        )

    def load_model(self):
        return self._llm

    def supports_structured_outputs(self) -> bool:
        return True

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        """
        DeepEval calls this with `schema` when a metric wants structured output.

        Ollama can constrain decoding to a JSON schema, so the judge is made to
        answer in shape rather than asked politely to. Without this the metrics
        fell back to free-text parsing and every judge below 70b failed with
        "Evaluation LLM outputted an invalid JSON".

        Returning a schema instance lets DeepEval skip its own JSON parsing.
        """
        if schema is None:
            return self._llm.invoke(prompt).content
        try:
            return self._llm.with_structured_output(schema).invoke(prompt)
        except Exception:
            # A model that cannot honour the constraint still gets a chance to
            # answer; DeepEval will parse and report the failure as before.
            return self._llm.invoke(prompt).content

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"


DATASETS_DIR = Path(__file__).parent / "datasets"
GOLDEN_DATASET_PATH = DATASETS_DIR / (os.getenv("GOLDEN_DATASET", "golden_v2.json").strip() or "golden_v2.json")


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> dict:
    """
    Load the labeled golden set from disk.

    The dataset lives outside the code so labels can be reviewed and versioned
    independently of scoring logic. Select a different file with GOLDEN_DATASET.
    """
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def stratified_sample(cases: list[dict], size: int, seed: int = 0) -> list[dict]:
    """
    Take `size` cases while keeping every scenario_key represented.

    A full run costs 4 LLM calls per case, so a subset is often the only
    affordable option; drawing it evenly stops the subset from silently
    over-representing one fault class.
    """
    if size <= 0 or size >= len(cases):
        return cases

    buckets: dict[str, list[dict]] = {}
    for case in cases:
        buckets.setdefault(case.get("scenario_key", "unknown"), []).append(case)
    for bucket in buckets.values():
        random.Random(seed).shuffle(bucket)

    picked: list[dict] = []
    round_index = 0
    while len(picked) < size:
        added = False
        for key in sorted(buckets):
            if round_index < len(buckets[key]) and len(picked) < size:
                picked.append(buckets[key][round_index])
                added = True
        if not added:
            break
        round_index += 1
    return picked


_GOLDEN = load_golden_dataset()
DATASET_VERSION = _GOLDEN["dataset_version"]
APM_TEST_SCENARIOS = _GOLDEN["apm_cases"]
LOG_TEST_SCENARIOS = _GOLDEN["log_cases"]


@dataclass
class DeepEvalMetrics:
    hallucination: float = 0.0
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    hallucination_reason: str = ""
    relevancy_reason: str = ""
    faithfulness_reason: str = ""


@dataclass
class QualityGateResult:
    passed: bool
    status: str                       # PASS | FAIL | INCONCLUSIVE
    score: float
    failed_rules: list[str]              # measured and below threshold
    unmeasured_rules: list[str]          # no valid sample -> cannot certify either way
    insufficient_sample_rules: list[str]  # above threshold but measured on too few cases
    metrics: dict                        # value, or None when unmeasured
    sample_counts: dict[str, int]
    coverage: dict[str, float]           # valid samples / applicable cases
    thresholds: dict[str, float]
    min_coverage: float = QUALITY_GATE_MIN_COVERAGE


@dataclass
class EvalReport:
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    apm_results: list[dict] = field(default_factory=list)
    log_results: list[dict] = field(default_factory=list)
    overall_score: float = 0.0
    quality_gate: QualityGateResult | None = None
    dataset_version: str = DATASET_VERSION
    analysis_model: str = OLLAMA_MODEL
    judge_model: str = EVAL_JUDGE_MODEL
    prompt_version: str = APM_PROMPT_VERSION
    judge_mode: str = JUDGE_MODE_DOMAIN
    ollama_version: str = ""
    analysis_digest: str = ""
    judge_digest: str = ""
    cases_available: int = 0
    sample_seed: int = 0


def _avg(values: list[float]) -> float | None:
    """Mean of the sample, or None when there is no sample at all.

    Returning None keeps "never measured" distinguishable from "measured as 0.0";
    collapsing the two used to fail the gate on a judge-model outage.
    """
    return round(sum(values) / len(values), 2) if values else None


def summarize_hallucination(report: "EvalReport") -> dict:
    """
    How often the analysis asserted something the metrics do not support.

    Derived, not measured: it is a roll-up of `cause_grounding` and
    `action_grounding`, which are the two places this project can detect a
    hallucination deterministically. It is reported rather than gated, so the same
    evidence is never counted twice - the gate already holds both axes to 0.80.

    Reported as a rate (0.0 = nothing unsupported) so the direction is obvious.
    The old inverted `hallucination` metric read backwards from its own name and
    scored 1.0 on a case that contradicted the data outright.
    """
    axes = (("cause", "cause_grounding_score", "cause_grounding_unsupported"),
            ("action", "action_grounding_score", "action_grounding_unsupported"))

    measurable = 0
    hallucinated: list[dict] = []
    by_axis = {"cause": 0, "action": 0}
    resource_counts: dict[str, int] = {}

    for result in report.apm_results:
        scores = [result.get(key) for _, key, _ in axes]
        if not any(isinstance(score, (int, float)) for score in scores):
            continue
        measurable += 1

        unsupported: dict[str, list[str]] = {}
        for axis, score_key, unsupported_key in axes:
            if not isinstance(result.get(score_key), (int, float)):
                continue
            resources = result.get(unsupported_key) or []
            if resources:
                by_axis[axis] += 1
                unsupported[axis] = list(resources)
                for resource in resources:
                    resource_counts[resource] = resource_counts.get(resource, 0) + 1

        if unsupported:
            hallucinated.append({"case_id": result.get("case_id"), "unsupported": unsupported})

    rate = round(len(hallucinated) / measurable, 3) if measurable else None
    return {
        "derived_from": ["cause_grounding", "action_grounding"],
        "gated": False,
        "rate": rate,
        "cases_with_hallucination": len(hallucinated),
        "cases_measurable": measurable,
        "by_axis": by_axis,
        "most_common_unsupported": dict(sorted(resource_counts.items(), key=lambda kv: -kv[1])),
        "examples": hallucinated[:10],
    }


def summarize_by_scenario(report: "EvalReport") -> dict:
    """
    Accuracy per fault class, and per phrasing within a class.

    An aggregate score hides brittleness: a class can look fine at 67% while in
    fact failing every case of one particular wording. `by_variant` makes that
    visible, which is the point of carrying several phrasings per class.
    """
    summary: dict[str, dict] = {}
    tracks = (
        (report.apm_results, "fault_type_correct"),
        (report.log_results, "error_type_correct"),
    )
    for results, correct_key in tracks:
        for result in results:
            key = result.get("scenario_key") or "unknown"
            entry = summary.setdefault(
                key, {"cases": 0, "correct": 0, "unmeasured": 0, "accuracy": None, "by_variant": {}}
            )
            entry["cases"] += 1
            if result.get("classification_unmeasured"):
                entry["unmeasured"] += 1
                continue
            hit = bool(result.get(correct_key))
            entry["correct"] += hit

            variant = result.get("variant")
            if variant is not None:
                bucket = entry["by_variant"].setdefault(f"v{variant + 1}", {"cases": 0, "correct": 0})
                bucket["cases"] += 1
                bucket["correct"] += hit

    for entry in summary.values():
        measured = entry["cases"] - entry["unmeasured"]
        entry["accuracy"] = round(entry["correct"] / measured, 2) if measured else None
        for bucket in entry["by_variant"].values():
            bucket["accuracy"] = round(bucket["correct"] / bucket["cases"], 2) if bucket["cases"] else None

    # Weakest first: that is the order someone reading the report needs.
    return dict(sorted(summary.items(), key=lambda kv: (kv[1]["accuracy"] is None, kv[1]["accuracy"] or 0)))


def evaluate_quality_gate(report: EvalReport) -> QualityGateResult:
    all_results = report.apm_results + report.log_results

    # Classification accuracy counts a case only when the label was actually
    # measurable: a fallback/unparsable answer is not evidence of correctness.
    apm_scored = [r for r in report.apm_results if not r.get("classification_unmeasured")]
    log_scored = [r for r in report.log_results if not r.get("classification_unmeasured")]

    # A fallback answer is boilerplate, not model output, so judging it measures
    # nothing about the model under test.
    answered = [r for r in all_results if not r.get("llm_failed")]

    def judged(key: str) -> list[float]:
        return [r[key] for r in answered if isinstance(r.get(key), (int, float)) and r[key] >= 0]

    samples = {
        "apm_fault_type_accuracy": [1.0 if r.get("fault_type_correct") else 0.0 for r in apm_scored],
        # None means the text named no checkable resource -> not a sample.
        "cause_grounding": [
            r["cause_grounding_score"] for r in report.apm_results
            if isinstance(r.get("cause_grounding_score"), (int, float))
        ],
        "action_grounding": [
            r["action_grounding_score"] for r in report.apm_results
            if isinstance(r.get("action_grounding_score"), (int, float))
        ],
        "log_error_type_accuracy": [1.0 if r.get("error_type_correct") else 0.0 for r in log_scored],
        "relevancy": judged("relevancy_score"),
        "faithfulness": judged("faithfulness_score"),
    }

    metrics: dict = {"overall_score": round(report.overall_score, 2)}
    metrics.update({name: _avg(values) for name, values in samples.items()})

    sample_counts = {"overall_score": len(all_results)}
    sample_counts.update({name: len(values) for name, values in samples.items()})

    # How many cases each metric could have been measured on.
    applicable = {
        "overall_score": len(all_results),
        "apm_fault_type_accuracy": len(report.apm_results),
        "action_grounding": len(report.apm_results),
        "cause_grounding": len(report.apm_results),
        "log_error_type_accuracy": len(report.log_results),
        "relevancy": len(answered),
        "faithfulness": len(answered),
    }
    coverage = {
        name: (round(sample_counts[name] / applicable[name], 2) if applicable[name] else 0.0)
        for name in metrics
    }

    failed_rules: list[str] = []
    unmeasured_rules: list[str] = []
    insufficient_sample_rules: list[str] = []
    for name, value in metrics.items():
        if value is None:
            unmeasured_rules.append(name)
            continue
        if value < QUALITY_GATE_THRESHOLDS[f"{name}_min"]:
            # A below-threshold reading is evidence of a problem even on thin data.
            failed_rules.append(name)
        elif coverage[name] < QUALITY_GATE_MIN_COVERAGE:
            insufficient_sample_rules.append(name)

    total = len(metrics)
    not_certified = set(failed_rules) | set(unmeasured_rules) | set(insufficient_sample_rules)
    score = round((total - len(not_certified)) / total, 2)

    if failed_rules:
        status = "FAIL"
    elif unmeasured_rules or insufficient_sample_rules:
        status = "INCONCLUSIVE"
    else:
        status = "PASS"

    return QualityGateResult(
        passed=(status == "PASS"),
        status=status,
        score=score,
        failed_rules=failed_rules,
        unmeasured_rules=unmeasured_rules,
        insufficient_sample_rules=insufficient_sample_rules,
        metrics=metrics,
        sample_counts=sample_counts,
        coverage=coverage,
        thresholds=QUALITY_GATE_THRESHOLDS,
        min_coverage=QUALITY_GATE_MIN_COVERAGE,
    )


# A metric line such as "  - DB 커넥션: 12/50 (24%)" is an independently
# checkable fact; the section headers around it are not.
_FACT_LINE = re.compile(r"^\s*-\s*(?P<body>\S.*\S)\s*$")


# The context lists the breached thresholds first, then every metric. Everything
# before this heading belongs to the "detected" block.
_METRIC_SECTION = "전체 지표"

ABNORMAL = "[ABNORMAL]"
NORMAL = "[NORMAL]"


def split_context_into_facts(context: str) -> list[str]:
    """
    Break an anomaly context into separately judgeable statements.

    Falls back to the whole block when nothing line-like is found, so a caller
    passing free text still gets a usable (if coarse) context.
    """
    facts = [
        match.group("body")
        for line in (context or "").splitlines()
        if (match := _FACT_LINE.match(line))
    ]
    return facts or [context]


def label_facts(context: str) -> list[str]:
    """
    Tag each metric with the detector's own verdict.

    An 8B judge cannot reliably decide whether 94.1% clears an 80% threshold - it
    read genuine breaches as "within normal range" and invented contradictions
    from them. The detector already made that call, so the judge is handed the
    label and only has to compare it against what the analysis claims. Same rule
    as the action-grounding check: never re-derive a decision the detector owns.
    """
    facts = split_context_into_facts(context)
    if not facts or facts == [context]:
        return facts

    in_metric_section = False
    breached: list[str] = []
    labelled: list[str] = []
    for line in (context or "").splitlines():
        if _METRIC_SECTION in line:
            in_metric_section = True
            continue
        match = _FACT_LINE.match(line)
        if not match:
            continue
        body = match.group("body")
        if not in_metric_section:
            breached.append(body)
            labelled.append(f"{ABNORMAL} {body}")
        else:
            # A metric restated in the breach block is abnormal; the rest is not.
            name = body.split(":")[0].split()[0]
            is_breached = any(name and name in rule for rule in breached)
            labelled.append(f"{ABNORMAL if is_breached else NORMAL} {body}")
    return labelled or facts


def _blend_scores(custom_score: float, deepeval: "DeepEvalMetrics") -> tuple[float | None, float]:
    """
    Combine the rule-based score with the judge score (50:50).

    When every judge metric failed there is no independent measurement, so the
    judge half is reported as None rather than being back-filled with the custom
    score - back-filling used to count the same evidence twice and quietly
    inflate the total during a judge outage.
    """
    valid = [v for v in (deepeval.hallucination, deepeval.answer_relevancy, deepeval.faithfulness) if v >= 0]
    if not valid:
        return None, round(custom_score, 2)
    deepeval_score = round(sum(valid) / len(valid), 2)
    return deepeval_score, round(custom_score * 0.5 + deepeval_score * 0.5, 2)


def _match_label(expected: list[str], parsed_field: str, raw_response: str) -> tuple[bool, bool]:
    """
    Compare expected labels against the model answer.

    Returns (strict, loose):
      strict - matched against the PARSED classification field, which is what the
               alert actually shows the operator. This is the score that counts.
      loose  - matched anywhere in the raw response. Kept only for comparison, to
               show how much the previous whole-response matching inflated results.
    """
    field_low = (parsed_field or "").lower()
    raw_low = (raw_response or "").lower()
    strict = any(label.lower() in field_low for label in expected)
    loose = strict or any(label.lower() in raw_low for label in expected)
    return strict, loose


class AIQualityEvaluator:
    def __init__(self, sample_size: int = 0, sample_seed: int = 0, judge_mode: str = JUDGE_MODE_DOMAIN):
        self.apm_analyzer = APMAIAnalyzer()
        self.log_classifier = LogAIClassifier()
        self._eval_model = OllamaEvalModel()
        self.sample_size = sample_size
        self.sample_seed = sample_seed
        self.judge_mode = judge_mode

    def _cases(self, cases: list[dict]) -> list[dict]:
        """Apply the run's sampling, splitting the budget across both tracks."""
        if self.sample_size <= 0:
            return cases
        total = len(APM_TEST_SCENARIOS) + len(LOG_TEST_SCENARIOS)
        share = max(1, round(self.sample_size * len(cases) / total)) if total else 0
        return stratified_sample(cases, share, self.sample_seed)

    def _judge(self, question: str, answer: str, context: str) -> DeepEvalMetrics:
        """Route to the configured judge; both return the same three scores."""
        if self.judge_mode == JUDGE_MODE_DOMAIN:
            scores = domain_judge(self._eval_model, question, answer, label_facts(context))
            return DeepEvalMetrics(
                answer_relevancy=scores.answer_relevancy,
                faithfulness=scores.faithfulness,
                relevancy_reason=scores.relevancy_reason,
                faithfulness_reason=scores.faithfulness_reason,
                hallucination=-1.0,   # this axis is graded by cause_grounding
                hallucination_reason=scores.judge_error,
            )
        return self._run_deepeval_metrics(question, answer, context)

    def _run_deepeval_metrics(self, question: str, answer: str, context: str) -> DeepEvalMetrics:
        metrics = DeepEvalMetrics()
        facts = split_context_into_facts(context)
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            # HallucinationMetric emits one verdict per context document, so a
            # single blob can only ever score 0.0 or 1.0 - and it scored 1.0 on
            # every case, including an answer that contradicted the metrics.
            # Splitting the snapshot into separate facts gives the metric
            # something to discriminate with, and names the fact that clashed.
            retrieval_context=facts,
            context=facts,
        )

        try:
            hallucination_metric = HallucinationMetric(threshold=0.5, model=self._eval_model, include_reason=True)
            hallucination_metric.measure(test_case)
            metrics.hallucination = round(1.0 - (hallucination_metric.score or 0.0), 2)
            metrics.hallucination_reason = hallucination_metric.reason or ""
        except Exception as exc:
            print(f"[Eval] Hallucination metric failed: {exc}")
            metrics.hallucination = -1.0

        try:
            relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=self._eval_model, include_reason=True)
            relevancy_metric.measure(test_case)
            metrics.answer_relevancy = round(relevancy_metric.score or 0.0, 2)
            metrics.relevancy_reason = relevancy_metric.reason or ""
        except Exception as exc:
            print(f"[Eval] AnswerRelevancy metric failed: {exc}")
            metrics.answer_relevancy = -1.0

        try:
            faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=self._eval_model, include_reason=True)
            faithfulness_metric.measure(test_case)
            metrics.faithfulness = round(faithfulness_metric.score or 0.0, 2)
            metrics.faithfulness_reason = faithfulness_metric.reason or ""
        except Exception as exc:
            print(f"[Eval] Faithfulness metric failed: {exc}")
            metrics.faithfulness = -1.0

        return metrics

    def run_apm_eval(self) -> list[dict]:
        results: list[dict] = []
        for scenario in self._cases(APM_TEST_SCENARIOS):
            print(f"[APM Eval] {scenario['name']}")
            analysis = self.apm_analyzer.analyze(scenario["context"])
            response_text = analysis.raw_response.lower()

            keyword_hits = sum(1 for kw in scenario["expected_keywords"] if kw.lower() in response_text)
            keyword_score = round(keyword_hits / len(scenario["expected_keywords"]), 2)
            fault_matched, fault_matched_loose = _match_label(
                scenario["expected_fault_types"], analysis.fault_type, analysis.raw_response
            )
            # A fallback answer carries no classification evidence either way.
            classification_unmeasured = analysis.llm_failed or analysis.fault_type_missing
            completeness_score = round(
                sum(
                    [
                        1 if analysis.root_cause and len(analysis.root_cause) > 10 else 0,
                        1 if analysis.immediate_action and len(analysis.immediate_action) > 10 else 0,
                        1 if analysis.prevention and len(analysis.prevention) > 10 else 0,
                    ]
                )
                / 3,
                2,
            )

            triggered = scenario.get("triggered_metrics", [])
            grounding = check_action_grounding(analysis.immediate_action, triggered)
            cause = check_cause_grounding(analysis.root_cause, triggered)

            deepeval = self._judge(
                question=f"{APM_JUDGE_TASK}\n{scenario['context']}",
                answer=analysis.raw_response,
                context=scenario["context"],
            )

            custom_score = keyword_score * 0.4 + (1.0 if fault_matched else 0.0) * 0.3 + completeness_score * 0.3
            deepeval_score, overall = _blend_scores(custom_score, deepeval)

            results.append(
                {
                    "case_id": scenario["case_id"],
                    "scenario_key": scenario.get("scenario_key"),
                    "scenario": scenario["name"],
                    "answer_excerpt": analysis.raw_response[:400],
                    "fault_type": analysis.fault_type,
                    "fault_type_correct": fault_matched,
                    "fault_type_correct_loose": fault_matched_loose,
                    "classification_unmeasured": classification_unmeasured,
                    "llm_failed": analysis.llm_failed,
                    "keyword_score": keyword_score,
                    "completeness_score": completeness_score,
                    "cause_grounding_score": cause.score,
                    "cause_grounding_unsupported": cause.unsupported,
                    "cause_grounding_reason": cause.reason,
                    "action_grounding_score": grounding.score,
                    "action_grounding_targeted": grounding.targeted,
                    "action_grounding_unsupported": grounding.unsupported,
                    "action_grounding_reason": grounding.reason,
                    "custom_score": round(custom_score, 2),
                    "hallucination_score": deepeval.hallucination,
                    "hallucination_reason": deepeval.hallucination_reason,
                    "relevancy_score": deepeval.answer_relevancy,
                    "relevancy_reason": deepeval.relevancy_reason,
                    "faithfulness_score": deepeval.faithfulness,
                    "faithfulness_reason": deepeval.faithfulness_reason,
                    "deepeval_score": deepeval_score,
                    "deepeval_unmeasured": deepeval_score is None,
                    "overall_score": overall,
                    "severity": analysis.severity,
                }
            )
        return results

    def run_log_eval(self) -> list[dict]:
        results: list[dict] = []
        for scenario in self._cases(LOG_TEST_SCENARIOS):
            print(f"[Log Eval] {scenario['name']}")
            classification = self.log_classifier.classify(scenario["context"])
            response_text = classification.raw_response.lower()

            type_matched, type_matched_loose = _match_label(
                scenario["expected_error_types"], classification.error_type, classification.raw_response
            )
            classification_unmeasured = classification.llm_failed or classification.error_type_missing
            action_present = len(classification.recommended_action) > 10

            deepeval = self._judge(
                question=f"{LOG_JUDGE_TASK}\n{scenario['context']}",
                answer=classification.raw_response,
                context=scenario["context"],
            )

            custom_score = (
                (1.0 if type_matched else 0.0) * 0.5
                + (1.0 if action_present else 0.0) * 0.3
                + (1.0 if classification.severity else 0.0) * 0.2
            )
            deepeval_score, overall = _blend_scores(custom_score, deepeval)

            results.append(
                {
                    "case_id": scenario["case_id"],
                    "scenario_key": scenario.get("scenario_key"),
                    "variant": scenario.get("variant"),
                    "scenario": scenario["name"],
                    "answer_excerpt": classification.raw_response[:400],
                    "error_type": classification.error_type,
                    "error_type_correct": type_matched,
                    "error_type_correct_loose": type_matched_loose,
                    "classification_unmeasured": classification_unmeasured,
                    "llm_failed": classification.llm_failed,
                    "severity": classification.severity,
                    "action_present": action_present,
                    "custom_score": round(custom_score, 2),
                    "hallucination_score": deepeval.hallucination,
                    "hallucination_reason": deepeval.hallucination_reason,
                    "relevancy_score": deepeval.answer_relevancy,
                    "relevancy_reason": deepeval.relevancy_reason,
                    "faithfulness_score": deepeval.faithfulness,
                    "faithfulness_reason": deepeval.faithfulness_reason,
                    "deepeval_score": deepeval_score,
                    "deepeval_unmeasured": deepeval_score is None,
                    "overall_score": overall,
                }
            )
        return results

    def run_full_eval(self) -> EvalReport:
        print("[EvalSuite] Starting full evaluation")
        total = len(APM_TEST_SCENARIOS) + len(LOG_TEST_SCENARIOS)
        planned = len(self._cases(APM_TEST_SCENARIOS)) + len(self._cases(LOG_TEST_SCENARIOS))
        print(f"[EvalSuite] dataset={DATASET_VERSION} prompt={APM_PROMPT_VERSION} "
              f"analysis={OLLAMA_MODEL} judge={EVAL_JUDGE_MODEL} mode={self.judge_mode}")
        if planned < total:
            print(f"[EvalSuite] Sampling {planned}/{total} cases (seed {self.sample_seed}) - scores cover the sample only")
        else:
            print(f"[EvalSuite] Running all {total} cases (~{total * 4} LLM calls)")
        if EVAL_JUDGE_MODEL == OLLAMA_MODEL:
            print("[EvalSuite] WARNING: judge == model under test (self-grading). Set EVAL_JUDGE_MODEL to separate them.")
        report = EvalReport()
        report.apm_results = self.run_apm_eval()
        report.log_results = self.run_log_eval()

        all_scores = [r["overall_score"] for r in report.apm_results] + [r["overall_score"] for r in report.log_results]
        report.overall_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
        report.quality_gate = evaluate_quality_gate(report)
        report.ollama_version = get_ollama_version()
        report.analysis_digest = get_model_digest(report.analysis_model)
        report.judge_digest = get_model_digest(report.judge_model)
        report.cases_available = total
        report.sample_seed = self.sample_seed
        report.judge_mode = self.judge_mode

        failed_calls = sum(1 for r in report.apm_results + report.log_results if r.get("llm_failed"))
        if failed_calls:
            print(f"[EvalSuite] WARNING: {failed_calls} case(s) fell back after LLM failure; excluded from judge metrics.")

        print(f"[EvalSuite] Overall score: {report.overall_score:.0%}")
        print(f"[EvalSuite] Quality gate: {report.quality_gate.status}")
        if report.quality_gate.unmeasured_rules:
            print(f"[EvalSuite] Unmeasured metrics: {', '.join(report.quality_gate.unmeasured_rules)}")
        halluc = summarize_hallucination(report)
        if halluc["rate"] is not None:
            print(f"[EvalSuite] Hallucination rate: {halluc['rate']:.0%} "
                  f"({halluc['cases_with_hallucination']}/{halluc['cases_measurable']} cases) "
                  f"- derived from cause/action grounding, not gated")

        if report.quality_gate.insufficient_sample_rules:
            print(f"[EvalSuite] Thin-sample metrics: {', '.join(report.quality_gate.insufficient_sample_rules)}")
        return report


def save_eval_report_json(report: EvalReport, output_path: str = "reports/eval_result.json") -> dict:
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": report.timestamp,
        "run_metadata": {
            "dataset_version": report.dataset_version,
            "analysis_model": report.analysis_model,
            "judge_model": report.judge_model,
            "analysis_digest": report.analysis_digest,
            "judge_digest": report.judge_digest,
            "judge_mode": report.judge_mode,
            "prompt_version": report.prompt_version,
            "cases_evaluated": len(report.apm_results) + len(report.log_results),
            "cases_available": report.cases_available,
            "sample_seed": report.sample_seed,
            "self_grading": report.analysis_model == report.judge_model,
            "ollama_version": report.ollama_version,
        },
        "overall_score": report.overall_score,
        "quality_gate": asdict(report.quality_gate) if report.quality_gate else None,
        "by_scenario": summarize_by_scenario(report),
        "hallucination": summarize_hallucination(report),
        "apm_eval": report.apm_results,
        "log_eval": report.log_results,
    }
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    print(f"[EvalSuite] Saved report JSON: {output_path}")

    history_path = _append_history(data, output_path)
    if history_path:
        print(f"[EvalSuite] Appended history: {history_path}")
    return data


def _append_history(data: dict, output_path: str) -> str | None:
    """
    Keep every run under reports/history/ so scores can be compared over time.
    The latest-run file is overwritten on each run and would otherwise be lost.
    """
    import os

    try:
        stamp = str(data.get("timestamp", "")).replace("-", "").replace(":", "").replace(" ", "_")
        history_dir = os.path.join(os.path.dirname(output_path) or ".", "history")
        os.makedirs(history_dir, exist_ok=True)
        prompt_version = (data.get("run_metadata") or {}).get("prompt_version") or "na"
        history_path = os.path.join(history_dir, f"eval_result_{stamp}_prompt-{prompt_version}.json")
        with open(history_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        return history_path
    except OSError as exc:
        print(f"[EvalSuite] History write skipped: {exc}")
        return None
