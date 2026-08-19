"""
DeepEval-based AI quality evaluation suite.
Includes a quality gate with pass/fail criteria for portfolio-friendly QA evidence.
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_ollama import ChatOllama

from apm.ai_analyzer import APMAIAnalyzer
from config.settings import APM_PROMPT_VERSION, EVAL_JUDGE_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL, get_ollama_version
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
    "hallucination_min": 0.65,
    "relevancy_min": 0.60,
    "faithfulness_min": 0.60,
}


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

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"


GOLDEN_DATASET_PATH = Path(__file__).parent / "datasets" / "golden_v1.json"


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> dict:
    """
    Load the labeled golden set from disk.

    The dataset lives outside the code so labels can be reviewed and versioned
    independently of scoring logic.
    """
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


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
    ollama_version: str = ""


def _avg(values: list[float]) -> float | None:
    """Mean of the sample, or None when there is no sample at all.

    Returning None keeps "never measured" distinguishable from "measured as 0.0";
    collapsing the two used to fail the gate on a judge-model outage.
    """
    return round(sum(values) / len(values), 2) if values else None


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
        "log_error_type_accuracy": [1.0 if r.get("error_type_correct") else 0.0 for r in log_scored],
        "hallucination": judged("hallucination_score"),
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
        "log_error_type_accuracy": len(report.log_results),
        "hallucination": len(answered),
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
    def __init__(self):
        self.apm_analyzer = APMAIAnalyzer()
        self.log_classifier = LogAIClassifier()
        self._eval_model = OllamaEvalModel()

    def _run_deepeval_metrics(self, question: str, answer: str, context: str) -> DeepEvalMetrics:
        metrics = DeepEvalMetrics()
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=[context],
            context=[context],
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
        for scenario in APM_TEST_SCENARIOS:
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

            deepeval = self._run_deepeval_metrics(
                question=f"Analyze root cause and action for:\n{scenario['context']}",
                answer=analysis.raw_response,
                context=scenario["context"],
            )

            custom_score = keyword_score * 0.4 + (1.0 if fault_matched else 0.0) * 0.3 + completeness_score * 0.3
            deepeval_score, overall = _blend_scores(custom_score, deepeval)

            results.append(
                {
                    "case_id": scenario["case_id"],
                    "scenario": scenario["name"],
                    "answer_excerpt": analysis.raw_response[:400],
                    "fault_type": analysis.fault_type,
                    "fault_type_correct": fault_matched,
                    "fault_type_correct_loose": fault_matched_loose,
                    "classification_unmeasured": classification_unmeasured,
                    "llm_failed": analysis.llm_failed,
                    "keyword_score": keyword_score,
                    "completeness_score": completeness_score,
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
        for scenario in LOG_TEST_SCENARIOS:
            print(f"[Log Eval] {scenario['name']}")
            classification = self.log_classifier.classify(scenario["context"])
            response_text = classification.raw_response.lower()

            type_matched, type_matched_loose = _match_label(
                scenario["expected_error_types"], classification.error_type, classification.raw_response
            )
            classification_unmeasured = classification.llm_failed or classification.error_type_missing
            action_present = len(classification.recommended_action) > 10

            deepeval = self._run_deepeval_metrics(
                question=f"Classify error and action:\n{scenario['context']}",
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
        print(f"[EvalSuite] dataset={DATASET_VERSION} prompt={APM_PROMPT_VERSION} analysis={OLLAMA_MODEL} judge={EVAL_JUDGE_MODEL}")
        if EVAL_JUDGE_MODEL == OLLAMA_MODEL:
            print("[EvalSuite] WARNING: judge == model under test (self-grading). Set EVAL_JUDGE_MODEL to separate them.")
        report = EvalReport()
        report.apm_results = self.run_apm_eval()
        report.log_results = self.run_log_eval()

        all_scores = [r["overall_score"] for r in report.apm_results] + [r["overall_score"] for r in report.log_results]
        report.overall_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
        report.quality_gate = evaluate_quality_gate(report)
        report.ollama_version = get_ollama_version()

        failed_calls = sum(1 for r in report.apm_results + report.log_results if r.get("llm_failed"))
        if failed_calls:
            print(f"[EvalSuite] WARNING: {failed_calls} case(s) fell back after LLM failure; excluded from judge metrics.")

        print(f"[EvalSuite] Overall score: {report.overall_score:.0%}")
        print(f"[EvalSuite] Quality gate: {report.quality_gate.status}")
        if report.quality_gate.unmeasured_rules:
            print(f"[EvalSuite] Unmeasured metrics: {', '.join(report.quality_gate.unmeasured_rules)}")
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
            "prompt_version": report.prompt_version,
            "self_grading": report.analysis_model == report.judge_model,
            "ollama_version": report.ollama_version,
        },
        "overall_score": report.overall_score,
        "quality_gate": asdict(report.quality_gate) if report.quality_gate else None,
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
