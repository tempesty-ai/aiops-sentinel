"""
DeepEval-based AI quality evaluation suite.
Includes a quality gate with pass/fail criteria for portfolio-friendly QA evidence.
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_ollama import ChatOllama

from apm.ai_analyzer import APMAIAnalyzer
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from logwatch.ai_classifier import LogAIClassifier

SCHEMA_VERSION = "1.1.0"

QUALITY_GATE_THRESHOLDS = {
    "overall_score_min": 0.70,
    "apm_fault_type_accuracy_min": 0.67,
    "log_error_type_accuracy_min": 0.50,
    "hallucination_min": 0.65,
    "relevancy_min": 0.60,
    "faithfulness_min": 0.60,
}


class OllamaEvalModel(DeepEvalBaseLLM):
    def __init__(self):
        self._llm = ChatOllama(
            model=OLLAMA_MODEL,
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
        return f"ollama/{OLLAMA_MODEL}"


APM_TEST_SCENARIOS = [
    {
        "name": "CPU spike incident",
        "context": """
[APM anomaly]
server=was_sample_01
cpu=92.5%
response_time_ms=3200
db_connections=12/50
error_rate=0.5%
""".strip(),
        "expected_keywords": ["cpu", "response", "action"],
        "expected_fault_types": ["cpu", "bottleneck", "unknown"],
    },
    {
        "name": "DB pool saturation",
        "context": """
[APM anomaly]
server=docker_service_12001
cpu=45.0%
response_time_ms=6500
db_connections=48/50
error_rate=18.5%
""".strip(),
        "expected_keywords": ["db", "connection", "action"],
        "expected_fault_types": ["db", "connection", "unknown"],
    },
    {
        "name": "Memory pressure",
        "context": """
[APM anomaly]
server=Was_Agent_1
memory=93.2%
heap=955/1024MB
response_time_ms=1850
""".strip(),
        "expected_keywords": ["memory", "heap", "action"],
        "expected_fault_types": ["memory", "leak", "unknown"],
    },
]

LOG_TEST_SCENARIOS = [
    {
        "name": "Connection refused",
        "context": """
[Log error]
module=agent_sample_a
line=ERROR Connection refused to target host: 10.10.4.83:3306
""".strip(),
        "expected_error_types": ["network", "connection", "refused"],
    },
    {
        "name": "OutOfMemoryError",
        "context": """
[Log error]
module=data_forwarder_01
line=Exception java.lang.OutOfMemoryError: Java heap space
""".strip(),
        "expected_error_types": ["memory", "oom", "outofmemory"],
    },
]


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
    score: float
    failed_rules: list[str]
    metrics: dict[str, float]
    thresholds: dict[str, float]


@dataclass
class EvalReport:
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    apm_results: list[dict] = field(default_factory=list)
    log_results: list[dict] = field(default_factory=list)
    overall_score: float = 0.0
    quality_gate: QualityGateResult | None = None


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def evaluate_quality_gate(report: EvalReport) -> QualityGateResult:
    apm_accuracy = _avg([1.0 if r.get("fault_type_correct") else 0.0 for r in report.apm_results])
    log_accuracy = _avg([1.0 if r.get("error_type_correct") else 0.0 for r in report.log_results])

    all_results = report.apm_results + report.log_results
    hallucination = _avg([r.get("hallucination_score", -1) for r in all_results if r.get("hallucination_score", -1) >= 0])
    relevancy = _avg([r.get("relevancy_score", -1) for r in all_results if r.get("relevancy_score", -1) >= 0])
    faithfulness = _avg([r.get("faithfulness_score", -1) for r in all_results if r.get("faithfulness_score", -1) >= 0])

    metrics = {
        "overall_score": round(report.overall_score, 2),
        "apm_fault_type_accuracy": apm_accuracy,
        "log_error_type_accuracy": log_accuracy,
        "hallucination": hallucination,
        "relevancy": relevancy,
        "faithfulness": faithfulness,
    }

    failed_rules: list[str] = []
    if metrics["overall_score"] < QUALITY_GATE_THRESHOLDS["overall_score_min"]:
        failed_rules.append("overall_score")
    if metrics["apm_fault_type_accuracy"] < QUALITY_GATE_THRESHOLDS["apm_fault_type_accuracy_min"]:
        failed_rules.append("apm_fault_type_accuracy")
    if metrics["log_error_type_accuracy"] < QUALITY_GATE_THRESHOLDS["log_error_type_accuracy_min"]:
        failed_rules.append("log_error_type_accuracy")
    if metrics["hallucination"] < QUALITY_GATE_THRESHOLDS["hallucination_min"]:
        failed_rules.append("hallucination")
    if metrics["relevancy"] < QUALITY_GATE_THRESHOLDS["relevancy_min"]:
        failed_rules.append("relevancy")
    if metrics["faithfulness"] < QUALITY_GATE_THRESHOLDS["faithfulness_min"]:
        failed_rules.append("faithfulness")

    passed_checks = 6 - len(failed_rules)
    score = round(passed_checks / 6, 2)

    return QualityGateResult(
        passed=(len(failed_rules) == 0),
        score=score,
        failed_rules=failed_rules,
        metrics=metrics,
        thresholds=QUALITY_GATE_THRESHOLDS,
    )


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
            fault_matched = any(
                ft.lower() in analysis.fault_type.lower() or ft.lower() in response_text
                for ft in scenario["expected_fault_types"]
            )
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
            valid_deepeval = [s for s in [deepeval.hallucination, deepeval.answer_relevancy, deepeval.faithfulness] if s >= 0]
            deepeval_score = round(sum(valid_deepeval) / len(valid_deepeval), 2) if valid_deepeval else round(custom_score, 2)
            overall = round((custom_score * 0.5 + deepeval_score * 0.5), 2)

            results.append(
                {
                    "scenario": scenario["name"],
                    "fault_type": analysis.fault_type,
                    "fault_type_correct": fault_matched,
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

            type_matched = any(
                expected.lower() in classification.error_type.lower() or expected.lower() in response_text
                for expected in scenario["expected_error_types"]
            )
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
            valid_deepeval = [s for s in [deepeval.hallucination, deepeval.answer_relevancy, deepeval.faithfulness] if s >= 0]
            deepeval_score = round(sum(valid_deepeval) / len(valid_deepeval), 2) if valid_deepeval else round(custom_score, 2)
            overall = round((custom_score * 0.5 + deepeval_score * 0.5), 2)

            results.append(
                {
                    "scenario": scenario["name"],
                    "error_type": classification.error_type,
                    "error_type_correct": type_matched,
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
                    "overall_score": overall,
                }
            )
        return results

    def run_full_eval(self) -> EvalReport:
        print("[EvalSuite] Starting full evaluation")
        report = EvalReport()
        report.apm_results = self.run_apm_eval()
        report.log_results = self.run_log_eval()

        all_scores = [r["overall_score"] for r in report.apm_results] + [r["overall_score"] for r in report.log_results]
        report.overall_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
        report.quality_gate = evaluate_quality_gate(report)

        print(f"[EvalSuite] Overall score: {report.overall_score:.0%}")
        print(f"[EvalSuite] Quality gate: {'PASS' if report.quality_gate.passed else 'FAIL'}")
        return report


def save_eval_report_json(report: EvalReport, output_path: str = "reports/eval_result.json") -> dict:
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": report.timestamp,
        "overall_score": report.overall_score,
        "quality_gate": asdict(report.quality_gate) if report.quality_gate else None,
        "apm_eval": report.apm_results,
        "log_eval": report.log_results,
    }
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    print(f"[EvalSuite] Saved report JSON: {output_path}")
    return data
