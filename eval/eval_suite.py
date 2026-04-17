"""
DeepEval 기반 AI 품질 평가 Suite
APM 분석 AI와 로그 분류 AI의 품질을 자동으로 측정

평가 메트릭:
1. Hallucination  - AI가 APM 데이터에 없는 말을 지어냈나?
2. AnswerRelevancy - 장애 상황에 맞는 답변인가?
3. Faithfulness   - 주어진 데이터 기반으로 답했나?
4. 커스텀: 분류 정확도, 완결성
"""
import json
from datetime import datetime
from dataclasses import dataclass, field

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_ollama import ChatOllama

from apm.ai_analyzer import APMAIAnalyzer
from logwatch.ai_classifier import LogAIClassifier
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


# ── Ollama → DeepEval 연동 커스텀 모델 ───────────────────────────────────────
class OllamaEvalModel(DeepEvalBaseLLM):
    """
    DeepEval이 Ollama 모델을 사용해서 채점할 수 있도록 연결
    DeepEval 메트릭이 기본적으로 OpenAI를 쓰는 걸 Ollama로 교체
    """

    def __init__(self):
        self._llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,  # 채점은 일관성이 중요하므로 0
        )

    def load_model(self):
        return self._llm

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"ollama/{OLLAMA_MODEL}"


# ── APM 평가 시나리오 ─────────────────────────────────────────────────────────
APM_TEST_SCENARIOS = [
    {
        "name": "CPU 폭증 장애",
        "context": """
[APM 모니터링 이상 감지 - 2026-04-16 10:25:00]
서버: juntion_9100
감지된 이상 항목:
  - CPU 사용률 92.5% (임계값: 80%)
  - 응답시간 3200ms (임계값: 2000ms)
  - 액티브 트랜잭션: 38건
전체 지표:
  - CPU 사용률: 92.5%
  - 메모리 사용률: 65.0%
  - DB 커넥션: 12/50
  - 에러율: 0.5%
""",
        "expected_keywords": ["CPU", "프로세스", "스레드", "조치"],
        "expected_fault_types": ["CPU 폭증", "CPU 과부하", "복합"],
    },
    {
        "name": "DB 커넥션 풀 고갈",
        "context": """
[APM 모니터링 이상 감지 - 2026-04-16 10:30:00]
서버: docker_재무_12001
감지된 이상 항목:
  - DB 커넥션 48/50 (96%, 임계값: 80%)
  - 응답시간 6500ms (임계값: 2000ms)
  - 에러율 18.5% (임계값: 5%)
전체 지표:
  - CPU 사용률: 45.0%
  - DB 커넥션: 48/50
  - 에러율: 18.5%
""",
        "expected_keywords": ["DB", "커넥션", "풀", "조치"],
        "expected_fault_types": ["DB 커넥션", "커넥션 풀", "복합"],
    },
    {
        "name": "메모리 누수",
        "context": """
[APM 모니터링 이상 감지 - 2026-04-16 11:00:00]
서버: Was_Agent_1
감지된 이상 항목:
  - 메모리 사용률 93.2% (임계값: 90%)
  - 힙 메모리 955MB/1024MB (93%)
  - 응답시간 1850ms
전체 지표:
  - CPU 사용률: 35.0%
  - 메모리 사용률: 93.2%
  - 힙 메모리: 955MB/1024MB
""",
        "expected_keywords": ["메모리", "힙", "GC", "누수"],
        "expected_fault_types": ["메모리 누수", "메모리 부족", "복합"],
    },
]

# ── 로그 평가 시나리오 ────────────────────────────────────────────────────────
LOG_TEST_SCENARIOS = [
    {
        "name": "네트워크 연결 오류",
        "context": """
[로그 모듈 오류 감지 - 2026-04-16 10:26:00]
모듈: collector_agent_03
감지된 키워드: ERROR
오류 로그:
  ERROR [DataCollector] Connection refused to target host: 10.10.4.83:3306
최근 로그 컨텍스트:
  INFO [DataCollector] Collection cycle starting...
  ERROR [DataCollector] Connection refused to target host: 10.10.4.83:3306
""",
        "expected_error_types": ["네트워크", "연결 오류", "Connection"],
    },
    {
        "name": "OOM 에러",
        "context": """
[로그 모듈 오류 감지 - 2026-04-16 11:15:00]
모듈: data_forwarder_01
감지된 키워드: Exception
오류 로그:
  Exception in thread 'collector-worker-3' java.lang.OutOfMemoryError: Java heap space
최근 로그 컨텍스트:
  WARN [MetricForwarder] Memory usage high: 88%
  Exception in thread 'collector-worker-3' java.lang.OutOfMemoryError: Java heap space
""",
        "expected_error_types": ["메모리", "OOM", "OutOfMemory", "힙"],
    },
]


@dataclass
class DeepEvalMetrics:
    hallucination: float = 0.0      # 환각 점수 (낮을수록 좋음 → 1-score로 변환)
    answer_relevancy: float = 0.0   # 관련성 점수
    faithfulness: float = 0.0       # 근거 기반 점수
    hallucination_reason: str = ""
    relevancy_reason: str = ""
    faithfulness_reason: str = ""


@dataclass
class EvalReport:
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    apm_results: list[dict] = field(default_factory=list)
    log_results: list[dict] = field(default_factory=list)
    overall_score: float = 0.0


class AIQualityEvaluator:
    """
    DeepEval 메트릭 기반 AI 품질 평가 시스템

    평가 레이어 2단계:
    1단계 (커스텀): 분류 정확도, 키워드 포함율, 완결성
    2단계 (DeepEval): Hallucination, AnswerRelevancy, Faithfulness
    """

    def __init__(self):
        self.apm_analyzer = APMAIAnalyzer()
        self.log_classifier = LogAIClassifier()
        self._eval_model = OllamaEvalModel()

    def _run_deepeval_metrics(self, question: str, answer: str, context: str) -> DeepEvalMetrics:
        """DeepEval 메트릭 3종 실행"""
        metrics = DeepEvalMetrics()

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=[context],
            context=[context],
        )

        try:
            # 1. Hallucination - 데이터에 없는 말 지어냈나?
            hallucination_metric = HallucinationMetric(
                threshold=0.5,
                model=self._eval_model,
                include_reason=True,
            )
            hallucination_metric.measure(test_case)
            # hallucination score는 낮을수록 좋음 (0=환각없음, 1=환각많음)
            # 1 - score 로 변환해서 높을수록 좋게 만들기
            metrics.hallucination = round(1.0 - (hallucination_metric.score or 0.0), 2)
            metrics.hallucination_reason = hallucination_metric.reason or ""
        except Exception as e:
            print(f"    [Hallucination 측정 실패] {e}")
            metrics.hallucination = -1.0

        try:
            # 2. AnswerRelevancy - 질문에 맞는 답인가?
            relevancy_metric = AnswerRelevancyMetric(
                threshold=0.5,
                model=self._eval_model,
                include_reason=True,
            )
            relevancy_metric.measure(test_case)
            metrics.answer_relevancy = round(relevancy_metric.score or 0.0, 2)
            metrics.relevancy_reason = relevancy_metric.reason or ""
        except Exception as e:
            print(f"    [AnswerRelevancy 측정 실패] {e}")
            metrics.answer_relevancy = -1.0

        try:
            # 3. Faithfulness - 주어진 데이터 기반으로 답했나?
            faithfulness_metric = FaithfulnessMetric(
                threshold=0.5,
                model=self._eval_model,
                include_reason=True,
            )
            faithfulness_metric.measure(test_case)
            metrics.faithfulness = round(faithfulness_metric.score or 0.0, 2)
            metrics.faithfulness_reason = faithfulness_metric.reason or ""
        except Exception as e:
            print(f"    [Faithfulness 측정 실패] {e}")
            metrics.faithfulness = -1.0

        return metrics

    def run_apm_eval(self) -> list[dict]:
        results = []
        for scenario in APM_TEST_SCENARIOS:
            print(f"  [APM Eval] {scenario['name']} 평가 중...")
            analysis = self.apm_analyzer.analyze(scenario["context"])
            response_text = analysis.raw_response.lower()

            # ── 커스텀 메트릭 ──────────────────────────────────────────
            keyword_hits = sum(
                1 for kw in scenario["expected_keywords"]
                if kw.lower() in response_text
            )
            keyword_score = round(keyword_hits / len(scenario["expected_keywords"]), 2)

            fault_matched = any(
                ft.lower() in analysis.fault_type.lower() or ft.lower() in response_text
                for ft in scenario["expected_fault_types"]
            )

            completeness_score = round(sum([
                1 if analysis.root_cause and len(analysis.root_cause) > 10 else 0,
                1 if analysis.immediate_action and len(analysis.immediate_action) > 10 else 0,
                1 if analysis.prevention and len(analysis.prevention) > 10 else 0,
            ]) / 3, 2)

            # ── DeepEval 메트릭 ────────────────────────────────────────
            print(f"    → DeepEval 메트릭 측정 중...")
            deepeval = self._run_deepeval_metrics(
                question=f"다음 APM 장애 상황의 원인과 조치방안을 분석해주세요:\n{scenario['context']}",
                answer=analysis.raw_response,
                context=scenario["context"],
            )

            # ── 종합 점수 (커스텀 50% + DeepEval 50%) ─────────────────
            custom_score = (keyword_score * 0.4 + (1.0 if fault_matched else 0.0) * 0.3 + completeness_score * 0.3)

            valid_deepeval = [s for s in [deepeval.hallucination, deepeval.answer_relevancy, deepeval.faithfulness] if s >= 0]
            deepeval_score = round(sum(valid_deepeval) / len(valid_deepeval), 2) if valid_deepeval else custom_score

            overall = round((custom_score * 0.5 + deepeval_score * 0.5), 2)

            results.append({
                "scenario": scenario["name"],
                "fault_type": analysis.fault_type,
                "fault_type_correct": fault_matched,
                "keyword_score": keyword_score,
                "completeness_score": completeness_score,
                "custom_score": round(custom_score, 2),
                # DeepEval 메트릭
                "hallucination_score": deepeval.hallucination,
                "hallucination_reason": deepeval.hallucination_reason,
                "relevancy_score": deepeval.answer_relevancy,
                "relevancy_reason": deepeval.relevancy_reason,
                "faithfulness_score": deepeval.faithfulness,
                "faithfulness_reason": deepeval.faithfulness_reason,
                "deepeval_score": deepeval_score,
                "overall_score": overall,
                "severity": analysis.severity,
            })

        return results

    def run_log_eval(self) -> list[dict]:
        results = []
        for scenario in LOG_TEST_SCENARIOS:
            print(f"  [Log Eval] {scenario['name']} 평가 중...")
            classification = self.log_classifier.classify(scenario["context"])
            response_text = classification.raw_response.lower()

            # ── 커스텀 메트릭 ──────────────────────────────────────────
            type_matched = any(
                et.lower() in classification.error_type.lower() or et.lower() in response_text
                for et in scenario["expected_error_types"]
            )
            action_present = len(classification.recommended_action) > 10

            # ── DeepEval 메트릭 ────────────────────────────────────────
            print(f"    → DeepEval 메트릭 측정 중...")
            deepeval = self._run_deepeval_metrics(
                question=f"다음 로그 에러를 분석하고 에러 유형과 조치방안을 알려주세요:\n{scenario['context']}",
                answer=classification.raw_response,
                context=scenario["context"],
            )

            custom_score = (
                (1.0 if type_matched else 0.0) * 0.5 +
                (1.0 if action_present else 0.0) * 0.3 +
                (1.0 if classification.severity in ["높음", "중간", "낮음"] else 0.0) * 0.2
            )

            valid_deepeval = [s for s in [deepeval.hallucination, deepeval.answer_relevancy, deepeval.faithfulness] if s >= 0]
            deepeval_score = round(sum(valid_deepeval) / len(valid_deepeval), 2) if valid_deepeval else custom_score

            overall = round((custom_score * 0.5 + deepeval_score * 0.5), 2)

            results.append({
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
            })

        return results

    def run_full_eval(self) -> EvalReport:
        print("\n[EvalSuite] AI 품질 평가 시작...")
        print("[EvalSuite] 평가 메트릭: Hallucination / AnswerRelevancy / Faithfulness + 커스텀\n")

        report = EvalReport()

        print("── APM 분석 AI 평가 ──────────────────")
        report.apm_results = self.run_apm_eval()

        print("\n── 로그 분류 AI 평가 ─────────────────")
        report.log_results = self.run_log_eval()

        all_scores = (
            [r["overall_score"] for r in report.apm_results] +
            [r["overall_score"] for r in report.log_results]
        )
        report.overall_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0

        print(f"\n[EvalSuite] 완료 - 전체 품질 점수: {report.overall_score:.0%}")
        return report


def save_eval_report_json(report: EvalReport, output_path: str = "reports/eval_result.json"):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "timestamp": report.timestamp,
        "overall_score": report.overall_score,
        "apm_eval": report.apm_results,
        "log_eval": report.log_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[EvalSuite] 결과 저장: {output_path}")
    return data
