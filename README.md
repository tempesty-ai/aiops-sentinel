# AIOps Sentinel - APM 모니터링, LLM 분석, AI 출력 품질 평가

> 비정상 APM 지표를 감지하고, LLM에 원인 분석을 요청한 뒤, 그 AI 분석 결과의 품질까지 평가하는 실험 프로젝트입니다.
> 핵심 QA 질문은 "AI가 만든 결과는 누가 검증하는가?"입니다.

## 데이터 안내

이 저장소의 APM 데이터, 호스트명, WAS명, 에이전트명, 기준값, 알림, 예시는 모두 100% 가상 샘플 데이터입니다.

- 데이터는 `apm/mock_generator.py`에서 생성합니다.
- `was_sample_01`, `agent_sample_a` 같은 식별자는 문서와 테스트용 가상 이름입니다.
- CPU 80%, DB connection 80% 같은 기준값은 일반적인 모니터링 예시이며 회사나 고객사의 운영 기준이 아닙니다.
- 이 저장소에는 상용 APM 제품의 내부 구조, 데이터 스키마, 운영 비밀 정보, 운영 데이터가 포함되어 있지 않습니다.

## 이 프로젝트가 다루는 문제

| # | 문제 | 접근 방식 |
| --- | --- | --- |
| 1 | 비정상 APM 지표가 발생할 때 반복되는 수동 분석 | 기준값 기반 감지와 LangChain/Ollama 분석 |
| 2 | AI 분석 결과 자체의 품질 검증 | DeepEval 기반 Hallucination, Faithfulness, Relevancy 측정 |

두 번째 항목이 이 프로젝트의 핵심입니다. 단순히 LLM을 사용하는 것보다, LLM 출력 주변에 QA 메커니즘을 붙이는 데 초점을 둡니다.

## 시스템 개요

```text
AIOps Sentinel
- 기능 1: APM
  - Mock APM 데이터
  - 기준값 기반 이상 감지
  - AI 원인 분석
- 기능 2: Log Watch
  - Mock 로그 시뮬레이터
  - 여러 로그 파일 tailing
  - AI 오류 분류
- 출력
  - Mattermost Incoming Webhook 알림
  - HTML 리포트
  - Eval Suite 품질 점수
```

## 기술 스택

| 영역 | 기술 |
| --- | --- |
| 언어 | Python 3.11+ |
| AI/LLM | LangChain + Ollama (`llama3.1:8b`) |
| 알림 | Mattermost Incoming Webhook |
| 품질 평가 | DeepEval |
| 데이터 | 직접 생성한 Mock APM 데이터 |

## 빠른 시작

```bash
ollama pull llama3.1:8b
pip install -r requirements.txt
cp .env.example .env
```

Webhook 전송이 필요하면 `.env`에 `MATTERMOST_WEBHOOK_URL`을 설정합니다.

```bash
python main.py
python main.py --eval
python main.py --report
```

## 샘플 알림

아래 예시는 문서용 가상 샘플입니다.

```text
[Critical] APM anomaly detected - was_sample_01

Metrics
- CPU usage 92.5% (threshold 80%)
- DB connections 48/50 (96%, threshold 80%)

AI analysis
- Type: DB connection pool exhaustion
- Cause: Slow queries holding connections
- Action: Restart connection pool and inspect slow queries
```

```text
[High] Collector module error - agent_sample_a

Log
ERROR [DataCollector] Connection refused to target host

AI classification
- Type: Network connection error
- Repeatability: Persistent
- Action: Check target host network connectivity
```

## AI 품질 평가

DeepEval을 사용해 LLM 출력이 제공된 모니터링 데이터에 근거하고 있는지 측정합니다.

| 지표 | 목적 |
| --- | --- |
| Hallucination | 답변이 근거 없는 사실을 만들어내는지 확인 |
| Answer Relevancy | 답변이 이상 상황 맥락에 맞는지 확인 |
| Faithfulness | 답변이 모니터링 데이터에 충실한지 확인 |
| Custom score | 기대 이상 유형, 키워드 포함률, 답변 완성도를 조합 |
| Overall quality score | Custom 지표와 DeepEval 지표의 가중 평균 |

QA 관점에서 중요한 점은 그럴듯한 AI 출력이 자동으로 정답이 되지는 않는다는 것입니다. 이 프로젝트는 그 간극을 보이게 만들고 측정합니다.

## 프로젝트 구조

```text
aiops-sentinel/
- apm/
  - mock_generator.py
  - anomaly_detector.py
  - ai_analyzer.py
- logwatch/
  - log_simulator.py
  - log_tailer.py
  - ai_classifier.py
- alert/
  - mattermost.py
- eval/
  - eval_suite.py
  - report_generator.py
- config/
  - settings.py
- main.py
```

## 한계

- 운영용 AIOps 솔루션이 아닙니다.
- 모니터링 데이터는 Mock 데이터이며, 기준값 로직은 의도적으로 단순하게 구성했습니다.
- 선택한 LLM 모델은 학습과 실험용이며, 검증된 운영 추천 모델이 아닙니다.
- DeepEval 점수는 절대적인 진실이 아닙니다. 샘플은 여전히 사람이 검토해야 합니다.
- Mattermost 알림은 단방향입니다. 사용자 피드백은 아직 모델 선택이나 평가에 다시 반영되지 않습니다.

## 로드맵

- README에 실제 측정된 DeepEval 결과 추가
- 동일 이상 케이스에 대해 여러 LLM 모델 비교
- 알림 유용성에 대한 사용자 피드백 루프 추가
- 같은 평가 접근을 형제 프로젝트 `botserver`에 적용
