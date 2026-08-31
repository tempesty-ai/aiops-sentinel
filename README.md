# 🛡️ `aiops-sentinel` — AIOps + AI 출력 품질 평가

> APM 장애 지능형 분석 + 수집 모듈 로그 감시 + Mattermost 알람 + **AI 품질 평가(DeepEval)**.
> 핵심은 "AI에게 장애 분석을 시키는 것"이 아니라, **"그 분석을 운영에 써도 되는지 수치로 게이트하는 것"** 입니다.

---

## 한 줄 가치

> **"AI가 만든 장애 분석을 그대로 신뢰해도 되는가?"** 를 듀얼 트랙(운영 + 평가)으로 답하는 저장소.
> 다른 AIOps 도구가 "AI가 알아서 분석합니다"에서 멈추는 동안, 이 저장소는 그 분석을 **수치로 게이트**합니다.
>
> 102건 전수 측정 결과: **장애유형 분류는 98% 정확한데, 원인·조치의 65%가 관측 지표로 뒷받침되지 않았습니다.**
> 분류 정확도만 봤다면 운영 투입 가능으로 읽혔을 결과입니다.

---

## 개요

**AIOps Sentinel**은 APM(Application Performance Monitoring) 환경에서 발생하는 두 가지 핵심 문제를 AI로 자동화하고, 그 자동화의 **품질을 다시 측정**합니다.

1. **APM 장애 분석**: WAS 서버의 CPU/메모리/DB/응답시간 이상을 감지하고 AI가 원인 분석 및 조치 방안 생성
2. **수집 모듈 로그 감시**: 리눅스 수집 에이전트 로그를 실시간 tail하여 ERROR 발생 시 AI 분류 및 알람
3. **AI 품질 평가** (이 저장소의 차별점): 위 1·2의 AI 출력을 DeepEval로 정량 측정

---

## 발견한 크리티컬 리스크

| # | 리스크 | 의미 |
| --- | --- | --- |
| R1 | **장애 감지 지연** | WAS의 CPU/메모리/DB 커넥션/응답시간 이상이 임계값을 넘는 순간을 사람이 보지 못함 → 장애 인지 지연 |
| R2 | **수집 모듈 침묵** | 리눅스 에이전트 로그의 ERROR가 무시되어 데이터가 끊긴 줄도 모르는 상태가 누적됨 |
| R3 | **AI 분석 신뢰 (가장 중요)** | AI가 그럴듯한 원인 분석을 **만들어내지만(hallucination)**, 실제 메트릭과 무관하거나, 핵심 키워드가 빠지거나, 조치 방안이 없는 경우 → 잘못된 조치로 **장애를 키울 수 있음**<br/>→ `cause_grounding`(규칙) + `faithfulness`(LLM 심판)로 측정하고, 리포트에 **환각률**로 집계 |

→ R3가 이 저장소의 가장 차별적인 문제 인식입니다. "AI를 도입했더니 가이드가 틀려서 장애가 더 커졌다"를 막기 위해 **AI 출력 자체를 게이트** 합니다.

---

## 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│                    AIOps Sentinel                       │
├─────────────────────┬───────────────────────────────────┤
│   Feature 1: APM    │   Feature 2: Log Watch            │
│                     │                                   │
│  Mock APM 데이터    │  Log Simulator (mock)              │
│       ↓             │       ↓                            │
│  이상 감지          │  tail -f (다중 파일)                │
│  (임계값 기반)      │  ERROR 키워드 감지                  │
│       ↓             │       ↓                            │
│  AI 원인 분석       │  AI 에러 분류                       │
│  (LangChain+Ollama) │  (LangChain+Ollama)                │
│       ↓             │       ↓                            │
│       └─────────────┴─────────────┐                     │
│                                   ↓                     │
│                    Mattermost Incoming Webhook          │
│                    HTML 리포트 + Eval Suite              │
└─────────────────────────────────────────────────────────┘
```

---

## 테스트 설계 — 듀얼 트랙

### 트랙 1: 운영 자동화

- **APM 트랙**: Mock APM 데이터 → 임계값 기반 이상 감지 → LangChain + Ollama(`llama3.1:8b`)로 원인 분석 → Mattermost Webhook 알람
- **로그 트랙**: 수집 에이전트 로그 → `tail -f` 다중 파일 → ERROR 키워드 감지 → AI 분류 → Mattermost 알람

### 트랙 2: AI 출력 품질 평가

이 저장소의 핵심 차별점. **"AI의 답을 운영에 쓸 수 있는가"** 를 두 경로로 판정합니다.

#### 경로 A — 케이스 점수 (50:50)

케이스 1건의 점수는 규칙 채점과 심판 채점을 반반으로 합칩니다.

```
케이스 점수 = 규칙 채점 × 0.5  +  심판 채점 × 0.5
```

| 규칙 채점 (LLM 불필요) | 검증 의도 |
| --- | --- |
| 장애유형 분류 정확도 | 골든셋 정답 레이블과 일치하는가 |
| 키워드 포함율 | 필수 키워드가 응답에 있는가 |
| 응답 완결성 | 원인·조치·예방 섹션이 채워졌는가 (길이 기준) |

| 심판 채점 (LLM 심판) | 검증 의도 |
| --- | --- |
| **Faithfulness** | 주장이 관측된 지표로 뒷받침되는 비율 |
| **Answer Relevancy** | 요구된 과제에 답한 진술의 비율 |

> **왜 반반인가**: 두 채점이 잡는 결함이 다릅니다. 규칙은 재현성 100%지만 단어와 구조만 보고,
> 심판은 의미적 오류를 보지만 심판 모델이 실패하면 무방비입니다. 어느 쪽이 더 중요한지 검증할
> 데이터가 없어 균등 배분했고, 100:0부터 0:100까지 바꿔 계산해도 판정 결론이 불변임을
> 확인했습니다.

#### 경로 B — 품질 게이트 (7개 지표, AND 조건)

게이트는 경로 A의 점수를 그대로 쓰지 않습니다. 지표 7개를 **각각 자기 임계값과 개별 비교**하고,
하나라도 미달하면 전체 FAIL입니다.

| 지표 | 채점 방식 | 임계값 |
| --- | --- | --- |
| `overall_score` | 케이스 점수 평균 | 0.70 |
| `apm_fault_type_accuracy` | 규칙 | 0.67 |
| `log_error_type_accuracy` | 규칙 | 0.50 |
| `cause_grounding` | 규칙 — 원인 진단이 **실제로 이상 감지된 항목**을 지목했는가 | 0.80 |
| `action_grounding` | 규칙 — 조치가 **실제로 이상 감지된 항목**을 지목했는가 | 0.80 |
| `relevancy` | LLM 심판 | 0.60 |
| `faithfulness` | LLM 심판 | 0.60 |

판정은 3분류입니다 — `PASS` / `FAIL`(측정값이 기준 미달) / `INCONCLUSIVE`(측정 불가 또는 표본 부족).
**측정하지 못한 것을 품질 실패로 집계하지 않습니다.**

#### 환각 측정

`cause_grounding`이 잡는 것이 이 프로젝트가 겨냥한 환각입니다. 실제 검거 사례:

```
관측  cpu=92.5%,  db_connections=12/50   (DB는 감지 안 됨)
답변  "Insufficient database connections" → 커넥션을 75로 늘려라
      → cause_grounding 0.5,  action_grounding 0.5
      → DeepEval HallucinationMetric은 이 답변에 1.0 만점을 줬음
```

리포트에는 **환각률**이 함께 출력됩니다. 게이트 지표는 아니고 위 두 규칙 지표에서 파생한 집계입니다.

```
Hallucination rate: 33% - 20 of 60 cases asserted something the metrics
do not support (diagnosis 14, action 11). Most often: cpu x18, memory x9
```

> `Hallucination` 지표 자체는 게이트에서 제외했습니다. DeepEval 70b 심판에서 5건 전부 1.0으로
> 포화되어 명백한 오답에도 만점을 줬고, 자체 8b 심판으로도 3건 중 2건을 오판했습니다.
> 어느 채점기가 어느 축을 맡을지는 취향이 아니라 실측으로 정했습니다 —
> [QUALITY_GATE.md](QUALITY_GATE.md)의 *Who grades what* 참고.

#### 평가 데이터셋

```
eval/datasets/golden_v2.json     102건 (APM 60 + 로그 42), 버전 2.2.0
```

`build_golden.py`가 **운영 코드 경로 그대로** 생성합니다 — `MockAPMGenerator(시나리오 강제)` →
`AnomalyDetector` → `context_for_ai`. 주입한 장애 시나리오가 정답이므로 **정답이 구성상 확정**되고,
평가 입력과 운영 입력의 형식이 동일합니다. 같은 seed면 바이트 단위로 재생성됩니다.

#### 심판 모드

```bash
python main.py --eval           # 자체 심판: 케이스당 1회 호출  (기본)
python main.py --eval --deep    # DeepEval:  케이스당 9회 호출  (대조용, 훨씬 느림)
```

DeepEval은 지표마다 추출·판정·근거를 따로 호출합니다. 호스팅 API에는 합리적이지만 로컬 CPU에서는
102건에 918회 호출이라 실행 자체를 안 하게 됩니다. 자체 심판은 같은 질문을 구조화 출력 1회로
합쳐 **4.5배 빠릅니다**. `judge_mode`가 리포트에 기록되고 `compare_runs`가 이를 변수로 취급하므로,
두 모드 결과를 회귀로 착각하지 않습니다.

---

### 측정 결과 (102건 전수, 2026-08-25)

```
분석 llama3.1:8b  /  심판 llama3.1:8b (domain 모드)  /  데이터셋 2.2.0
소요 약 1시간 20분

품질 게이트   FAIL   (7개 중 5개 인증)
```

| 지표 | 값 | 표본 | 임계값 | |
| --- | --- | --- | --- | --- |
| `apm_fault_type_accuracy` | **0.98** | 60 | 0.67 | 통과 |
| `log_error_type_accuracy` | 0.76 | 42 | 0.50 | 통과 |
| `overall_score` | 0.78 | 102 | 0.70 | 통과 |
| `relevancy` | 0.67 | 95 | 0.60 | 통과 |
| `faithfulness` | 0.63 | 91 | 0.60 | 통과 |
| `cause_grounding` | **0.61** | 57 | 0.80 | **미달** |
| `action_grounding` | **0.56** | 45 | 0.80 | **미달** |

```
환각률 65%  (60건 중 39건)
  원인 단계 37건 / 조치 단계 25건
  근거 없이 지목된 리소스:  memory 61회,  cpu 59회,  db_connection 2회,  response_time 1회
```

#### 무엇이 드러났는가

**장애유형은 98% 맞히는데, 원인·조치에서 65%가 근거 없는 지목입니다.**
그리고 지목 대상이 압도적으로 `memory`·`cpu`입니다 — 감지된 항목이 DB 커넥션이든 에러율이든
**"서버 자원(CPU, 메모리)을 늘려라"로 수렴하는 상투적 답변**을 반복합니다.

분류만 봤다면 "98% 정확, 운영 투입 가능"으로 읽혔을 결과입니다. 정합성 검증이 없으면
**틀린 조치를 권고하는 답변이 높은 점수로 통과합니다.**

#### 표현 취약성

문구별 분해에서 로그 트랙의 취약점이 드러납니다.

| 에러 유형 | 정확도 | 문구별 (`p1`/`p2`/`p3`) |
| --- | --- | --- |
| `malformed_log` | **0%** | p1 0%, p2 0%, p3 0% |
| `http_503` | 33% | p1 100%, p2 0%, p3 0% |
| `socket_timeout` | 67% | p1 0%, p2 100%, p3 100% |
| `db_query_failed` | 67% | p1 100%, p2 100%, p3 0% |
| (APM 5종) | 92~100% | — |

> `p1`~`p3`은 **같은 에러의 다른 문구**입니다. 아래 프롬프트 `v1`/`v2`와는 다른 축입니다.

APM 5종은 전부 92% 이상인데 **로그 14종 중 7종이 문구에 따라 갈립니다.** `malformed_log`는
세 문구 모두 실패 — 유형 자체를 인식하지 못합니다. 집계값 67%만 보면 노이즈처럼 보이지만,
문구별로 나누면 **특정 표현에서만 매번 실패**하는 것이 보입니다.

### 프롬프트 A/B — v1 vs v2 (102건 전수, 2026-08-31)

게이트가 `cause_grounding` / `action_grounding`에서 미달했으므로, 그걸 겨냥한 프롬프트를 만들어
검증했습니다. `v2`는 `v1`에 네 줄만 더합니다. 섹션 구성은 동일합니다.

```
Ground every statement in the metrics you were given:
- 각 주장마다 근거 메트릭을 이름과 값으로 인용하라 ("cpu=92.5%" 처럼)
- 주어진 메트릭이 뒷받침하지 않는 원인을 단정하지 마라
- 입력에 없는 메트릭은 아예 추론하지 마라
```

데이터셋·분석 모델·심판 모델·모델 digest·심판 모드가 전부 동일하고 **프롬프트만 다릅니다.**
`compare_runs`가 변경된 변수를 확인하고 "이 차이는 프롬프트에 귀속 가능"이라고 판정했습니다.

| 지표 | v1 | v2 | 차이 | |
| --- | --- | --- | --- | --- |
| `action_grounding` | 0.56 | **0.69** | **+0.13** | 개선 |
| `faithfulness` | 0.63 | 0.68 | +0.05 | 개선 |
| `relevancy` | 0.67 | 0.68 | +0.01 | — |
| `overall_score` | 0.78 | 0.78 | 0.00 | — |
| `cause_grounding` | 0.61 | 0.61 | **0.00** | 변화 없음 |
| `log_error_type_accuracy` | 0.76 | 0.74 | −0.02 | **역행** |
| `apm_fault_type_accuracy` | 0.98 | **0.92** | **−0.06** | **역행** |

```
환각률   65% → 53%   (39건 → 32건)
  원인 단계  37 → 31
  조치 단계  25 → 16      ← 크게 줄어듦
  근거 없이 지목된 memory  61회 → 19회
```

#### 읽는 법

**조치 단계에는 들었고, 원인 진단에는 안 들었습니다.** `action_grounding`이 +0.13,
조치 단계 환각이 25건에서 16건으로 줄었습니다. 반면 `cause_grounding`은 0.61에서
**정확히 그대로**입니다. 같은 지시가 처방은 고쳤지만 진단은 못 고쳤습니다.

**그리고 대가가 있었습니다.** `apm_fault_type_accuracy`가 0.98에서 0.92로 떨어졌습니다.
60건 중 4건을 더 틀린 것으로, 근거 인용을 강제하자 장애유형을 보수적으로 답하면서
분류가 흐려진 것으로 보입니다. `compare_runs`가 자동으로 표시합니다.

```
Side effects: apm_fault_type_accuracy -0.06, log_error_type_accuracy -0.02
A gain in one metric paid for elsewhere is not an improvement.
```

**게이트는 여전히 FAIL입니다** — `cause_grounding` 0.61, `action_grounding` 0.69로
둘 다 기준 0.80 미만입니다. 프롬프트 한 줄로 지표가 올랐다는 것보다,
**어디에 듣고 어디에 안 들으며 무엇을 대가로 치르는지를 분리해 측정한 것**이 이 실험의 결과입니다.
지표 하나만 봤다면 "+0.13, 개선됨"으로 끝났을 기록입니다.

---

#### 남은 과제

1. `cause_grounding`은 `v2`로 움직이지 않았습니다. 원인 진단에 특화한 프롬프트를 더 시도할지,
   8b의 한계로 보고 기록할지 판단이 필요합니다.
2. `v2`의 분류 정확도 역행(−0.06)이 실제 손실인지 표본 변동인지 반복 실행으로 확인해야 합니다.
3. `malformed_log` 같은 인식 실패 유형에 골든셋 케이스 보강
4. 심판이 8b 자기채점이라 `relevancy` / `faithfulness`는 `--deep` 또는 70b 심판으로 대조 필요

---

## 자동화의 비즈니스 임팩트

| 임팩트 | 어떻게 발생하는가 |
| --- | --- |
| **MTTR 단축 가능성** | 이상 감지 → 분석 → 알람 → 조치 권고가 한 사이클로 묶임. 사람이 대시보드를 응시하지 않아도 발견 시점이 빨라짐 |
| **AI 채택의 안전장치** | DeepEval 점수가 합격선 아래일 때 자동 분석을 운영에 반영하지 않는 의사결정 근거 제공 → 잘못된 가이드로 장애가 커지는 시나리오 차단 |
| **모델 교체 결정의 객관화** | 다른 LLM/프롬프트로 교체했을 때 동일 메트릭으로 비교 가능 → 모델 업그레이드 결정이 감이 아니라 데이터로 |
| **수집 침묵 인지 가속** | 로그 ERROR가 즉시 분류·알람으로 전환 → 데이터 끊김 인지 지연 감소 |

---

## Mattermost 알람 예시

**APM 장애 알람:**

```
🔴 [심각] APM 장애 감지 - juntion_9100

📊 이상 지표
• CPU 사용률 92.5% (임계값: 80%)
• DB 커넥션 48/50 (96%, 임계값: 80%)

🤖 AI 분석 결과
장애 유형: DB 커넥션 풀 고갈
근본 원인: 슬로우 쿼리로 인한 커넥션 점유
즉각 조치: 커넥션 풀 재시작 및 슬로우 쿼리 점검
```

**로그 에러 알람:**

```
🟠 [높음] 수집 모듈 오류 - collector_agent_03

📄 오류 로그
ERROR [DataCollector] Connection refused to target host

🤖 AI 분류 결과
에러 유형: 네트워크 연결 오류
반복 가능성: 지속적
권고 조치: 대상 호스트 네트워크 연결 확인
```

---

## 기술 스택

| 구분 | 기술 |
| --- | --- |
| 언어 | Python 3.11+ |
| AI/LLM | LangChain + Ollama (llama3.1:8b) |
| 알람 | Mattermost Incoming Webhook |
| 품질 평가 | 자체 도메인 심판 (기본) + **DeepEval** (`--deep` 대조 모드) |
| 검증 자동화 | pytest — 게이트·데이터셋·정합성 로직 회귀 테스트 |
| 데이터 | Mock APM (InterMax 구조 기반) |

---

## 설치 및 실행

### 1. 사전 준비

```
# Ollama 설치 후 모델 다운로드
ollama pull llama3.1:8b

# Python 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 설정

```
cp .env.example .env
# .env 파일에서 MATTERMOST_WEBHOOK_URL 설정
# 평가를 쓸 경우 EVAL_JUDGE_MODEL을 OLLAMA_MODEL과 다른 모델로 지정
# (같으면 피평가 모델이 자기 출력을 채점하며, 리포트에 self_grading=true로 기록됨)
```

### 3. 실행

```
# 실시간 모니터링 (APM + 로그 감시 + Mattermost 알람)
python main.py

# AI 품질 평가만 실행
python main.py --eval

# HTML 리포트 생성
python main.py --report

# 품질 게이트를 CI 종료 코드로 강제 (0 통과 / 2 미달 / 3 설정오류 / 4 측정불가)
python main.py --eval --gate

# 층화 샘플 20건만 평가 (개발 중 빠른 확인용. 보고하는 점수는 전수 실행)
python main.py --eval --sample 20

# DeepEval 3지표로 대조 (케이스당 심판 9회. 훨씬 느림)
python main.py --eval --deep

# 골든셋 재생성 (같은 seed면 동일 파일)
python -m eval.datasets.build_golden --apm 60 --log 42

# 프롬프트 A/B — 데이터셋·심판 모델을 고정하고 프롬프트만 바꿔 비교
APM_PROMPT_VERSION=v1 python main.py --eval
APM_PROMPT_VERSION=v2 python main.py --eval
python -m eval.compare_runs
```

평가 결과는 `reports/eval_result.json`(최신)과 `reports/history/eval_result_<timestamp>.json`(이력)에
저장되며, 각 결과에는 데이터셋 버전·분석 모델·심판 모델·Ollama 버전이 함께 기록됩니다.
자세한 게이트 규칙은 [QUALITY_GATE.md](QUALITY_GATE.md)를 참고하세요.

---

## 프로젝트 구조

```
aiops-sentinel/
├── apm/
│   ├── mock_generator.py    # InterMax 구조 Mock 데이터
│   ├── anomaly_detector.py  # 임계값 기반 이상 감지
│   └── ai_analyzer.py       # AI 장애 원인 분석
├── logwatch/
│   ├── log_simulator.py     # 수집 모듈 로그 시뮬레이터
│   ├── log_tailer.py        # tail -f + 키워드 감지
│   └── ai_classifier.py     # AI 에러 분류
├── alert/
│   └── mattermost.py        # Mattermost Webhook 알람
├── eval/
│   ├── action_grounding.py  # 원인·조치 정합성 검증 (LLM 심판이 놓치는 것을 규칙으로)
│   ├── domain_judge.py      # 심판 1회 호출 모드 (DeepEval 9회 대비 4.5배 빠름)
│   ├── datasets/
│   │   ├── build_golden.py  # 운영 코드 경로에서 골든셋 생성 (정답은 구성상 확정)
│   │   └── golden_v2.json   # 102건, 생성 — 활성 데이터셋
│   │                        #   로그는 유형 14종 x 문구 3종 = 42건 전부 다른 표현
│   ├── eval_suite.py        # DeepEval 품질 평가 + 품질 게이트
│   ├── compare_runs.py      # 실행 간 회귀 비교 (프롬프트/심판 모델 A/B)
│   └── report_generator.py  # HTML 리포트 생성
├── config/
│   └── settings.py          # 설정 관리
└── main.py                  # 실행 진입점
```
