# 🛡️ `aiops-sentinel` — AIOps + AI 출력 품질 평가

> APM 장애 지능형 분석 + 수집 모듈 로그 감시 + Mattermost 알람 + **AI 품질 평가(DeepEval)**.
> 핵심은 "AI에게 장애 분석을 시키는 것"이 아니라, **"그 분석을 운영에 써도 되는지 수치로 게이트하는 것"** 입니다.

---

## 한 줄 가치

> **"AI가 만든 장애 분석을 그대로 신뢰해도 되는가?"** 를 듀얼 트랙(운영 + 평가)으로 답하는 저장소.
> 다른 AIOps 도구가 "AI가 알아서 분석합니다"에서 멈추는 동안, 이 저장소는 그 분석을 **DeepEval로 측정**합니다.

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
| R3 | **AI 분석 신뢰 (가장 중요)** | AI가 그럴듯한 원인 분석을 **만들어내지만(hallucination)**, 실제 메트릭과 무관하거나, 핵심 키워드가 빠지거나, 조치 방안이 없는 경우 → 잘못된 조치로 **장애를 키울 수 있음** |

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

### 트랙 2: AI 출력 품질 평가 (DeepEval Suite)

이 저장소의 핵심 차별점. 50:50 가중 평균으로 AI 분석 품질을 정량화합니다.

**커스텀 메트릭 (50%)**

| 평가 지표 | 검증 의도 |
| --- | --- |
| 장애유형 분류 정확도 | 예상 장애유형과 일치 여부 |
| 키워드 포함율 | 분석에 필수 키워드 포함 여부 |
| 응답 완결성 | **원인 / 조치 / 예방** 3요소 포함 여부 |

**DeepEval 메트릭 (50%)**

| 평가 지표 | 검증 의도 |
| --- | --- |
| **Hallucination** | AI가 근거 없는 내용을 생성하지 않는 비율 |
| **Answer Relevancy** | 장애 상황과 관련성 있는 답변 비율 |
| **Faithfulness** | 실제 모니터링 데이터 기반으로 답변한 비율 |
| 종합 품질 점수 | 커스텀 50% + DeepEval 50% 가중 평균 |

→ **"AI의 답을 운영에 쓸 수 있는가"** 를 점수로 게이트.

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
| 품질 평가 | **DeepEval (Agentic AI Eval Framework)** |
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
```

### 3. 실행

```
# 실시간 모니터링 (APM + 로그 감시 + Mattermost 알람)
python main.py

# AI 품질 평가만 실행
python main.py --eval

# HTML 리포트 생성
python main.py --report
```

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
│   ├── eval_suite.py        # DeepEval 품질 평가
│   └── report_generator.py  # HTML 리포트 생성
├── config/
│   └── settings.py          # 설정 관리
└── main.py                  # 실행 진입점
```
