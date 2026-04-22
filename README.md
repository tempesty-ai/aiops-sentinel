# 🛡️ AIOps Sentinel

> APM 장애 지능형 분석 + 수집 모듈 로그 감시 + Mattermost 알람 + AI 품질 평가

## 개요

**AIOps Sentinel**은 APM(Application Performance Monitoring) 환경에서 발생하는 두 가지 핵심 문제를 AI로 자동화합니다:

1. **APM 장애 분석**: WAS 서버의 CPU/메모리/DB/응답시간 이상을 감지하고 AI가 원인 분석 및 조치 방안 생성
2. **수집 모듈 로그 감시**: 리눅스 수집 에이전트 로그를 실시간 tail하여 ERROR 발생 시 AI 분류 및 알람

## 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│                    AIOps Sentinel                       │
├─────────────────────┬───────────────────────────────────┤
│   Feature 1: APM    │   Feature 2: Log Watch            │
│                     │                                   │
│  Mock APM 데이터     │  Log Simulator (mock)             │
│       ↓             │       ↓                           │
│  이상 감지           │  tail -f (다중 파일)               │
│  (임계값 기반)       │  ERROR 키워드 감지                 │
│       ↓             │       ↓                           │
│  AI 원인 분석        │  AI 에러 분류                      │
│  (LangChain+Ollama) │  (LangChain+Ollama)               │
│       ↓             │       ↓                           │
│       └─────────────┴─────────────┐                    │
│                                   ↓                    │
│                    Mattermost Incoming Webhook         │
│                    HTML 리포트 + Eval Suite             │
└─────────────────────────────────────────────────────────┘
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.11+ |
| AI/LLM | LangChain + Ollama (llama3.1:8b) |
| 알람 | Mattermost Incoming Webhook |
| 품질 평가 | DeepEval (Agentic AI Eval Framework) |
| 데이터 | Mock APM (InterMax 구조 기반) |

## 설치 및 실행

### 1. 사전 준비

```bash
# Ollama 설치 후 모델 다운로드
ollama pull llama3.1:8b

# Python 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
cp .env.example .env
# .env 파일에서 MATTERMOST_WEBHOOK_URL 설정
```

### 3. 실행

```bash
# 실시간 모니터링 (APM + 로그 감시 + Mattermost 알람)
python main.py

# AI 품질 평가만 실행
python main.py --eval

# HTML 리포트 생성
python main.py --report
```

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

## AI 품질 평가 (Eval Suite)

Agentic AI 평가 프레임워크(DeepEval)를 활용하여 AI 분석 품질을 자동으로 측정합니다.

### 커스텀 메트릭 (50%)
| 평가 지표 | 설명 |
|-----------|------|
| 장애유형 분류 정확도 | 예상 장애유형과 일치 여부 |
| 키워드 포함율 | 분석에 필수 키워드 포함 여부 |
| 응답 완결성 | 원인/조치/예방 3요소 포함 여부 |

### DeepEval 메트릭 (50%)
| 평가 지표 | 설명 |
|-----------|------|
| Hallucination | AI가 근거 없는 내용을 생성하지 않는 비율 |
| Answer Relevancy | 질문(장애 상황)과 관련성 있는 답변 비율 |
| Faithfulness | 실제 모니터링 데이터 기반으로 답변한 비율 |
| 종합 품질 점수 | 커스텀 50% + DeepEval 50% 가중 평균 |

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
