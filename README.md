# LLM Jailbreak Detector

BERT 파인튜닝 + GPT API 데이터 증강 기반의 LLM Jailbreak 프롬프트 탐지 시스템입니다.  
텍스트 프롬프트가 **Jailbreak 시도**인지 **정상 요청**인지를 실시간으로 분류하며,  
**Gradio Live Demo**를 통해 즉시 체험할 수 있습니다.

---

## 주요 성과

| 지표 | 값 |
|------|----|
| Test Accuracy | **97.14%** |
| Test F1 (macro) | **0.9714** |
| Test Precision | **97.22%** |
| Test Recall | **97.14%** |

- 총 데이터셋: **14,000개** (공개 4,000개 + GPT API 직접 생성 10,000개)
- Roleplay / DAN / Prompt Injection 유형 탐지율 **100%**
- Gradio 웹 인터페이스로 **Live Demo** 제공 (`http://127.0.0.1:7860`)

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    데이터 파이프라인                        │
│                                                         │
│  공개 데이터셋 (HuggingFace)                              │
│  ├─ walledai/JailbreakHub          (Jailbreak ~1,500)   │
│  ├─ TrustAIRLab/in-the-wild-...   (Jailbreak   ~300)   │
│  ├─ rubend18/ChatGPT-Jailbreak-.. (Jailbreak   ~200)   │
│  ├─ tatsu-lab/alpaca               (Normal    ~1,500)   │
│  └─ databricks/databricks-dolly.. (Normal      ~500)   │
│                         +                               │
│  GPT API 생성 (10,000개)                                 │
│  ├─ 한국어 Jailbreak 패턴        1,500개                  │
│  ├─ 오타/특수문자 변형            1,000개                  │
│  ├─ 점진적 유도 방식             1,000개                  │
│  ├─ 역할극 변형 패턴             1,000개                  │
│  ├─ 기타 변형 패턴                 500개                  │
│  └─ 정상 프롬프트 (한국어)        5,000개                  │
└───────────────────────┬─────────────────────────────────┘
                        │ merge + 전처리
                        ▼
┌─────────────────────────────────────────────────────────┐
│              BERT 파인튜닝 (bert-base-uncased)            │
│                                                         │
│  ├─ 입력: max_length=128 토큰                            │
│  ├─ Batch: 32 / LR: 2e-5 / Warmup: 200 steps           │
│  ├─ Epochs: 3 (Early stopping 적용)                     │
│  └─ 출력: Normal(0) / Jailbreak(1) 이진 분류             │
└───────────────────────┬─────────────────────────────────┘
                        │ best checkpoint
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  Gradio Live Demo                        │
│                                                         │
│  demo.py → http://127.0.0.1:7860                        │
│  ├─ 텍스트 입력 (한국어/영어)                             │
│  ├─ 판정 결과: ✅ Normal / 🚨 Jailbreak                   │
│  └─ 확률 점수: Normal/Jailbreak 각 % 표시                 │
└─────────────────────────────────────────────────────────┘
```

---

## 프로젝트 구조

```
llm-jailbreak-detector/
├── data/
│   ├── raw/                      # 원시 수집/생성 데이터
│   └── processed/                # 전처리 완료 데이터 (train/val/test.csv)
├── src/
│   ├── data_collection.py        # HuggingFace 공개 데이터셋 수집
│   ├── generate_dataset.py       # GPT API로 10,000개 생성
│   ├── merge_datasets.py         # 데이터셋 병합 및 분할
│   ├── dataset.py                # PyTorch Dataset / 전처리
│   ├── model.py                  # BertForSequenceClassification 래퍼
│   ├── train_pytorch.py          # Pure PyTorch BERT 학습 (4K 베이스)
│   ├── train_augmented.py        # BERT 재학습 — 14K 통합 데이터셋 (최종)
│   ├── train_roberta.py          # RoBERTa 비교 실험
│   ├── evaluate.py               # 테스트셋 평가 및 플롯
│   ├── analysis.py               # Jailbreak 유형 분석 + 오류 분석
│   └── compare_models.py         # BERT vs RoBERTa 비교 시각화
├── results/
│   ├── models/
│   │   ├── bert-jailbreak/       # BERT 체크포인트 (4K 학습)
│   │   ├── roberta-jailbreak/    # RoBERTa 체크포인트
│   │   └── bert-augmented/       # BERT 체크포인트 (14K 학습, 최종)
│   ├── plots/                    # 학습 곡선, 혼동행렬 등
│   ├── val_metrics.json          # BERT (4K) 평가 지표
│   ├── roberta_metrics.json      # RoBERTa 평가 지표
│   ├── augmented_metrics.json    # BERT (14K) 평가 지표
│   └── error_analysis.txt        # 오류 분석 리포트
├── demo.py                       # Gradio Live Demo
└── requirements.txt
```

---

## AI 도구 활용 전략 (Prompting Log)

본 프로젝트는 **Claude Code**를 팀원처럼 매니징하면서 개발했습니다.  
단순 코드 생성이 아닌, 설계 결정·디버깅·리팩터링·데이터 전략 전 과정에 걸쳐 프롬프팅을 활용했습니다.

### Claude Code 활용 방식

- **설계 단계**: 데이터 파이프라인 구조, 모델 선택 근거를 프롬프트로 논의
- **구현 단계**: 각 스크립트의 스켈레톤 생성 → 검토 후 수정
- **디버깅 단계**: 에러 메시지와 스택 트레이스를 그대로 붙여넣어 원인 분석
- **실험 단계**: 하이퍼파라미터 조정, 데이터 증강 전략 논의

### 주요 프롬프팅 사례 6개

**1. 데이터 수집 설계**
```
HuggingFace에서 Jailbreak 탐지용 공개 데이터셋을 찾아줘.
Jailbreak 쪽 2,000개, Normal 쪽 2,000개를 균형 있게 구성하고
중복 제거 로직도 포함해줘. CSV로 저장하는 data_collection.py를 작성해.
```
→ `walledai/JailbreakHub`, `tatsu-lab/alpaca` 등 5개 데이터셋 조합과
  `_deduplicate()` 함수가 포함된 스크립트 생성

**2. GPT API 데이터 생성 안전성 처리**
```
GPT API 호출 중 중간에 끊기면 데이터가 날아가는 문제가 있어.
배치마다 즉시 CSV에 append하고, 재시작 시 이어서 생성하는 로직을 넣어줘.
연속 실패 10회 시 해당 배치는 스킵하고 계속 진행해야 해.
```
→ `load_existing_counts()` + CSV append 모드 + `MAX_CONSECUTIVE_FAILS` 로직 구현

**3. BERT fp16 학습 붕괴 디버깅**
```
BERT 학습 중 Loss가 갑자기 NaN이 됐어. TrainingArguments에서
fp16=True로 설정했는데 이게 원인일까? 해결 방법을 알려줘.
```
→ `bert-base-uncased`의 LayerNorm이 fp16에서 언더플로우를 일으킬 수 있음을 분석,
  `fp16=False`로 변경 (`# disabled: caused BERT encoder collapse` 주석 포함)

**4. Jailbreak 유형 분류기 설계**
```
테스트셋에서 Jailbreak 샘플을 DAN, Prompt Injection, Roleplay 등 유형별로
분류하고 유형별 탐지율을 시각화하는 코드를 작성해줘.
규칙 기반으로 우선순위 패턴 매칭을 하고, 나머지는 'Other/Mixed'로.
```
→ `TYPE_RULES` 정규식 우선순위 리스트와 탐지율 바차트 생성 코드 구현

**5. RoBERTa 비교 실험 추가**
```
BERT와 동일한 조건(LR, batch size, max_length)으로 RoBERTa를 학습하는
train_roberta.py를 만들고, 두 모델의 metrics를 비교하는
compare_models.py도 작성해줘.
```
→ `roberta-base` 기반 학습 스크립트 + Val F1 학습 곡선 오버레이 플롯 생성

**6. 14K 데이터 증강 후 재학습**
```
GPT API로 생성한 10,000개 데이터와 기존 4,000개를 합쳐서
BERT를 재학습하는 train_augmented.py를 작성해줘.
기존 모델 대비 성능 비교표도 출력해줘.
```
→ `PREV_METRICS` 기준 비교 출력 + `bert-augmented` 체크포인트 저장 로직 구현

### Git 커밋 히스토리

| 커밋 | 내용 |
|------|------|
| `e144ec2` | 최초 커밋 |
| `e078af7` | 프로젝트 구조 초기화 (디렉터리, .gitignore, requirements.txt) |
| `4f46e65` | HuggingFace 공개 데이터셋 수집 완료 (~4,000개) |
| `1f31bd2` | EDA 및 전처리 완료 (train/val/test 분할, 정제) |
| `b9bc6fb` | BERT 학습 완료 (Val F1=0.9883, Test F1=0.9933) |
| `50f3a26` | Jailbreak 유형 분석 + 오류 분석 추가 |
| `112a7b6` | RoBERTa 비교 실험 추가 (BERT 우위 확인) |
| `7ea3d32` | GPT API로 10,000개 데이터 생성 |
| `dbcd9bf` | Gradio 웹 데모 추가 (`demo.py`) |
| `99a69f8` | 종합 README 추가 |
| `633e6f9` | 14K 증강 데이터셋으로 BERT 재학습 |
| `543386d` | 데모가 증강 모델(`bert-augmented`) 사용하도록 업데이트 |

---

## 실행 방법 (How to Run)

### 1. 환경 세팅

```bash
# 가상환경 생성 (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 생성

```bash
# (선택) HuggingFace 공개 데이터셋 수집
python src/data_collection.py

# (선택) GPT API로 10,000개 생성 — OPENAI_API_KEY 필요
$env:OPENAI_API_KEY="sk-..."     # Windows PowerShell
# export OPENAI_API_KEY=sk-...   # macOS/Linux
python src/generate_dataset.py

# 데이터셋 병합 및 train/val/test 분할
python src/merge_datasets.py
```

> `data/processed/` 아래에 `train.csv`, `val.csv`, `test.csv`가 생성됩니다.

### 3. 모델 학습

```bash
# BERT 학습 — 4K 공개 데이터 (베이스 모델)
python src/train_pytorch.py
# → results/models/bert-jailbreak/ 에 체크포인트 저장

# BERT 재학습 — 14K 통합 데이터 (최종 모델, 데모용)
python src/train_augmented.py
# → results/models/bert-augmented/ 에 체크포인트 저장

# (선택) 테스트셋 최종 평가
python src/evaluate.py

# (선택) Jailbreak 유형 분석
python src/analysis.py

# (선택) RoBERTa 비교 실험
python src/train_roberta.py
python src/compare_models.py
```

### 4. Live Demo 실행

```bash
python demo.py
# → http://127.0.0.1:7860 에서 접속
```

학습된 모델(`results/models/bert-augmented/best/`)이 없으면 먼저 3번 학습 단계를 거쳐야 합니다.

---

## 데이터셋 구성

| 소스 | 종류 | 개수 |
|------|------|------|
| walledai/JailbreakHub | Jailbreak | ~1,500 |
| TrustAIRLab/in-the-wild-jailbreak-prompts | Jailbreak | ~300 |
| rubend18/ChatGPT-Jailbreak-Prompts | Jailbreak | ~200 |
| tatsu-lab/alpaca | Normal | ~1,500 |
| databricks/databricks-dolly-15k | Normal | ~500 |
| **소계 (공개 데이터셋)** | | **~4,000** |
| GPT API: 한국어 Jailbreak 패턴 | Jailbreak | 1,500 |
| GPT API: 오타/특수문자 변형 | Jailbreak | 1,000 |
| GPT API: 점진적 유도 방식 | Jailbreak | 1,000 |
| GPT API: 역할극 변형 패턴 | Jailbreak | 1,000 |
| GPT API: 기타 변형 패턴 | Jailbreak | 500 |
| GPT API: 정상 프롬프트 (한국어) | Normal | 5,000 |
| **소계 (생성 데이터)** | | **10,000** |
| **총계** | | **14,000** |

**분할 비율**: Train 80% / Val 10% / Test 10%

### Jailbreak 유형별 탐지율 (테스트셋)

| 유형 | 샘플 수 | 탐지율 |
|------|--------|--------|
| Roleplay / Act-as | 70 | **100.0%** |
| Prompt Injection | 18 | **100.0%** |
| DAN / Developer-Mode | 11 | **100.0%** |
| Hypothetical / Fictional | 11 | **100.0%** |
| Persona / Impersonation | 9 | **100.0%** |
| Other / Mixed | 181 | **97.8%** |

---

## 비교 실험 결과

### BERT vs RoBERTa (4K 공개 데이터셋 기준)

| 모델 | 데이터셋 | Test Accuracy | Test F1 (macro) | Test Precision | Test Recall |
|------|---------|--------------|----------------|--------------|------------|
| **BERT** (bert-base-uncased) | 4,000 | **99.33%** | **0.9933** | **99.34%** | **99.33%** |
| RoBERTa (roberta-base) | 4,000 | 98.33% | 0.9833 | 98.39% | 98.33% |

### 데이터 증강 전후 비교 (BERT)

| 데이터셋 | 크기 | Test Accuracy | Test F1 (macro) | 비고 |
|---------|------|--------------|----------------|------|
| 공개 데이터 (기존) | 4,000 | 99.33% | 0.9933 | 오탐(FP) 0건 |
| 공개 + GPT API 생성 | **14,000** | 97.14% | **0.9714** | 다국어·변형 패턴 포함 |

> 14K 모델은 수치 지표가 소폭 낮아졌으나, 한국어 및 변형 패턴에 대한 **일반화 성능**이 향상되었습니다.

---

## 한계점 및 향후 과제

### 현재 한계점

1. **일부 영어 Jailbreak 미탐지**  
   "Ignore all previous instructions..." 같은 단순 패턴을 일부 Normal로 분류합니다.  
   학습 데이터의 클래스 분포 및 패턴 다양성 부족이 원인으로 추정됩니다.

2. **128 토큰 제한**  
   BERT의 최대 입력 길이(512) 중 128만 사용하여 긴 프롬프트는 뒷부분이 잘립니다.  
   멀티턴 대화나 장문 Jailbreak 패턴에 취약합니다.

3. **영어 중심 사전학습 모델**  
   `bert-base-uncased`는 영어 중심이라 한국어 표현의 미묘한 의미 차이를 충분히 포착하지 못할 수 있습니다.  
   `klue/bert-base` 등 한국어 특화 모델과의 비교가 필요합니다.

4. **정적 모델**  
   신규 Jailbreak 기법이 등장해도 재학습 없이는 대응이 어렵습니다.

### 향후 과제

- [ ] `klue/bert-base` 또는 다국어 모델(`mbert`, `xlm-roberta`)로 교체 실험
- [ ] 입력 길이 256~512 토큰으로 확장
- [ ] 데이터 증강: adversarial examples, back-translation
- [ ] 온라인 학습(Online Learning) 파이프라인 구축으로 신규 패턴 대응
- [ ] 앙상블(BERT + RoBERTa) 시도
- [ ] Gradio 데모에 설명 가능 AI(SHAP/LIME) 시각화 추가

---

## 의존성

```
torch>=2.2.0
transformers>=4.40.0
datasets>=2.19.0
gradio>=4.0.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
openai>=1.0.0        # 데이터 생성 시에만 필요
```
