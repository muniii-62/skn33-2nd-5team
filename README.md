# 팀 - 이탈하지 말아조😭
## 👥 팀원 소개


<table>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/e10357cb-19ec-4651-b25c-1aee30a1674f" width="100" height="100" style="object-fit: cover; border-radius: 50%;"/></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/d7ed0097-03ee-4e24-91d1-c72274685cd5" width="100" height="100" style="object-fit: cover; border-radius: 50%;"/></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/f0a8eb97-8453-4629-bb0e-683736f2e06b" width="100" height="100" style="object-fit: cover; border-radius: 50%;"/></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/732c55fd-b4a0-45eb-b9d3-767e3d7db91b" width="100" height="100" style="object-fit: cover; border-radius: 50%;"/></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/c2a45d8f-7c50-4a59-bcc6-fe9b150302fc" width="100" height="100" style="object-fit: cover; border-radius: 50%;"/></td>
  </tr>
  <tr>
    <td align="center"><b>김문규 (kmk)</b></td>
    <td align="center"><b>이서영 (lsy)</b></td>
    <td align="center"><b>정현두 (jhd)</b></td>
    <td align="center"><b>권세진 (ksj)</b></td>
    <td align="center"><b>허유나 (hyn)</b></td>
  </tr>
  <tr>
    <td align="center">팀장 · EDA·데이터 전처리 · Target 정의, RandomForest 모델링</td>
    <td align="center">RandomForest 모델링 · 하이퍼파라미터 튜닝 · 발표</td>
    <td align="center">XGBoost 모델링 · 임계값 최적화 · Streamlit 대시보드 개발</td>
    <td align="center">Logistic Regression 모델링 · Streamlit 대시보드 개발 · PPT 제작</td>
    <td align="center">LightGBM 모델링 · Feature Importance·SHAP 분석 · PPT 제작</td>
  </tr>
</table>

# 이커머스 구매 고객 재구매 이탈 예측

UCI **Online Retail II**의 영국 이커머스 실거래 로그를 고객 단위로 집계해,
향후 90일 동안 재구매하지 않을 가능성이 높은 구매 고객을 예측하는 프로젝트입니다. 예측 결과는
Streamlit 대시보드에서 캠페인 대상 선정, 위험 고객 조회, 개별 예측 및 ROI
시뮬레이션으로 연결합니다.

## 프로젝트 요약

| 항목 | 내용 |
|---|---|
| 원본 데이터 | Online Retail II, 2009-12-01~2011-12-09 |
| 원본 규모 | 1,067,371개 거래 라인, 8개 컬럼 |
| 분석 단위 | CustomerID가 확인되는 구매 고객 1명 = 1행 |
| 기준일 | 2011-09-10 |
| 분석 대상 | 기준일 이전 365일 내 구매 이력이 있는 활성 고객 |
| Target | 기준일 이후 90일간 무구매 시 `churn=1`, 재구매 시 `churn=0` |
| 최종 데이터 | 고객 4,320명, 10개 피처 |
| 클래스 비율 | 이탈 49.4%, 잔류 50.6% |
| 최종 모델 | XGBoost, 권장 기본 Threshold 0.38 |
| 핵심 평가 기준 | 이탈 고객을 놓치지 않기 위한 Recall 우선 |

## 문제 정의

원본 거래 로그에는 이탈 여부가 따로 기록되어 있지 않습니다. 따라서 특정 기준일을
중심으로 과거 행동과 미래 재구매 여부를 분리했습니다.

이 프로젝트의 `churn`은 회원 탈퇴, 계약 해지 또는 계정 삭제를 의미하지 않습니다.
데이터에 가입일과 탈퇴 기록이 없으므로, **구매 이력이 있는 고객이 결과 관찰 기간에
재구매하지 않은 상태**를 재구매 이탈의 대리변수로 정의합니다. 따라서 화면과 결과서의
`고객 활동 기간`은 가입 기간이 아니라 **첫 구매일부터 기준일까지의 경과 기간**입니다.

```text
과거 거래 관찰 구간                  기준일                  결과 관찰 구간
───────────────────────────────────  2011-09-10  ──────────────────────────
피처 생성                                                향후 90일 재구매 확인

최근 365일 내 구매 이력 있음 + 향후 90일 내 재구매 없음 → churn = 1
최근 365일 내 구매 이력 있음 + 향후 90일 내 재구매 있음 → churn = 0
```

90일은 클래스 비율을 정확히 50:50으로 만들기 위한 수학적 최적값이 아니라, 재구매
간격 분포, 캠페인 실행 가능 기간, 충분한 이탈 관찰 기간과 클래스 균형을 함께 고려한
실무적 절충값입니다. 60~120일 민감도 분석에서도 기간이 길어질수록 이탈률은
감소하지만 모델의 순위 판별력은 크게 변하지 않아, 90일을 유일한 정답이 아닌
운영 기준으로 사용했습니다.

## 원본 데이터 구성

원본 CSV는 주문 한 건이 아니라 **주문에 포함된 상품 한 줄당 1행**으로 구성됩니다.
동일한 `Invoice`가 여러 행에 반복될 수 있습니다.

| 원본 컬럼 | 자료형 | 설명 | 분석 시 주의점 |
|---|---|---|---|
| `Invoice` | 문자열 | 주문·송장 번호 | `C`로 시작하면 취소 주문 |
| `StockCode` | 문자열 | 상품 코드 | 상품 외 내부 기록 코드가 포함될 수 있음 |
| `Description` | 문자열 | 상품 설명 | 4,382건 결측 |
| `Quantity` | 정수 | 주문 수량 | 음수는 반품·취소 수량 |
| `InvoiceDate` | 날짜·시간 | 주문 시각 | 로드 후 `datetime`으로 변환 |
| `Price` | 실수 | 상품 1개당 가격(GBP) | 0 이하 값은 정상 판매에서 제외 |
| `Customer ID` | 실수 | 익명 고객 번호 | 243,007건 결측, 로드 후 `CustomerID`로 변경 |
| `Country` | 문자열 | 주문 국가 | 영국 여부 피처 생성에 사용 |

예시:

```text
Invoice  StockCode  Description                         Quantity  InvoiceDate          Price  Customer ID  Country
489434   85048      15CM CHRISTMAS GLASS BALL 20 LIGHTS 12        2009-12-01 07:45:00  6.95   13085        United Kingdom
489434   79323P     PINK CHERRY LIGHTS                  12        2009-12-01 07:45:00  6.75   13085        United Kingdom
```

금액 단위는 원본 데이터에 맞춰 **파운드(GBP, £)**를 사용합니다. `Customer ID`는
익명 숫자이지만, 이름·이메일 등 직접 식별정보는 포함되어 있지 않습니다.

## 데이터 정제와 전처리

### 정상 구매 데이터

재구매 여부를 판정할 때는 다음 조건을 모두 만족하는 정상 구매만 사용합니다.

- `Invoice`가 `C`로 시작하지 않음
- `StockCode`가 상품 코드 형식에 해당함
- `Quantity > 0`, `Price > 0`
- `CustomerID`가 존재함

정제 후 데이터는 802,632개 거래 라인, 5,852명 고객입니다. 단, 순매출과 반품
관련 피처를 만들 때는 상품 라인의 취소·반품 기록도 포함해 구매와 상쇄합니다.

### 데이터 생성 흐름

```text
원본 1,067,371행
  ↓ 정상 구매·상품·회원 조건 적용
정제 거래 802,632행
  ↓ 기준일 이전 활성 고객 선별 및 고객 단위 집계
고객 스냅샷 4,320행
  ↓ 층화 분할(random_state=42)
Train 2,592명 / Validation 864명 / Test 864명
  ↓ Train으로만 전처리기와 is_low_value 기준 학습
최종 10개 피처
```

- 분할 비율: Train 60% / Validation 20% / Test 20%
- `is_low_value`: Train의 평균 주문금액 하위 20% 기준을 세 데이터에 동일 적용
- `net_revenue`: 음수 0 클리핑 → `log1p` → `StandardScaler`
- 연속형 피처: `StandardScaler`
- 이진·비율 피처: 원값 유지

## 최종 피처

| 피처 | 설명 | 처리 |
|---|---|---|
| `net_revenue` | 취소·반품을 상쇄한 과거 순매출 | 0 클리핑 → log1p → 표준화 |
| `recency_days` | 최근 거래 활동일 후 기준일까지 경과일 | 표준화 |
| `frequency` | 관찰 구간의 고유 주문 횟수 | 표준화 |
| `distinct_products` | 구매한 고유 상품 종류 수 | 표준화 |
| `tenure_days` | 첫 구매 후 기준일까지 경과일 | 표준화 |
| `avg_days_between_orders` | 고객 활동 기간 ÷ 주문 횟수 | 표준화 |
| `is_low_value` | 평균 주문금액이 Train 하위 20%인지 여부 | 이진값 |
| `is_uk` | 주 이용 국가가 영국인지 여부 | 이진값 |
| `has_return` | 취소·반품 경험 여부 | 이진값 |
| `recent_activity_ratio` | 전체 주문 중 최근 90일 주문 비중 | 비율값 |

`recency_days`와 `avg_days_between_orders`의 상관계수는 약 0.72입니다. 트리 모델은
그대로 사용했지만, Logistic Regression 같은 선형모델에서는 다중공선성에 유의해야
합니다. `return_ratio`와 연속형 `avg_order_value`는 신호와 중복성을 검토한 뒤 최종
피처에서 제외했습니다.

## 병렬 모델링과 모델 선택

팀원들이 동일한 Train/Validation 데이터와 `random_state=42`를 사용해 여러 모델을
병렬로 실험했습니다.

| 담당 폴더 | 주요 모델·역할 |
|---|---|
| `models/ksj/` | Logistic Regression |
| `models/kmk/` | Random Forest |
| `models/lsy/` | Random Forest 및 파라미터 분석 |
| `models/jhd/` | XGBoost, SVC, Soft Voting, 교차검증·튜닝 |
| `models/hyn/` | LightGBM |
| `models/final/` | 후보 비교, 최종 모델 및 Test 평가 |

모델과 하이퍼파라미터는 Validation에서만 비교했고, Test는 최종 XGBoost와 Threshold
0.38을 확정한 뒤 평가했습니다. Recall 0.85 이상인 Threshold 중 F1이 가장 높은
0.38을 권장 기본값으로 선정했습니다.

## 최종 Test 결과

| 모델 | Accuracy | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic baseline, Threshold 0.50 | 0.683 | 0.691 | 0.675 | 0.683 | 0.754 | 0.726 |
| XGBoost, Threshold 0.38 | 0.679 | **0.859** | 0.628 | **0.726** | 0.752 | 0.704 |

최종 운영안은 Logistic baseline보다 Recall이 16.9%p, F1이 4.3%p 높았습니다.
False Negative는 132명에서 60명으로 감소해 이탈 고객 72명을 추가로 탐지했으며,
그 대가로 Precision과 PR-AUC 일부를 양보했습니다. 이는 이탈 고객을 놓치는 비용을
오탐 캠페인 비용보다 크게 본 프로젝트 목적에 따른 선택입니다.

## 실행 방법

프로젝트 루트에서 실행합니다.

```bash
pip install -r requirements.txt

# 최초 1회: 원본 다운로드 및 전처리 데이터 생성
python -m src.data
python -m src.prepare_data

# Streamlit 대시보드 실행
streamlit run streamlit_app/app.py
```

`python -m src.data`는 로컬 CSV가 없을 때 KaggleHub의 `mashlyn/online-retail-ii-uci`
미러를 내려받습니다. Kaggle 인증이 필요한 환경에서는 Kaggle 토큰 설정이 필요할 수
있습니다. `data/`는 용량과 원본 데이터 관리 문제로 Git에 포함하지 않습니다.

분석 노트북은 다음 순서로 실행하면 전체 흐름을 확인하기 쉽습니다. 파일명에는 순번을
붙이지 않았지만 아래 순서가 프로젝트의 분석 진행 순서입니다.

1. `notebooks/check.ipynb` — 원본 데이터와 기본 품질 점검
2. `notebooks/eda_log.ipynb` — 거래 행 단위 EDA와 정제 기준 확인
3. `notebooks/eda_customer.ipynb` — 고객 Feature와 Target 관계 분석
4. `notebooks/preprocessing.ipynb` — 분할·변환·최종 학습 데이터 확인
5. `notebooks/model_experiments.ipynb` — 후보 모델 비교, Threshold 결정, Test 평가와 오류 분석

`X_test`, `y_test`는 최종 모델과 Threshold를 확정한 뒤 평가에만 사용합니다.

## Streamlit 대시보드

현재 Streamlit은 제출·독립 예측용 단일 Pipeline인
`models/churn_pipeline.joblib`을 직접 불러옵니다. Pipeline 내부의 동일한 전처리기와
XGBoost 분류기를 고객 목록·개별 예측·ROI에서 공통으로 사용합니다.

대시보드는 다음 5개 메뉴로 구성됩니다.

- **데이터 분석 및 모델 성능**: Target 분포와 핵심 EDA, 모델별 Validation 비교, 최종 Test 성능·오류 분석 확인
- **캠페인 대상 선정**: 38% 선정 근거와 Validation 예상 성능, 현재 고객 대상 규모를 확인하고 다른 Threshold의 민감도를 참고용으로 비교
- **고객 현황 및 목록**: 캠페인 대상 고객 조회·검색·필터링, 고객 상세 확인 및 Excel/CSV 다운로드
- **개별 고객 예측**: 고객 행동 정보를 입력해 38% 기준 이탈 판정과 권장 관리 방향 확인
- **ROI 시뮬레이터**: 38% 이상 캠페인 대상의 비용·유지이익·순이익·ROI 비교

38%는 Validation에서 정한 최종 운영 기준입니다. 첫 화면의 비교 슬라이더는
Validation 예상 결과만 바꾸며, 고객 목록·개별 예측·ROI에는 항상 38%가 적용됩니다.
개별 예측의 평균 구매 간격, 평균 주문금액과 저가치 여부는 기본 입력값에서 자동 계산됩니다.

ROI는 실제 성과를 보장하는 예측치가 아니라 비용, 캠페인 성공률, 매출총이익률과
분석 기간을 조정하는 가정 기반 시뮬레이션입니다. 모든 금액은 GBP로 표시합니다.

## 폴더 구조

```text
Project2/
├─ data/                               # 로컬 데이터(.gitignore 적용)
│  ├─ raw/online_retail_II.csv
│  └─ preprocessed/                    # X/y 분할 데이터·전처리기·임계값
├─ notebooks/                          # 품질 점검·EDA·전처리·최종 모델 실험
│  ├─ check.ipynb
│  ├─ eda_log.ipynb
│  ├─ eda_customer.ipynb
│  ├─ preprocessing.ipynb
│  └─ model_experiments.ipynb
├─ src/                                # 다운로드, 피처 생성, 전처리 파이프라인
├─ models/
│  ├─ churn_pipeline.joblib            # 제출·신규 고객 예측용 단일 Pipeline
│  ├─ baseline_logistic.ipynb          # Logistic baseline
│  ├─ example.ipynb                    # 공통 데이터 로드 예시
│  ├─ hyn/, jhd/, kmk/, ksj/, lsy/     # 팀원별 병렬 모델 실험
│  └─ final/                           # Streamlit용 최종 모델·전처리기·평가 노트북
├─ artifacts/
│  ├─ feature_schema.json              # Pipeline 입력 Feature와 Target 정의
│  ├─ model_metadata.json              # 모델·Threshold·해시·학습 조건
│  ├─ metrics.csv                      # Validation 후보 비교와 Test 결과
│  ├─ error_analysis.csv               # TP·TN·FP·FN 고객군 비교
│  └─ calibration_metrics.csv          # Brier Score와 확률 보정 점검
├─ reports/
│  ├─ preprocessing_report.md          # 데이터 전처리 결과서
│  ├─ training_report.md               # 인공지능 모델 학습 결과서
│  └─ figures/                         # 전처리·학습 결과서 그래프
├─ streamlit_app/
│  ├─ app.py                           # 대시보드 진입점
│  ├─ config.py                        # 경로·Threshold·피처 설정
│  ├─ model_loader.py                  # 모델·전처리기 검증 및 로드
│  ├─ customer_scoring.py              # 현재 고객 스냅샷·점수 계산
│  └─ tabs/                            # 대상 선정·고객 목록·개별 예측·ROI
├─ outputs/                            # 발표자료 등 최종 산출물
├─ requirements.txt
└─ README.md
```

## 데이터 출처와 한계

| 항목 | 내용 |
|---|---|
| 데이터 | UCI Online Retail II |
| UCI URL | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| 라이선스 | CC BY 4.0 |
| 실제/합성 | 실제 영국 온라인 소매 거래 로그 |
| 개인정보 | 익명 CustomerID 외 직접 식별정보 없음 |

주요 한계는 다음과 같습니다.

- 단일 영국 소매업체의 2009~2011년 데이터이므로 현재 시장에 바로 일반화하기 어려움
- 계약 해지 기록이 없어 90일 무구매를 이탈의 대리변수로 사용
- CustomerID 결측 거래는 고객 단위 분석에서 제외
- Calibration Curve와 Brier Score로 확률 상태를 점검했지만 별도 보정 모델은 적용하지 않아 ROI의 기대인원 해석에 오차 가능
- ROI 비용·성공률·이익률은 외부 참고값을 이용한 가정이며 실제 A/B 테스트로 교체 필요
- Random 분할을 사용했으므로 실제 운영 전 시간순 검증과 데이터 드리프트 점검 필요

## 참고 산출물

- `reports/preprocessing_report.md`: 데이터 소개부터 누수 방지·최종 Feature까지 정리한 전처리 결과서
- `reports/training_report.md`: 모델 비교·Threshold·Test·오류 분석·확률 보정·모델 해석 결과서
- `artifacts/feature_schema.json`: 최종 Pipeline 입력 Feature 정의
- `artifacts/model_metadata.json`: 최종 모델과 예측 기준 메타데이터
- `notebooks/check.ipynb`: 원본 데이터 기본 점검
- `notebooks/eda_log.ipynb`: 원본 거래 구조 및 정제 기준
- `notebooks/eda_customer.ipynb`: 고객 피처와 이탈 관계
- `notebooks/preprocessing.ipynb`: 분할과 전처리 결과 검증
- `notebooks/model_experiments.ipynb`: 모델 비교·오류 분석·Calibration 재현
- `models/final/model_comparison.ipynb`: Logistic baseline과 최종 모델 비교
- `models/final/final_result.ipynb`: 최종 Test 평가 지표와 그래프
- `프로젝트_진행정리.pdf`: 프로젝트 진행 과정 정리
- `outputs/이탈하지말아조_고객이탈예측_발표자료.pptx`: 발표자료
