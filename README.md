# 이커머스 고객 이탈 예측 프로젝트(계속해서 수정 예정)

Online Retail II(UCI) 실거래 로그를 기반으로 고객 이탈을 예측한다.

## 실행 방법 (최초 1회, 터미널에 작성해주세요)

```bash
pip install -r requirements.txt
python src/data.py          # 원본 데이터 다운로드 (data/raw/)
python src/prepare_data.py  # 전처리 완료 데이터 생성 (data/preprocessed/)
```

## 모델 학습 시작하기

`models/example_load.py`를 본인 노트북 맨 위 셀에 그대로 복사해서 붙여넣으면
`X_train`, `y_train`, `X_val`, `y_val`이 바로 준비됩니다.

```python
model.fit(X_train, y_train)
pred = model.predict(X_val)
```

## 규칙 (반드시 지켜주세요)

- **Val로 모델을 비교**하고 튜닝하세요. 여러 번 확인해도 괜찮습니다.
- **Test는 절대 개인적으로 사용하지 마세요.** 팀 전체가 "이 모델로 최종 확정" 합의한 뒤,
  단 한 번만 평가합니다. (`data/preprocessed/X_test.csv`, `y_test.csv`에 존재하지만
  `example_load.py`에서 의도적으로 불러오지 않습니다.)
- **비교 기준은 Recall 우선**입니다 (이탈 고객을 놓치지 않는 것이 중요).
  Precision, AUC도 함께 기록해 공유해주세요.
- `random_state=42`는 전체 팀 공통 시드입니다. 임의로 바꾸지 마세요 (재현성 깨짐).

## 데이터 요약

- 원본: Online Retail II, 2009-12 ~ 2011-12, 영국 도매상 실거래 로그
- 이탈 정의: 기준일(2011-09-10) 이전 365일 내 구매 이력 있는 활성 고객 대상,
  기준일 이후 90일간 무구매 = 이탈
- 최종 고객 수: 4,320명 (이탈률 49.4%)
- 피처: recency_days, frequency, distinct_products, net_revenue, tenure_days,
  is_low_value, is_uk (총 7개)
- 상세 근거는 `전처리_결과서.md` 및 `notebooks/01_eda_log.ipynb`, `02_eda_customer.ipynb` 참고

## 타깃 정의

`churn` 컬럼: **1 = 이탈, 0 = 잔류(유지)**

기준일(2011-09-10) 이전 365일 내 구매 이력이 있는 활성 고객 중,
기준일 이후 90일간 재구매가 없으면 이탈(1)로 라벨링됨.
## 폴더 구조

```
data/
  raw/            # 원본 (자동 생성, git 미포함)
  preprocessed/   # 전처리 완료 (자동 생성, git 미포함)
notebooks/        # EDA 노트북
src/
  data.py         # 원본 로드
  features.py     # 고객 스냅샷(RFM+파생피처) 생성
  transforms.py   # 전처리 변환 함수
  prepare_data.py # Train/Val/Test 분리 + 전처리 실행 스크립트
models/
  example_load.py # 팀원용 데이터 로드 예시
```
## 최종 피처 세트

| 피처 | 설명 | 신호 강도 | 처리 |
|---|---|---|---|
| avg_days_between_orders | 평균 구매 간격(일) | 최강 (이탈률 21%→76%, 상관 0.37) | StandardScaler |
| recency_days | 마지막 구매 후 경과일 | 강함 (24%→71%, 상관 0.35) | StandardScaler |
| frequency | 관찰구간 내 구매 횟수 | 강함 (69%→16%, 상관 -0.25) | StandardScaler |
| distinct_products | 구매한 상품 종류 수 | 강함 (69%→22%, 상관 -0.28) | StandardScaler |
| recent_activity_ratio | 최근 90일 구매비중 | 강함 (63%→31%, 상관 -0.13) | 없음(비율, 0~1) |
| is_low_value | 평균 주문금액 하위 20% 여부 | 강함 (44%→62%) | 없음(이진) |
| has_return | 취소 경험 여부 | 중간 (60%→37%, **방향 반전**) | 없음(이진) |
| **net_revenue** | 순매출(취소 상쇄) | 중간 (상관 -0.13), 파레토 구조 | **0 클리핑 → log1p(로그변환) → StandardScaler** |
| tenure_days | 첫 구매 후 경과일 | 중간 (상관 -0.14), 비단조 | StandardScaler |
| is_uk | UK 거주 여부 | 약함 (47%→50%) | 없음(이진) |

**제외된 피처**: return_ratio(무신호), avg_order_value 연속형(무신호, is_low_value로 대체)

**has_return 방향 반전**: 당초 "반품 많으면 불만족→이탈"을 가정했으나 실제로는 반대.
반품은 거래가 지속되는 증거이며, 이미 이탈한 고객은 반품 기회 자체가 없음.

**제외된 피처**: return_ratio(무신호, 상관 0.04), avg_order_value 연속형(무신호,
is_low_value로 대체) — 근거는 `전처리_결과서.md` 참고

> 로그 변환은 net_revenue에만 적용됩니다. EDA(07)에서 이 피처만 소수 고객이
> 매출 대부분을 차지하는 파레토 구조(평균≫중앙값)임을 확인했고, 나머지 피처는
> 이런 치우침이 없어 스케일링만으로 충분합니다.


## 주의사항: 다중공선성 (모델 선택 시 참고)

`recency_days`와 `avg_days_between_orders`는 상관계수 **0.72**로 사실상 겹치는 정보입니다.

- **트리 계열(XGBoost, RandomForest 등) 사용 시**: 신경 쓰지 않아도 됩니다. 10개 피처 그대로 사용하세요.
- **로지스틱 회귀 등 선형 계열 사용 시**: 둘 중 하나만 사용하는 것을 권장합니다.
  `recency_days`가 계산식이 더 단순(단일 지표)하고 해석이 쉬우므로 우선 권장하나,
  `avg_days_between_orders`가 이탈과의 상관은 근소하게 더 높습니다(0.37 vs 0.35).
  VIF를 직접 확인해 결정해도 좋습니다.

상세 상관관계는 `02_eda_customer.ipynb` 최종 히트맵 참고.

## Data Card

| 항목 | 작성 내용 |
|---|---|
| 데이터 이름 | Online Retail II |
| 출처 URL | UCI ML Repository (https://archive.ics.uci.edu/dataset/502/online+retail+ii), Kaggle 미러 `mashlyn/online-retail-ii-uci` |
| 실제/합성 여부 | 실제 데이터 (영국 온라인 도매상 실거래 로그) |
| 라이선스 | CC BY 4.0 |
| 수집 기간 | 2009-12-01 ~ 2011-12-09 (약 2년) |
| 행·열 수 | 원본 1,067,371행 × 8열 → 정제 후 802,632행 → 고객 스냅샷 4,320행 × 10피처 |
| 분석 단위 | 고객 1명 = 1행 (원본은 거래 라인 단위, 09에서 고객 단위로 집계) |
| Target | `churn` (1=이탈, 0=잔류) |
| Target 생성 규칙 | 기준일(2011-09-10) 이전 365일 내 구매 이력 있는 활성 고객 대상, 기준일 이후 90일간 재구매 없으면 이탈(1). 윈도우(90일)는 재구매 간격 분포의 75~90% 지점에서 도출 |
| 관찰 기간 | 데이터 시작 ~ 기준일(2011-09-10), 취소 포함 원본 기준 순매출 계산 |
| 결과 기간 | 기준일 이후 90일 (정제된 실구매 데이터 기준 재구매 여부 판정) |
| 주요 Feature | recency_days, frequency, distinct_products, net_revenue, tenure_days, avg_days_between_orders, has_return, recent_activity_ratio, is_uk, is_low_value (10개) |
| 클래스 비율 | 이탈 49.4% / 잔류 50.6% (균형에 가까움, 별도 샘플링 불필요) |
| 개인정보 포함 여부 | 없음 — CustomerID는 익명 숫자, 이름·이메일 등 식별정보 없음 |
| 예상 위험 | 취소-구매 쌍 미상쇄 시 유령 매출 발생(순매출 계산으로 해결), CustomerID 결측 22.8%가 정상구매(비회원)인지 검증 필요(3중 검증 완료), 컬럼-값 순서 밀림·is_low_value 데이터 누수 등 파이프라인 버그(발견 및 수정 완료), recency_days-avg_days_between_orders 다중공선성(상관 0.72, 선형모델 사용 시 주의) |