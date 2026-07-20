"""고객 스냅샷 로드 + 이탈확률 계산 공용 로직.
risk_segments 탭과 roi_simulator 탭이 같은 고객 모집단을 써야 하므로 여기로 분리."""

import json

import pandas as pd
import streamlit as st

from config import FEATURE_ORDER, PROJECT_ROOT
from src.features import make_snapshot

WINDOW = 90  # 라벨 계산용 (여기선 미사용, make_snapshot 인터페이스상 필수 인자)


@st.cache_data(persist="disk", show_spinner="고객 데이터 집계 중... (최초 1회만 오래 걸림)")
def load_customer_table():
    """raw 데이터에서 고객 스냅샷(원본 스케일)을 만들어 반환. 이탈확률은 아직 없음."""
    raw_max_date = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "online_retail_II.csv",
        encoding="ISO-8859-1", usecols=["InvoiceDate"]
    )["InvoiceDate"].max()
    cutoff = pd.to_datetime(raw_max_date)

    snap = make_snapshot(cutoff, window=WINDOW)

    # cutoff 이후 데이터가 없어 make_snapshot이 계산한 churn 라벨은 전부 1(이탈)로 나옴 —
    # 실제 라벨이 아니라 계산 부산물이므로 여기서 명시적으로 버려서 향후 오용 방지.
    snap = snap.drop(columns=["churn"])

    # 학습 시 Train만으로 계산한 is_low_value 임계값(q20)을 그대로 재사용 — 데이터 누수 없이
    # 재현 (prepare_data.py가 data/preprocessed/is_low_value_threshold.json에 저장해둠).
    threshold_path = PROJECT_ROOT / "data" / "preprocessed" / "is_low_value_threshold.json"
    with open(threshold_path) as f:
        q20 = json.load(f)["avg_order_value_q20"]
    snap["is_low_value"] = (snap["avg_order_value"] <= q20).astype(int)

    return snap


def score_customers(snap: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    """스냅샷에 이탈확률 컬럼을 붙여서 반환 (원본 df는 건드리지 않음)."""
    snap = snap.copy()
    input_df = snap[FEATURE_ORDER]
    processed = preprocessor.transform(input_df)
    processed_df = pd.DataFrame(processed, columns=FEATURE_ORDER)
    snap["이탈확률"] = model.predict_proba(processed_df)[:, 1]
    return snap


def segment(row) -> str:
    """룰 기반 3분류: 첫구매 / 이탈위험 / 장기주기(정상)."""
    if row["frequency"] == 1:
        return "첫 구매 고객"
    if row["avg_days_between_orders"] < 90 and row["recency_days"] >= 90:
        return "이탈 위험 높음"
    if row["avg_days_between_orders"] >= 90:
        return "장기 구매 주기"
    return "정상"