"""고객 스냅샷 로드 + 이탈확률 계산 공용 로직.
risk_segments 탭과 roi_simulator 탭이 같은 고객 모집단을 써야 하므로 여기로 분리."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import FEATURE_ORDER, PROJECT_ROOT
from src.features import make_snapshot

WINDOW = 90  # 라벨 계산용 (여기선 미사용, make_snapshot 인터페이스상 필수 인자)

# [Doo 작업] 원본 데이터와 저가치 기준 파일이 바뀌면 기존 디스크 캐시를
# 재사용하지 않도록 두 파일의 버전 정보를 Streamlit 캐시 키에 포함한다.
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.csv"
LOW_VALUE_THRESHOLD_PATH = (
    PROJECT_ROOT / "data" / "preprocessed" / "is_low_value_threshold.json"
)


def _file_version(path: Path) -> tuple[int, int]:
    """캐시 무효화에 사용할 파일의 수정 시각과 크기를 반환한다."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(persist="disk", show_spinner="고객 데이터 집계 중... (최초 1회만 오래 걸림)")
def _load_customer_table(raw_version: tuple[int, int], threshold_version: tuple[int, int]):
    """raw 데이터에서 고객 스냅샷(원본 스케일)을 만들어 반환. 이탈확률은 아직 없음."""
    # 파일 버전 인자는 Streamlit 캐시 키로 사용된다.
    _ = raw_version, threshold_version
    raw_max_date = pd.read_csv(
        RAW_DATA_PATH,
        encoding="ISO-8859-1", usecols=["InvoiceDate"]
    )["InvoiceDate"].max()
    cutoff = pd.to_datetime(raw_max_date)

    snap = make_snapshot(cutoff, window=WINDOW)

    # cutoff 이후 데이터가 없어 make_snapshot이 계산한 churn 라벨은 전부 1(이탈)로 나옴 —
    # 실제 라벨이 아니라 계산 부산물이므로 여기서 명시적으로 버려서 향후 오용 방지.
    snap = snap.drop(columns=["churn"])

    # 학습 시 Train만으로 계산한 is_low_value 임계값(q20)을 그대로 재사용 — 데이터 누수 없이
    # 재현 (prepare_data.py가 data/preprocessed/is_low_value_threshold.json에 저장해둠).
    with open(LOW_VALUE_THRESHOLD_PATH, encoding="utf-8") as f:
        q20 = json.load(f)["avg_order_value_q20"]
    snap["is_low_value"] = (snap["avg_order_value"] <= q20).astype(int)

    return snap


def load_customer_table():
    """[Doo 작업] 파일 버전을 캐시 키에 포함해 고객 스냅샷을 로드한다."""
    return _load_customer_table(
        _file_version(RAW_DATA_PATH),
        _file_version(LOW_VALUE_THRESHOLD_PATH),
    )


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
