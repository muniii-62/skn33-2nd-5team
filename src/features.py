"""고객 스냅샷(RFM + 파생피처 + 이탈 라벨) 생성 함수 모음"""

import pandas as pd
from src.data import load_raw, load_sales


def _prepare_base():
    """상품 라인 + 회원(CustomerID 보유)만 남긴 원본 (취소 포함, 순매출 계산용)"""
    raw = load_raw()
    is_product = raw["StockCode"].astype(str).str.match(r"^\d{5}[A-Za-z]*$")
    prod = raw[is_product & raw["CustomerID"].notna()].copy()
    prod["CustomerID"] = prod["CustomerID"].astype(int)
    return prod


def make_snapshot(cutoff: pd.Timestamp, window: int, active_days: int = 365) -> pd.DataFrame:
    """
    기준일(cutoff)과 예측 윈도우(window)를 받아 고객 스냅샷을 생성한다.

    - 관찰구간(cutoff 이전, 취소 포함)으로 RFM + 파생 피처 계산 (순매출 기준)
    - 활성 조건: cutoff 이전 active_days일 내 구매 이력 존재
    - 라벨: cutoff 이후 window일 내 재구매 없으면 이탈(1)
    """
    prod = _prepare_base()
    sales = load_sales()

    obs = prod[prod["InvoiceDate"] <= cutoff]
    fut = sales[(sales["InvoiceDate"] > cutoff) &
                (sales["InvoiceDate"] <= cutoff + pd.Timedelta(days=window))]

    last_purchase = obs.groupby("CustomerID")["InvoiceDate"].max()
    active = last_purchase[last_purchase >= cutoff - pd.Timedelta(days=active_days)].index
    obs_active = obs[obs["CustomerID"].isin(active)]

    # --- RFM 기본 ---
    snap = obs_active.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        frequency=("Invoice", "nunique"),
        net_revenue=("Quantity", lambda s: (s * obs_active.loc[s.index, "Price"]).sum()),
    ).reset_index()

    snap["recency_days"] = (cutoff - snap["last_purchase"]).dt.days
    snap["tenure_days"] = (cutoff - snap["first_purchase"]).dt.days

    # --- 09 파생 피처 ---
    extra = obs_active.groupby("CustomerID").agg(
        distinct_products=("StockCode", "nunique"),
        total_qty=("Quantity", lambda s: s.abs().sum()),
        return_qty=("Quantity", lambda s: s[s < 0].abs().sum()),
        country=("Country", lambda s: s.mode().iloc[0]),
    ).reset_index()
    extra["return_ratio"] = extra["return_qty"] / extra["total_qty"]

    snap = snap.merge(extra[["CustomerID", "distinct_products", "return_ratio", "country"]],
                       on="CustomerID", how="left")
    snap["avg_order_value"] = snap["net_revenue"] / snap["frequency"]

    # --- 09 확장 피처 ---
    snap["avg_days_between_orders"] = snap["tenure_days"] / snap["frequency"]
    snap["has_return"] = (snap["return_ratio"] > 0).astype(int)

    recent_start = cutoff - pd.Timedelta(days=90)
    recent_freq = (obs_active[obs_active["InvoiceDate"] >= recent_start]
                   .groupby("CustomerID")["Invoice"].nunique())
    snap["recent_activity_ratio"] = (snap["CustomerID"].map(recent_freq).fillna(0)
                                      / snap["frequency"])

    # --- 이진화 피처 ---
    q20 = snap["avg_order_value"].quantile(0.2)
    snap["is_low_value"] = (snap["avg_order_value"] <= q20).astype(int)
    snap["is_uk"] = (snap["country"] == "United Kingdom").astype(int)

    # --- 라벨 ---
    returned = fut["CustomerID"].unique()
    snap["churn"] = (~snap["CustomerID"].isin(returned)).astype(int)

    return snap


if __name__ == "__main__":
    snapshot_90 = make_snapshot(pd.Timestamp("2011-09-10"), window=90)
    print("90일 스냅샷:", snapshot_90.shape, "| 이탈률:", snapshot_90["churn"].mean().round(3))
    print(snapshot_90.columns.tolist())