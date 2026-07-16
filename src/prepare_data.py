"""
전처리 완료 데이터 자동 생성 스크립트

실행 (프로젝트 루트에서):
    python src/data.py          # 원본 다운로드 (최초 1회, data/raw/에 저장)
    python src/prepare_data.py  # 전처리 완료 파일 생성 (data/preprocessed/에 저장)

생성 파일 (data/preprocessed/):
    X_train.csv, X_val.csv, X_test.csv   — 전처리(로그+스케일링) 완료된 피처
    y_train.csv, y_val.csv, y_test.csv   — 타깃(churn)
    preprocessor.pkl                      — 학습된 전처리 Pipeline (재사용용)

주의: Test 파일도 여기 저장은 되지만, 팀원용 로드 예시(models/example_load.py)에서는
의도적으로 불러오지 않는다. Val로 모델을 충분히 비교한 뒤,
최종 후보가 정해지면 그 모델에 한해서만 Test로 평가한다 (딱 1회).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from src.transforms import clip_negative, log1p_transform
from src.features import make_snapshot

RANDOM_STATE = 42  # 팀 전체 공통 시드 — 변경 금지 (재현성)
CUTOFF = pd.Timestamp("2011-09-10")
WINDOW = 90

FEATURE_COLS = ["recency_days", "frequency", "distinct_products",
                 "net_revenue", "tenure_days", "is_low_value", "is_uk",
                 "avg_days_between_orders", "has_return", "recent_activity_ratio"]
TARGET_COL = "churn"

LOG_COLS = ["net_revenue"]
SCALE_COLS = ["recency_days", "frequency", "distinct_products", "tenure_days",
              "avg_days_between_orders"]
PASSTHROUGH_COLS = ["is_low_value", "is_uk", "has_return", "recent_activity_ratio"]





def build_preprocessor() -> ColumnTransformer:
    log_scale_pipe = Pipeline([
        ("clip", FunctionTransformer(clip_negative, validate=True)),
        ("log", FunctionTransformer(log1p_transform, validate=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("log_scale", log_scale_pipe, LOG_COLS),
        ("scale", StandardScaler(), SCALE_COLS),
        ("passthrough", "passthrough", PASSTHROUGH_COLS),
    ])


def main():
    snapshot = make_snapshot(CUTOFF, window=WINDOW)
    X = snapshot[FEATURE_COLS]
    y = snapshot[TARGET_COL]

    # Train(60%) / Val(20%) / Test(20%), 이탈 비율 층화 유지
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)  # Train으로만 fit (데이터 누수 방지)

    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    out_dir = Path(__file__).resolve().parent.parent / "data" / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train_processed, columns=FEATURE_COLS).to_csv(out_dir / "X_train.csv", index=False)
    pd.DataFrame(X_val_processed, columns=FEATURE_COLS).to_csv(out_dir / "X_val.csv", index=False)
    pd.DataFrame(X_test_processed, columns=FEATURE_COLS).to_csv(out_dir / "X_test.csv", index=False)

    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_val.to_csv(out_dir / "y_val.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    with open(out_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("생성 완료:", sorted(p.name for p in out_dir.glob("*")))
    print(f"Train {len(X_train)}행 (이탈률 {y_train.mean():.3f})")
    print(f"Val   {len(X_val)}행 (이탈률 {y_val.mean():.3f})")
    print(f"Test  {len(X_test)}행 (이탈률 {y_test.mean():.3f}) — 최종 평가 전용, 팀원 example에서 미사용")


if __name__ == "__main__":
    main()