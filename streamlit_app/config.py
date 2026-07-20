from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "rf_prototype.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "final" / "preprocessor_prototype.joblib"

FEATURE_ORDER = [
    "net_revenue", "recency_days", "frequency", "distinct_products",
    "tenure_days", "avg_days_between_orders", "is_low_value", "is_uk",
    "has_return", "recent_activity_ratio",
]

FEATURE_LABELS = {
    "net_revenue": "순매출",
    "recency_days": "최근 구매 후 경과일",
    "frequency": "구매 횟수",
    "distinct_products": "구매 상품 종류 수",
    "tenure_days": "가입 후 경과일",
    "avg_days_between_orders": "평균 구매 간격",
    "is_low_value": "저가치 고객 여부",
    "is_uk": "UK 거주 여부",
    "has_return": "취소 경험 여부",
    "recent_activity_ratio": "최근 구매 활동 비중",
}