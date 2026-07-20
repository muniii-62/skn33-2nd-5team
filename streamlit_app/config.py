from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "rf_prototype.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "final" / "preprocessor_prototype.joblib"

# [Doo 작업] Threshold 기능에서 사용하는 공통 기준값
# DEFAULT_THRESHOLD: 현재 RandomForest Validation에서 Recall 0.80 이상을
# 만족하는 운영 권장값입니다. 최종 모델이 교체되면 다시 검증해야 합니다.
# HIGH_RISK_THRESHOLD: 개별 예측 화면의 고위험 고객 구분 기준입니다.
DEFAULT_THRESHOLD = 0.44
HIGH_RISK_THRESHOLD = 0.65

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
