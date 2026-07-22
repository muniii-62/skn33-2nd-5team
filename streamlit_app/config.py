from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"



# [Doo 작업] Threshold 기능에서 사용하는 공통 기준값
# DEFAULT_THRESHOLD: 최종 XGBoost(jhd) 모델의 Validation 결과에서 Recall 0.85
# 이상을 만족하면서 F1이 가장 높은 운영 권장값입니다.
DEFAULT_THRESHOLD = 0.38

# [Doo 작업] 대시보드 헤더에 표시할 현재 평가 데이터 기준입니다.
# 헤더 HTML에 문구를 직접 고정하지 않고 프로젝트 설정에서 관리합니다.
EVALUATION_DATASET_NAME = "Validation"

FEATURE_ORDER = [
    "net_revenue", "recency_days", "frequency", "distinct_products",
    "tenure_days", "avg_days_between_orders", "is_low_value", "is_uk",
    "has_return", "recent_activity_ratio",
]

FEATURE_LABELS = {
    "net_revenue": "순매출",
    "recency_days": "최근 거래 활동 후 경과일",
    "frequency": "구매 횟수",
    "distinct_products": "구매 상품 종류 수",
    "tenure_days": "첫 구매 후 경과일",
    "avg_days_between_orders": "평균 구매 간격",
    "is_low_value": "저가치 고객 여부",
    "is_uk": "UK 거주 여부",
    "has_return": "취소 경험 여부",
    "recent_activity_ratio": "최근 구매 활동 비중",
}
