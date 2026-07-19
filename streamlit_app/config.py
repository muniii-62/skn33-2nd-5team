from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "rf_prototype.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "final" / "preprocessor_prototype.joblib"

FEATURE_ORDER = [
    "net_revenue", "recency_days", "frequency", "distinct_products",
    "tenure_days", "avg_days_between_orders", "is_low_value", "is_uk",
    "has_return", "recent_activity_ratio",
]