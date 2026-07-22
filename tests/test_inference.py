"""저장된 배포 파이프라인의 최소 추론 스모크 테스트."""

import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.config import FEATURE_ORDER, PIPELINE_PATH  # noqa: E402


class InferenceSmokeTest(unittest.TestCase):
    def test_saved_pipeline_predicts_one_customer(self):
        pipeline = joblib.load(PIPELINE_PATH)
        customer = pd.DataFrame(
            [
                {
                    "net_revenue": 500.0,
                    "recency_days": 60,
                    "frequency": 5,
                    "distinct_products": 12,
                    "tenure_days": 300,
                    "avg_days_between_orders": 60.0,
                    "is_low_value": 0,
                    "is_uk": 1,
                    "has_return": 0,
                    "recent_activity_ratio": 0.2,
                }
            ],
            columns=FEATURE_ORDER,
        )

        probability = float(pipeline.predict_proba(customer)[0, 1])

        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
