import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent      # streamlit_app 폴더 자신
PROJECT_ROOT = APP_DIR.parent                   # 프로젝트 루트

sys.path.append(str(APP_DIR))
sys.path.append(str(PROJECT_ROOT))

import streamlit as st

from model_loader import load_model
from tabs import individual_prediction, risk_segments, decile_analysis, roi_simulator, feature_importance

st.set_page_config(page_title="고객 이탈 예측", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.06);
        border-radius: 10px !important;
    }
    button[data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model, preprocessor = load_model()

st.markdown(
    """
    <div style="margin-bottom: 18px;">
        <span style="font-size: 26px; font-weight: 700;">🛒 이커머스 고객 이탈 예측</span>
        <div style="color: #2F80ED; font-size: 13px; font-weight: 600; margin-top: 4px;">
            RandomForest 기반 이탈 예측 · 위험고객 세분화 캠페인 대시보드
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "위험고객 세분화", "개별 고객 예측", "Decile/Lift 분석", "ROI 시뮬레이터", "Feature Importance",
])

with tab1:
    risk_segments.render(model, preprocessor)

with tab2:
    individual_prediction.render(model, preprocessor)

with tab3:
    decile_analysis.render(model, preprocessor)

with tab4:
    roi_simulator.render(model, preprocessor)

with tab5:
    feature_importance.render(model, preprocessor)