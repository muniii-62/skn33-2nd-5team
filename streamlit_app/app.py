import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent      # streamlit_app 폴더 자신
PROJECT_ROOT = APP_DIR.parent                   # 프로젝트 루트

sys.path.append(str(APP_DIR))
sys.path.append(str(PROJECT_ROOT))

import streamlit as st

from model_loader import load_model
from config import DEFAULT_THRESHOLD
from tabs import (
    Doo_threshold_settings,
    feature_importance,
    individual_prediction,
    risk_segments,
    roi_simulator,
)

st.set_page_config(page_title="고객 이탈 예측", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.06);
        border-radius: 10px !important;
    }
    div[data-testid="stSegmentedControl"] button {
        font-size: 15px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
model, preprocessor = load_model()

# [Doo 작업] 선택값과 실제 적용값을 분리합니다.
# 다른 화면은 `applied_threshold`만 사용하므로 슬라이더를 움직이는 것만으로는
# 위험고객·개별 예측·ROI 결과가 바뀌지 않습니다.
if "applied_threshold" not in st.session_state:
    st.session_state["applied_threshold"] = DEFAULT_THRESHOLD
if "draft_threshold" not in st.session_state:
    st.session_state["draft_threshold"] = st.session_state["applied_threshold"]

st.markdown(
    """
    <div style="margin-bottom: 18px;">
        <span style="font-size: 26px; font-weight: 700;">🛒 이커머스 고객 이탈 예측</span>
        <div style="color: #2F80ED; font-size: 13px; font-weight: 600; margin-top: 4px;">
            XGBoost 기반 이탈 예측 · 위험고객 세분화 캠페인 대시보드
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# [Doo 작업] st.tabs는 모든 탭의 코드를 한 번에 실행하므로, 메뉴 선택값에 따라
# 하나의 화면만 조건부로 렌더링합니다. 다른 화면의 콘텐츠가 아래로 이어지지 않습니다.
# [Doo 작업] CRM 운영 화면에 집중하기 위해 Decile/Lift 분석 메뉴를 제거했습니다.
menu_items = [
    "캠페인 기준 설정",
    "위험고객 세분화",
    "개별 고객 예측",
    "ROI 시뮬레이터",
    "Feature Importance",
]
selected_menu = st.segmented_control(
    "대시보드 메뉴",
    menu_items,
    default=menu_items[0],
    key="dashboard_menu",
    label_visibility="collapsed",
    width="stretch",
)
st.divider()

if selected_menu == "캠페인 기준 설정":
    Doo_threshold_settings.render(model)
elif selected_menu == "위험고객 세분화":
    risk_segments.render(model, preprocessor)
elif selected_menu == "개별 고객 예측":
    individual_prediction.render(model, preprocessor)
elif selected_menu == "ROI 시뮬레이터":
    roi_simulator.render(model, preprocessor)
elif selected_menu == "Feature Importance":
    feature_importance.render(model, preprocessor)
