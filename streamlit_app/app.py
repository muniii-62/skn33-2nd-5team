import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent      # streamlit_app 폴더 자신
PROJECT_ROOT = APP_DIR.parent                   # 프로젝트 루트

sys.path.append(str(APP_DIR))
sys.path.append(str(PROJECT_ROOT))

import streamlit as st

from model_loader import load_model
from config import DEFAULT_THRESHOLD, EVALUATION_DATASET_NAME
from customer_scoring import load_customer_table
from tabs import (
    Doo_threshold_settings,
    individual_prediction,
    risk_segments,
    roi_simulator,
)

st.set_page_config(page_title="고객 이탈 예측", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 100%;
        padding-top: 1.35rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.06);
        border-radius: 10px !important;
    }
    .doo-crm-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        width: 100%;
        margin: 0 0 10px 0;
        padding: 22px 24px;
        background: #F8FBFF;
        border: 1px solid #D8E6F3;
        border-left: 4px solid #2F80ED;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(27, 78, 125, 0.07);
    }
    .doo-crm-identity {
        display: flex;
        align-items: center;
        gap: 16px;
        min-width: 0;
    }
    .doo-crm-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 46px;
        width: 46px;
        height: 46px;
        background: #E8F2FF;
        border: 1px solid #C7DDF5;
        border-radius: 12px;
        font-size: 22px;
    }
    .doo-crm-copy {
        min-width: 0;
    }
    .doo-crm-title {
        margin: 0;
        color: #152238;
        font-size: clamp(23px, 2.2vw, 31px);
        font-weight: 750;
        letter-spacing: -0.025em;
        line-height: 1.25;
    }
    .doo-crm-subtitle {
        margin: 7px 0 0 0;
        color: #52647A;
        font-size: 14px;
        font-weight: 500;
        line-height: 1.45;
    }
    .doo-crm-status {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 8px;
        max-width: 430px;
    }
    .doo-crm-badge {
        display: inline-flex;
        align-items: center;
        min-height: 30px;
        padding: 5px 10px;
        color: #244664;
        background: #FFFFFF;
        border: 1px solid #C9DBEC;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 650;
        line-height: 1.2;
        white-space: nowrap;
    }
    div[data-testid="stButtonGroup"] {
        margin: 6px 0 14px 0;
        padding: 4px;
        background: #F7F9FC;
        border: 1px solid #D8E6F3;
        border-radius: 12px;
    }
    div[data-testid="stButtonGroup"] div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        width: 100%;
    }
    div[data-testid="stButtonGroup"] button {
        flex: 1 1 150px;
        min-width: 0;
        color: #4A5C70;
        background: transparent;
        border: 1px solid transparent;
        font-size: 15px;
        font-weight: 600;
        border-radius: 9px;
        transition: background-color 120ms ease, border-color 120ms ease;
    }
    div[data-testid="stButtonGroup"] button:hover {
        color: #244664;
        background: #EDF3F9;
        border-color: #D1DFEC;
    }
    div[data-testid="stButtonGroup"] button[aria-pressed="true"],
    div[data-testid="stButtonGroup"] button[aria-checked="true"],
    div[data-testid="stButtonGroup"] button[data-active="true"],
    div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {
        color: #FFFFFF;
        background: #2F80ED;
        border-color: #236FCC;
        box-shadow: inset 0 -3px 0 #1E61B5;
        font-weight: 700;
    }
    div[data-testid="stButtonGroup"] button[aria-pressed="true"] p,
    div[data-testid="stButtonGroup"] button[aria-checked="true"] p,
    div[data-testid="stButtonGroup"] button[data-active="true"] p,
    div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] p {
        color: inherit;
        font-weight: 700;
    }
    @media (max-width: 800px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .doo-crm-header {
            align-items: flex-start;
            flex-direction: column;
            gap: 14px;
            padding: 18px;
        }
        .doo-crm-status {
            justify-content: flex-start;
            max-width: none;
            padding-left: 62px;
        }
        div[data-testid="stButtonGroup"] button {
            flex-basis: calc(50% - 4px);
        }
    }
    @media (max-width: 480px) {
        .doo-crm-identity {
            align-items: flex-start;
        }
        .doo-crm-icon {
            flex-basis: 40px;
            width: 40px;
            height: 40px;
            font-size: 19px;
        }
        .doo-crm-status {
            padding-left: 0;
        }
        div[data-testid="stButtonGroup"] button {
            flex-basis: 100%;
        }
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

# [Doo 작업] 실제 모델·고객 데이터·적용 Threshold를 사용해 CRM 헤더 상태를 표시합니다.
# 숫자를 HTML에 고정하지 않아 데이터나 운영 기준이 바뀌면 헤더도 함께 갱신됩니다.
customer_count = len(load_customer_table())
model_class_name = type(model).__name__
model_display_name = "XGBoost" if model_class_name == "XGBClassifier" else model_class_name
applied_threshold = float(st.session_state["applied_threshold"])
evaluation_dataset_name = EVALUATION_DATASET_NAME

st.markdown(
    f"""
    <section class="doo-crm-header" role="banner" aria-label="고객 이탈 예측 대시보드 상태">
        <div class="doo-crm-identity">
            <div class="doo-crm-icon" aria-hidden="true">🛒</div>
            <div class="doo-crm-copy">
                <h1 class="doo-crm-title">이커머스 고객 이탈 예측</h1>
                <p class="doo-crm-subtitle">XGBoost 기반 이탈 예측과 CRM 캠페인 의사결정 대시보드</p>
            </div>
        </div>
        <div class="doo-crm-status" aria-label="현재 대시보드 상태">
            <span class="doo-crm-badge">모델 · {model_display_name}</span>
            <span class="doo-crm-badge">고객 · {customer_count:,}명</span>
            <span class="doo-crm-badge">적용 기준 · {applied_threshold:.0%}</span>
            <span class="doo-crm-badge">평가 · {evaluation_dataset_name}</span>
        </div>
    </section>
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
