import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent      # streamlit_app 폴더 자신
PROJECT_ROOT = APP_DIR.parent                   # 프로젝트 루트

sys.path.append(str(APP_DIR))
sys.path.append(str(PROJECT_ROOT))

import streamlit as st

from model_loader import load_model
from tabs import individual_prediction

st.set_page_config(page_title="고객 이탈 예측", layout="wide")

model, preprocessor = load_model()

st.title("🛒 이커머스 고객 이탈 예측")

tab1, tab2 = st.tabs(["개별 고객 예측", "위험고객 세분화"])

with tab1:
    individual_prediction.render(model, preprocessor)

with tab2:
    st.info("작업 중입니다. (risk_segments.py 추가 예정)")