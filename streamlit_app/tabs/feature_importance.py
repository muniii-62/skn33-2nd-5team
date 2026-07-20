"""Feature Importance 탭 — 모델에 내장된 feature_importances_로 빠르게 시각화.
(실시간 SHAP 계산은 앱 로딩이 느려질 수 있어 제외 — 노트북에서 별도 분석한 SHAP 결과와
방향성이 같은지는 아래 요약 텍스트로 교차 확인)"""

import altair as alt
import pandas as pd
import streamlit as st

from config import FEATURE_ORDER, FEATURE_LABELS


def render(model, preprocessor):
    st.markdown("### 🧭 어떤 정보가 이탈 예측에 중요할까?")
    st.caption("최종 XGBoost 모델이 예측할 때 각 정보를 얼마나 많이 참고하는지를 나타냅니다 (모델 내장 중요도 기준).")

    if not hasattr(model, "feature_importances_"):
        st.warning("현재 모델은 feature_importances_를 지원하지 않습니다 (트리 기반 모델에서만 제공).")
        return

    importances = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "importance": model.feature_importances_,
    })
    importances["label"] = importances["feature"].map(FEATURE_LABELS)
    importances = importances.sort_values("importance", ascending=False)

    chart = (
        alt.Chart(importances)
        .mark_bar(color="#2F80ED")
        .encode(
            x=alt.X("importance:Q", title="중요도"),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("label:N", title="정보"), alt.Tooltip("importance:Q", title="중요도", format=".4f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    top3 = importances.head(3)["label"].tolist()
    bottom2 = importances.tail(2)["label"].tolist()
    st.info(
        f"**{', '.join(top3)}** 가 예측에 가장 큰 영향을 주는 정보입니다.\n\n"
        f"반대로 **{', '.join(bottom2)}** 는 거의 참고되지 않습니다 — "
        "이는 SHAP 분석 결과와도 같은 방향입니다."
    )

    with st.expander("표로 자세히 보기"):
        show_df = importances[["label", "importance"]].rename(
            columns={"label": "정보", "importance": "중요도"}
        )
        show_df["중요도"] = show_df["중요도"].round(4)
        st.dataframe(show_df, hide_index=True, use_container_width=True)
