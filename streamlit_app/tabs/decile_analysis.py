"""Decile / Lift 분석 탭 — Validation셋 기준으로 모델의 위험군 분리력을 검증.
(위험고객 세분화·ROI 탭과 달리 실제 라벨이 있는 X_val/y_val을 사용 — 방법론 재현용)"""

import pandas as pd
import streamlit as st

from config import FEATURE_ORDER, PROJECT_ROOT


@st.cache_data
def load_val_data():
    val_dir = PROJECT_ROOT / "data" / "preprocessed"
    X_val = pd.read_csv(val_dir / "X_val.csv")
    y_val = pd.read_csv(val_dir / "y_val.csv")["churn"]
    return X_val, y_val


def render(model, preprocessor):
    st.markdown("### 📊 Decile / Lift 분석")
    st.caption(
        "Validation셋 고객을 이탈확률이 높은 순으로 10등분(decile)했을 때, "
        "각 구간의 실제 이탈률입니다. 모델이 위험군을 얼마나 잘 골라내는지 보여줍니다."
    )

    X_val, y_val = load_val_data()
    # X_val은 이미 학습 때와 동일한 전처리(로그+스케일링)가 끝난 상태 — preprocessor 재적용 불필요
    proba = model.predict_proba(X_val[FEATURE_ORDER])[:, 1]

    df = pd.DataFrame({"churn": y_val.values, "proba": proba})
    df = df.sort_values("proba", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, 10, labels=False) + 1  # 1 = 최상위 위험군

    overall_rate = df["churn"].mean()

    decile_stats = df.groupby("decile").agg(
        고객수=("churn", "size"),
        실제이탈률=("churn", "mean"),
    ).reset_index()
    decile_stats["Lift"] = (decile_stats["실제이탈률"] / overall_rate).round(2)
    decile_stats["실제이탈률(%)"] = (decile_stats["실제이탈률"] * 100).round(1)

    st.bar_chart(decile_stats.set_index("decile")["실제이탈률(%)"])
    st.caption(f"전체 평균 이탈률: {overall_rate:.1%} (점선 없이 참고용 기준선)")

    st.dataframe(
        decile_stats[["decile", "고객수", "실제이탈률(%)", "Lift"]],
        hide_index=True, use_container_width=True,
    )

    st.markdown("### 🎯 Top-K Capture Rate")
    st.caption(
        "상위 K%에게만 캠페인을 돌렸을 때, 전체 이탈 고객 중 몇 %를 잡아낼 수 있는지입니다."
    )

    ks = [10, 20, 30, 40, 50, 70, 100]
    total_churners = df["churn"].sum()
    capture_rows = []
    for k in ks:
        n = int(len(df) * k / 100)
        captured = df.iloc[:n]["churn"].sum()
        capture_rows.append({"상위 K%": k, "포착률(%)": round(captured / total_churners * 100, 1)})
    capture_df = pd.DataFrame(capture_rows)

    st.line_chart(capture_df.set_index("상위 K%"))
    st.dataframe(capture_df, hide_index=True, use_container_width=True)
