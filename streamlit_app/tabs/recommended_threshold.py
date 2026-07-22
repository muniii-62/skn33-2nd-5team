"""권장 운영 기준 화면: 기존 캠페인 대상 선정 화면을 38% 기본값으로 재배치한다."""

import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_THRESHOLD
from tabs import risk_segments
from tabs.Doo_threshold_settings import (
    _metrics_at_threshold,
    _render_selection_bars,
    load_validation_data,
)


def _metric_card(column, label, value, description):
    """기존 캠페인 대상 선정 탭과 같은 형태의 결과 카드를 표시한다."""
    with column:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.markdown(
                f"<div style='font-size:30px;font-weight:700'>{value}</div>",
                unsafe_allow_html=True,
            )
            st.caption(description)


def render(model, preprocessor):
    """38% 운영 기준과 Threshold 변화의 Validation 민감도만 보여줍니다."""
    if "recommended_draft_threshold" not in st.session_state:
        st.session_state["recommended_draft_threshold"] = DEFAULT_THRESHOLD

    X_val, y_val = load_validation_data()
    probabilities = model.predict_proba(X_val)[:, 1]
    recommended_metrics = _metrics_at_threshold(y_val, probabilities, DEFAULT_THRESHOLD)

    st.markdown("### 캠페인 대상 선정")
    st.markdown(
        "최종 운영 기준은 **이탈 확률 38%**입니다. Validation에서 Recall 85% 이상을 "
        "만족하는 기준 중 F1이 가장 높은 값을 선정했습니다."
    )
    st.info(
        f"모델 검증 결과 — Validation {len(y_val):,}명 기준입니다. "
        "실제 캠페인 운영 성과를 보장하는 값은 아닙니다."
    )

    st.markdown(
        f"""
        <div style="background:#EAF3FF;border:1px solid #BBD7FF;border-radius:10px;
                    padding:16px 20px;margin:12px 0 20px 0;">
            <div style="color:#1F6FCC;font-size:13px;font-weight:700;">
                권장 기준 {DEFAULT_THRESHOLD:.0%} · 균형 관리 전략
            </div>
            <div style="font-size:16px;font-weight:700;margin:8px 0;">
                이탈 고객 발견률 {recommended_metrics['recall']:.1%} &nbsp;·&nbsp;
                선정 고객 적중률 {recommended_metrics['precision']:.1%} &nbsp;·&nbsp;
                관리 대상 고객 {recommended_metrics['predicted_risk']:,}명
            </div>
            <div style="font-size:14px;color:#425466;">
                이탈 고객 발견률 85% 이상 조건에서 캠페인 대상 규모와 적중률의 균형을 고려한 기준입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 화면 상단과 고객 선정 결과는 항상 최종 운영 기준 38%로 고정합니다.
    st.markdown(f"#### 운영 기준 {DEFAULT_THRESHOLD:.0%}의 Validation 예상 결과")
    first_row = st.columns(3)
    _metric_card(
        first_row[0], "관리 대상 고객 수", f"{recommended_metrics['predicted_risk']:,}명",
        "캠페인을 진행하게 될 예상 고객 수",
    )
    _metric_card(
        first_row[1], "실제 이탈 고객 발견률", f"{recommended_metrics['recall']:.1%}",
        "목표 85% 이상 충족" if recommended_metrics["recall"] >= 0.85 else "목표 85%보다 낮음",
    )
    _metric_card(
        first_row[2], "놓칠 수 있는 이탈 고객 수", f"{recommended_metrics['fn']:,}명",
        "실제 이탈하지만 캠페인 대상에서 제외되는 고객",
    )

    second_row = st.columns(3)
    estimated_hits = round(recommended_metrics["precision"] * 3)
    precision_description = (
        f"선정 고객 3명 중 약 {estimated_hits}명이 실제 이탈 고객"
        if recommended_metrics["predicted_risk"] > 0
        else "선정된 위험 고객이 없습니다"
    )
    _metric_card(
        second_row[0], "선정 고객 적중률", f"{recommended_metrics['precision']:.1%}", precision_description,
    )
    _metric_card(
        second_row[1], "추가로 관리하는 유지 고객 수", f"{recommended_metrics['fp']:,}명",
        "실제로는 유지하지만 캠페인 대상에 포함되는 고객",
    )
    _metric_card(
        second_row[2], "발견·적중 균형 점수", f"{recommended_metrics['f1']:.3f}",
        "발견률과 적중률을 함께 고려한 비교 점수",
    )

    st.markdown("#### 고객 선정 결과")
    _render_selection_bars(recommended_metrics)

    with st.expander("기준 변화는 어떻게 해석하나요?"):
        st.markdown(
            "- 기준을 낮추면 더 많은 이탈 고객을 발견하지만 캠페인 대상과 FP가 증가합니다.\n"
            "- 기준을 높이면 대상과 FP가 줄지만 실제 이탈 고객을 더 놓칠 수 있습니다.\n"
            f"- 아래 비교표 이외의 화면에는 비교값과 관계없이 **{DEFAULT_THRESHOLD:.0%}**가 적용됩니다."
        )

    with st.expander("지표 용어는 무엇인가요?"):
        st.markdown(
            f"- **Threshold(최종 운영 기준)**: `{DEFAULT_THRESHOLD:.2f}`\n"
            f"- **Recall(실제 이탈 고객 발견률)**: `{recommended_metrics['recall']:.3f}`\n"
            f"- **Precision(선정 고객 적중률)**: `{recommended_metrics['precision']:.3f}`\n"
            f"- **F1-score(발견·적중 균형 점수)**: `{recommended_metrics['f1']:.3f}`\n"
            f"- **TP / FN / FP**: `{recommended_metrics['tp']}`명 / "
            f"`{recommended_metrics['fn']}`명 / `{recommended_metrics['fp']}`명"
        )

    # 실제 고객 선정 결과는 비교 슬라이더와 무관하게 최종 기준 38%로 계산합니다.
    risk_segments.render_summary(model, preprocessor)

    st.divider()
    st.markdown("#### Threshold 민감도 비교")
    st.caption(
        "아래 값은 Validation 예상 결과만 바꿉니다. 고객 목록·개별 예측·ROI의 최종 기준은 38%로 고정됩니다."
    )

    st.select_slider(
        "비교용 Threshold",
        options=[round(value, 2) for value in np.arange(0.10, 0.91, 0.01)],
        key="recommended_draft_threshold",
        format_func=lambda value: f"{value:.0%}",
        help="Validation 성능 민감도만 비교하며 실제 운영 기준은 변경하지 않습니다.",
    )
    selected_threshold = float(st.session_state["recommended_draft_threshold"])
    selected_metrics = _metrics_at_threshold(y_val, probabilities, selected_threshold)

    def _signed_count(value):
        return f"{value:+,}명"

    def _signed_points(value):
        return f"{value * 100:+.1f}%p"

    def _signed_score(value):
        return f"{value:+.3f}"

    comparison_rows = [
        {
            "지표": "캠페인 대상 고객",
            f"운영 기준 {DEFAULT_THRESHOLD:.0%}": f"{recommended_metrics['predicted_risk']:,}명",
            f"비교 기준 {selected_threshold:.0%}": f"{selected_metrics['predicted_risk']:,}명",
            "차이": _signed_count(selected_metrics["predicted_risk"] - recommended_metrics["predicted_risk"]),
        },
        {
            "지표": "이탈 고객 발견률",
            f"운영 기준 {DEFAULT_THRESHOLD:.0%}": f"{recommended_metrics['recall']:.1%}",
            f"비교 기준 {selected_threshold:.0%}": f"{selected_metrics['recall']:.1%}",
            "차이": _signed_points(selected_metrics["recall"] - recommended_metrics["recall"]),
        },
        {
            "지표": "선정 고객 적중률",
            f"운영 기준 {DEFAULT_THRESHOLD:.0%}": f"{recommended_metrics['precision']:.1%}",
            f"비교 기준 {selected_threshold:.0%}": f"{selected_metrics['precision']:.1%}",
            "차이": _signed_points(selected_metrics["precision"] - recommended_metrics["precision"]),
        },
        {
            "지표": "놓치는 이탈 고객",
            f"운영 기준 {DEFAULT_THRESHOLD:.0%}": f"{recommended_metrics['fn']:,}명",
            f"비교 기준 {selected_threshold:.0%}": f"{selected_metrics['fn']:,}명",
            "차이": _signed_count(selected_metrics["fn"] - recommended_metrics["fn"]),
        },
        {
            "지표": "추가 관리 유지 고객",
            f"운영 기준 {DEFAULT_THRESHOLD:.0%}": f"{recommended_metrics['fp']:,}명",
            f"비교 기준 {selected_threshold:.0%}": f"{selected_metrics['fp']:,}명",
            "차이": _signed_count(selected_metrics["fp"] - recommended_metrics["fp"]),
        },
        {
            "지표": "F1 점수",
            f"운영 기준 {DEFAULT_THRESHOLD:.0%}": f"{recommended_metrics['f1']:.3f}",
            f"비교 기준 {selected_threshold:.0%}": f"{selected_metrics['f1']:.3f}",
            "차이": _signed_score(selected_metrics["f1"] - recommended_metrics["f1"]),
        },
    ]
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")
    st.info(
        f"비교용 Threshold {selected_threshold:.0%}는 위 표에만 반영됩니다. "
        f"실제 운영 기준은 {DEFAULT_THRESHOLD:.0%}로 고정되어 있습니다."
    )
