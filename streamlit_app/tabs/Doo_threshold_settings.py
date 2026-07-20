"""Doo — 마케팅·CRM 담당자를 위한 캠페인 대상 선정 기준 화면.

[작성 구분]
- 작성자 식별명: Doo
- 신규 파일명: Doo_threshold_settings.py

[Doo 작업 내용]
1. 적극·균형·선별 관리 프리셋 제공
2. 조정 중인 기준과 실제 적용된 기준을 session_state에서 분리
3. `기준 적용하기`를 눌렀을 때만 다른 대시보드 화면에 반영
4. Validation 기반 비즈니스 핵심 지표와 기준 변화 그래프 제공
5. 고객 선정 결과를 비율 막대로 시각화
6. 전문 지표와 긴 설명은 접어서 표시

[Doo 연동 수정 파일]
- app.py: 상단 메뉴 조건부 렌더링 및 선택값·적용값 상태 초기화
- config.py: 최종 모델의 권장 기준과 고위험 기준 공통 상수 정의
- individual_prediction.py: 실제 적용 기준을 개별 고객 판정에 반영
- risk_segments.py: 실제 적용 기준으로 모델 위험군 필터링
- roi_simulator.py: 실제 적용 기준 기반 캠페인 대상 선정 지원

모델, 전처리, Threshold별 성능 계산 방식은 변경하지 않습니다.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import DEFAULT_THRESHOLD, PROJECT_ROOT


PRESETS = {
    "적극 관리": {
        "threshold": 0.30,
        "description": "더 많은 이탈 고객을 발견하지만 캠페인 대상과 비용이 증가합니다.",
    },
    "균형 관리": {
        "threshold": DEFAULT_THRESHOLD,
        "description": "이탈 고객 발견률과 선정 고객 적중률의 균형을 고려합니다.",
    },
    "선별 관리": {
        "threshold": 0.65,
        "description": "이탈 가능성이 높은 고객에게 집중해 캠페인 비용을 줄입니다.",
    },
}


@st.cache_data
def load_validation_data():
    """[Doo 작업] 현재 저장 모델과 같은 Validation 데이터를 불러옵니다."""
    data_dir = PROJECT_ROOT / "data" / "preprocessed"
    X_val = pd.read_csv(data_dir / "X_val.csv")
    y_val = pd.read_csv(data_dir / "y_val.csv")["churn"]
    return X_val, y_val


def _metrics_at_threshold(y_true, probabilities, threshold):
    """[Doo 작업] 기존 방식대로 Threshold별 지표와 혼동행렬 값을 계산합니다."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "predicted_risk": int(predictions.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _select_preset(threshold):
    """[Doo 작업] 프리셋을 조정 중인 기준에만 반영합니다."""
    st.session_state["draft_threshold"] = threshold
    st.session_state["threshold_apply_success"] = False


def _mark_pending():
    """[Doo 작업] 슬라이더 변경 시 아직 적용되지 않은 상태임을 기록합니다."""
    st.session_state["threshold_apply_success"] = False


def _apply_threshold():
    """[Doo 작업] 적용 버튼을 눌렀을 때만 전체 대시보드 기준을 갱신합니다."""
    st.session_state["applied_threshold"] = st.session_state["draft_threshold"]
    st.session_state["threshold_apply_success"] = True


def _metric_card(column, label, value, description):
    """비즈니스 지표와 짧은 해석을 같은 카드에 표시합니다."""
    with column:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.markdown(f"<div style='font-size:30px;font-weight:700'>{value}</div>", unsafe_allow_html=True)
            st.caption(description)


def _strategy_name(threshold):
    if threshold <= 0.35:
        return "적극 관리"
    if threshold <= 0.55:
        return "균형 관리"
    return "선별 관리"


def _build_sweep(y_val, probabilities):
    rows = []
    for candidate in np.arange(0.10, 0.91, 0.01):
        result = _metrics_at_threshold(y_val, probabilities, candidate)
        rows.append(
            {
                "Threshold": round(float(candidate), 2),
                "이탈 고객 발견률": result["recall"],
                "선정 고객 적중률": result["precision"],
                "발견·적중 균형 점수": result["f1"],
                "관리 대상 고객 수": result["predicted_risk"],
                "놓치는 이탈 고객 수": result["fn"],
                "추가 관리 유지 고객 수": result["fp"],
            }
        )
    return pd.DataFrame(rows)


def _render_threshold_chart(sweep_df, selected_threshold, applied_threshold, selected_metrics):
    """전체 기준 곡선과 조정 중·적용 기준, Recall 목표선을 함께 표시합니다."""
    id_columns = [
        "Threshold",
        "관리 대상 고객 수",
        "놓치는 이탈 고객 수",
        "추가 관리 유지 고객 수",
    ]
    chart_df = sweep_df.melt(
        id_vars=id_columns,
        value_vars=["이탈 고객 발견률", "선정 고객 적중률", "발견·적중 균형 점수"],
        var_name="지표",
        value_name="점수",
    )
    metric_colors = alt.Scale(
        domain=["발견·적중 균형 점수", "선정 고객 적중률", "이탈 고객 발견률"],
        range=["#2F80ED", "#71B7FF", "#FF4B4B"],
    )
    shared_tooltip = [
        alt.Tooltip("Threshold:Q", title="선정 기준", format=".0%"),
        alt.Tooltip("지표:N"),
        alt.Tooltip("점수:Q", format=".1%"),
        alt.Tooltip("관리 대상 고객 수:Q", format=","),
        alt.Tooltip("놓치는 이탈 고객 수:Q", format=","),
        alt.Tooltip("추가 관리 유지 고객 수:Q", format=","),
    ]
    lines = (
        alt.Chart(chart_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "Threshold:Q",
                scale=alt.Scale(domain=[0.10, 0.90]),
                axis=alt.Axis(format=".0%", title="캠페인 대상 선정 기준"),
            ),
            y=alt.Y(
                "점수:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", title="예상 비율"),
            ),
            color=alt.Color("지표:N", scale=metric_colors, title=None),
            tooltip=shared_tooltip,
        )
    )

    goal_rule = (
        alt.Chart(pd.DataFrame({"목표": [0.80]}))
        .mark_rule(color="#27AE60", strokeDash=[4, 4], strokeWidth=2)
        .encode(y="목표:Q")
    )
    goal_label = (
        alt.Chart(pd.DataFrame({"Threshold": [0.11], "목표": [0.80]}))
        .mark_text(align="left", dy=-8, color="#218C4A", fontWeight="bold")
        .encode(x="Threshold:Q", y="목표:Q", text=alt.value("이탈 고객 발견률 목표 80%"))
    )

    selected_rule = (
        alt.Chart(pd.DataFrame({"Threshold": [selected_threshold]}))
        .mark_rule(color="#F2994A", strokeDash=[6, 4], strokeWidth=3)
        .encode(x="Threshold:Q")
    )
    selected_label = (
        alt.Chart(pd.DataFrame({"Threshold": [selected_threshold], "점수": [0.97]}))
        .mark_text(align="left", dx=6, color="#C96C12", fontWeight="bold")
        .encode(
            x="Threshold:Q",
            y="점수:Q",
            text=alt.value(f"조정 중 {selected_threshold:.0%}"),
        )
    )
    applied_rule = (
        alt.Chart(pd.DataFrame({"Threshold": [applied_threshold]}))
        .mark_rule(color="#333333", strokeWidth=2)
        .encode(x="Threshold:Q")
    )
    applied_label = (
        alt.Chart(pd.DataFrame({"Threshold": [applied_threshold], "점수": [0.91]}))
        .mark_text(align="left", dx=6, color="#333333", fontWeight="bold")
        .encode(
            x="Threshold:Q",
            y="점수:Q",
            text=alt.value(f"실제 적용 {applied_threshold:.0%}"),
        )
    )

    selected_points_df = pd.DataFrame(
        {
            "Threshold": [selected_threshold] * 3,
            "지표": ["이탈 고객 발견률", "선정 고객 적중률", "발견·적중 균형 점수"],
            "점수": [selected_metrics["recall"], selected_metrics["precision"], selected_metrics["f1"]],
            "관리 대상 고객 수": [selected_metrics["predicted_risk"]] * 3,
            "놓치는 이탈 고객 수": [selected_metrics["fn"]] * 3,
            "추가 관리 유지 고객 수": [selected_metrics["fp"]] * 3,
        }
    )
    selected_points = (
        alt.Chart(selected_points_df)
        .mark_point(filled=True, size=130, stroke="white", strokeWidth=2)
        .encode(
            x="Threshold:Q",
            y="점수:Q",
            color=alt.Color("지표:N", scale=metric_colors, title=None),
            tooltip=shared_tooltip,
        )
    )

    chart = (
        lines
        + goal_rule
        + goal_label
        + applied_rule
        + applied_label
        + selected_rule
        + selected_label
        + selected_points
    ).properties(height=360)
    st.altair_chart(chart, width="stretch")
    st.caption(
        "🟠 현재 조정 중인 기준  ·  ⚫ 실제 전체 대시보드 적용 기준  ·  "
        "🟢 실제 이탈 고객 발견률 80% 목표선"
    )


def _render_selection_bars(metrics):
    """고객 선정 결과를 두 개의 100% 누적 막대로 표현합니다."""
    rows = [
        {"구분": "실제 이탈 고객", "결과": "발견한 고객", "고객 수": metrics["tp"]},
        {"구분": "실제 이탈 고객", "결과": "놓친 고객", "고객 수": metrics["fn"]},
        {"구분": "캠페인 대상 고객", "결과": "실제 이탈 고객", "고객 수": metrics["tp"]},
        {"구분": "캠페인 대상 고객", "결과": "실제 유지 고객", "고객 수": metrics["fp"]},
    ]
    result_df = pd.DataFrame(rows)
    selection_chart = (
        alt.Chart(result_df)
        .mark_bar(size=42, cornerRadius=5)
        .encode(
            y=alt.Y("구분:N", title=None, sort=["실제 이탈 고객", "캠페인 대상 고객"]),
            x=alt.X("sum(고객 수):Q", stack="normalize", axis=alt.Axis(format=".0%"), title="구성 비율"),
            color=alt.Color(
                "결과:N",
                scale=alt.Scale(
                    domain=["발견한 고객", "놓친 고객", "실제 이탈 고객", "실제 유지 고객"],
                    range=["#2F80ED", "#FFB4A9", "#27AE60", "#F2C94C"],
                ),
                title=None,
            ),
            tooltip=["구분:N", "결과:N", alt.Tooltip("고객 수:Q", format=",")],
        )
        .properties(height=150)
    )
    st.altair_chart(selection_chart, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"**실제 이탈 고객 {metrics['tp'] + metrics['fn']:,}명**  \n"
            f"발견한 고객 **{metrics['tp']:,}명** · 놓친 고객 **{metrics['fn']:,}명**"
        )
    with right:
        st.markdown(
            f"**캠페인 대상 고객 {metrics['predicted_risk']:,}명**  \n"
            f"실제 이탈 고객 **{metrics['tp']:,}명** · 실제 유지 고객 **{metrics['fp']:,}명**"
        )


def render(model):
    """[Doo 작업] 캠페인 대상 선정 기준 화면 전체를 렌더링합니다."""
    if "threshold_apply_success" not in st.session_state:
        st.session_state["threshold_apply_success"] = False

    # [Doo 작업] 사용자용 화면에서는 작성자 표시를 제거하고 기능 중심 제목만 보여줍니다.
    # 작업자 구분을 위한 Doo 표시는 파일명과 코드 내부 주석에 유지합니다.
    st.markdown("### 🎯 캠페인 대상 선정 기준")
    st.markdown(
        "이탈 가능성이 몇 % 이상인 고객에게 캠페인을 진행할지 결정합니다.  \n"
        "적용한 기준은 위험고객 세분화, 개별 고객 예측, ROI 화면에 공통 적용됩니다."
    )
    st.info(
        "화면의 수치는 과거 Validation 고객 데이터를 기준으로 계산한 예상 성능입니다. "
        "실제 캠페인 운영 성과를 보장하지 않습니다."
    )

    X_val, y_val = load_validation_data()
    probabilities = model.predict_proba(X_val)[:, 1]

    # 1) 운영 전략 선택
    st.markdown("#### 1. 운영 전략 선택")
    preset_columns = st.columns(3)
    for column, (name, preset) in zip(preset_columns, PRESETS.items()):
        with column:
            is_selected = abs(st.session_state["draft_threshold"] - preset["threshold"]) < 1e-9
            st.button(
                f"{name} · {preset['threshold']:.0%}",
                key=f"preset_{name}",
                type="primary" if is_selected else "secondary",
                on_click=_select_preset,
                args=(preset["threshold"],),
                width="stretch",
            )
            st.caption(preset["description"])

    # 2) 기준 슬라이더
    st.markdown("#### 2. 캠페인 대상 기준 조정")
    st.select_slider(
        "이탈 가능성이 몇 % 이상이면 관리 대상으로 선정할까요?",
        options=[round(value, 2) for value in np.arange(0.10, 0.91, 0.01)],
        key="draft_threshold",
        format_func=lambda value: f"{value:.0%}",
        on_change=_mark_pending,
        help="슬라이더를 움직여도 바로 적용되지 않습니다. 아래 적용 버튼을 눌러야 반영됩니다.",
    )

    selected_threshold = st.session_state["draft_threshold"]
    applied_threshold = st.session_state["applied_threshold"]
    selected_metrics = _metrics_at_threshold(y_val, probabilities, selected_threshold)
    recommended_metrics = _metrics_at_threshold(y_val, probabilities, DEFAULT_THRESHOLD)

    # 3) 선택값과 실제 적용값 분리
    status_left, status_right = st.columns(2)
    status_left.metric(
        "현재 조정 중인 기준",
        f"{selected_threshold:.0%}",
        help="그래프와 예상 지표를 미리 확인하는 값입니다.",
    )
    status_right.metric(
        "실제 전체 대시보드 적용 기준",
        f"{applied_threshold:.0%}",
        help="위험고객·개별 예측·ROI 화면이 실제로 사용하는 값입니다.",
    )

    if selected_threshold != applied_threshold:
        st.warning("현재 선택한 기준은 아직 적용되지 않았습니다.")
    elif st.session_state["threshold_apply_success"]:
        st.success(
            "캠페인 선정 기준이 적용되었습니다. "
            "위험고객 세분화, 개별 고객 예측, ROI 화면에도 동일하게 반영됩니다."
        )
    else:
        st.caption("현재 선택값과 실제 적용값이 같습니다.")

    st.button(
        "기준 적용하기",
        type="primary",
        on_click=_apply_threshold,
        width="stretch",
    )

    # 권장 기준 설명은 선택값과 별개로 항상 확인할 수 있게 유지합니다.
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
                이탈 고객을 80% 이상 발견하면서 캠페인 대상 규모와 적중률의 균형을 고려한 기준입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4) 핵심 지표 — 캠페인 담당자 우선순위에 맞춰 6개만 표시합니다.
    st.markdown(f"#### 현재 조정 중인 {selected_threshold:.0%} 기준 예상 결과")
    first_row = st.columns(3)
    _metric_card(
        first_row[0], "관리 대상 고객 수", f"{selected_metrics['predicted_risk']:,}명",
        "캠페인을 진행하게 될 예상 고객 수",
    )
    _metric_card(
        first_row[1], "실제 이탈 고객 발견률", f"{selected_metrics['recall']:.1%}",
        "목표 80% 이상 충족" if selected_metrics["recall"] >= 0.80 else "목표 80%보다 낮음",
    )
    _metric_card(
        first_row[2], "놓칠 수 있는 이탈 고객 수", f"{selected_metrics['fn']:,}명",
        "실제 이탈하지만 캠페인 대상에서 제외되는 고객",
    )
    second_row = st.columns(3)
    sample_size = 3
    estimated_hits = round(selected_metrics["precision"] * sample_size)
    precision_description = (
        f"선정 고객 {sample_size}명 중 약 {estimated_hits}명이 실제 이탈 고객"
        if selected_metrics["predicted_risk"] > 0 else "선정된 위험 고객이 없습니다"
    )
    _metric_card(
        second_row[0], "선정 고객 적중률", f"{selected_metrics['precision']:.1%}",
        precision_description,
    )
    _metric_card(
        second_row[1], "추가로 관리하는 유지 고객 수", f"{selected_metrics['fp']:,}명",
        "실제로는 유지하지만 캠페인 대상에 포함되는 고객",
    )
    _metric_card(
        second_row[2], "발견·적중 균형 점수", f"{selected_metrics['f1']:.3f}",
        "발견률과 적중률을 함께 고려한 비교 점수",
    )

    # 5) 기준 변화 그래프
    st.markdown("#### 기준 변화에 따른 예상 결과")
    sweep_df = _build_sweep(y_val, probabilities)
    _render_threshold_chart(sweep_df, selected_threshold, applied_threshold, selected_metrics)
    st.markdown(
        f"현재 조정 기준에서는 관리 대상 **{selected_metrics['predicted_risk']:,}명**, "
        f"놓치는 이탈 고객 **{selected_metrics['fn']:,}명**, "
        f"추가 관리 유지 고객 **{selected_metrics['fp']:,}명**으로 예상됩니다."
    )

    # 6) 고객 선정 결과 시각화
    st.markdown("#### 고객 선정 결과")
    _render_selection_bars(selected_metrics)

    # 7) 긴 설명과 전문 지표는 기본 화면에서 접어둡니다.
    with st.expander("선정 기준을 어떻게 정하면 좋을까요?"):
        st.markdown(
            "- **적극 관리(30%)**: 더 많은 이탈 고객을 발견하지만 캠페인 대상과 비용이 증가합니다.\n"
            f"- **균형 관리({DEFAULT_THRESHOLD:.0%}, 권장)**: 이탈 고객 발견률과 선정 고객 적중률의 균형을 고려합니다.\n"
            "- **선별 관리(65%)**: 이탈 가능성이 높은 고객에게 집중해 캠페인 비용을 줄입니다.\n\n"
            "캠페인 비용이 낮거나 이탈 고객을 놓치는 손실이 크다면 적극 관리가 적합합니다. "
            "캠페인 비용이 높다면 선별 관리 결과와 ROI 화면을 함께 확인하세요."
        )

    with st.expander("전문 지표는 무엇을 의미하나요?"):
        st.markdown(
            f"- **Threshold(캠페인 선정 기준)**: `{selected_threshold:.2f}`\n"
            f"- **Recall(실제 이탈 고객 발견률)**: `{selected_metrics['recall']:.3f}`\n"
            f"- **Precision(선정 고객 적중률)**: `{selected_metrics['precision']:.3f}`\n"
            f"- **F1-score(발견·적중 균형 점수)**: `{selected_metrics['f1']:.3f}`\n"
            f"- **Accuracy(전체 판정 정확도)**: `{selected_metrics['accuracy']:.3f}`\n"
            f"- **TP(발견한 실제 이탈 고객)**: `{selected_metrics['tp']}`명\n"
            f"- **TN(유지 고객으로 올바르게 제외)**: `{selected_metrics['tn']}`명\n"
            f"- **FP / False Positive(추가로 관리하는 유지 고객)**: `{selected_metrics['fp']}`명\n"
            f"- **FN / False Negative(놓칠 수 있는 이탈 고객)**: `{selected_metrics['fn']}`명"
        )
        matrix_df = pd.DataFrame(
            [
                [selected_metrics["tn"], selected_metrics["fp"]],
                [selected_metrics["fn"], selected_metrics["tp"]],
            ],
            index=["실제 유지", "실제 이탈"],
            columns=["유지로 예측", "이탈로 예측"],
        )
        st.markdown("**Confusion Matrix(혼동행렬)**")
        st.dataframe(matrix_df, width="stretch")
