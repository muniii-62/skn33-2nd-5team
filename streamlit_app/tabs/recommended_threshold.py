"""권장 운영 기준 화면: 기존 캠페인 대상 선정 화면을 38% 기본값으로 재배치한다."""

import numpy as np
import streamlit as st

from config import DEFAULT_THRESHOLD
from tabs import risk_segments
from tabs.Doo_threshold_settings import (
    PRESETS,
    _build_sweep,
    _metrics_at_threshold,
    _render_selection_bars,
    _render_threshold_chart,
    load_validation_data,
)


def _select_recommended_preset(threshold):
    """새 탭 안에서만 사용하는 미리보기 기준을 프리셋 값으로 바꾼다."""
    st.session_state["recommended_draft_threshold"] = threshold
    st.session_state["recommended_threshold_apply_success"] = False


def _mark_recommended_pending():
    """슬라이더 변경 후 아직 전체 대시보드에 적용되지 않았음을 표시한다."""
    st.session_state["recommended_threshold_apply_success"] = False


def _apply_recommended_threshold():
    """새 탭에서 고른 기준을 전체 대시보드의 실제 적용 기준으로 반영한다."""
    selected_threshold = st.session_state["recommended_draft_threshold"]
    st.session_state["applied_threshold"] = selected_threshold
    # 기존 탭으로 이동했을 때도 선택값과 실제 적용값이 어긋나지 않게 맞춘다.
    st.session_state["draft_threshold"] = selected_threshold
    st.session_state["recommended_threshold_apply_success"] = True


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
    """38% 권장값으로 시작하고, 제어 영역을 화면 하단에 둔 캠페인 대상 선정 화면."""
    # 기존 탭의 조정값과 분리해, 새 탭을 처음 열면 언제나 38%부터 시작합니다.
    if "recommended_draft_threshold" not in st.session_state:
        st.session_state["recommended_draft_threshold"] = DEFAULT_THRESHOLD
    if "recommended_threshold_apply_success" not in st.session_state:
        st.session_state["recommended_threshold_apply_success"] = False

    X_val, y_val = load_validation_data()
    probabilities = model.predict_proba(X_val)[:, 1]
    selected_threshold = float(st.session_state["recommended_draft_threshold"])
    applied_threshold = float(st.session_state["applied_threshold"])
    selected_metrics = _metrics_at_threshold(y_val, probabilities, selected_threshold)
    recommended_metrics = _metrics_at_threshold(y_val, probabilities, DEFAULT_THRESHOLD)

    st.markdown("### ⭐ 권장 운영 기준")
    st.markdown(
        "최종 모델 검증 결과로 선정한 **균형 관리 38%**를 기본으로 보여드립니다.  \n"
        "아래 결과를 확인한 뒤, 별도 운영 제약이 있을 때만 화면 하단에서 기준을 조정하세요."
    )
    st.info(
        "이 화면의 수치는 과거 Validation 고객 데이터를 기준으로 계산한 예상 성능입니다. "
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
                이탈 고객을 80% 이상 발견하면서 캠페인 대상 규모와 적중률의 균형을 고려한 기준입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 화면 상단 결과는 첫 진입 시 균형 관리 38% 기준으로 나타납니다.
    st.markdown(f"#### 현재 선택된 {selected_threshold:.0%} 기준 예상 결과")
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
    estimated_hits = round(selected_metrics["precision"] * 3)
    precision_description = (
        f"선정 고객 3명 중 약 {estimated_hits}명이 실제 이탈 고객"
        if selected_metrics["predicted_risk"] > 0
        else "선정된 위험 고객이 없습니다"
    )
    _metric_card(
        second_row[0], "선정 고객 적중률", f"{selected_metrics['precision']:.1%}", precision_description,
    )
    _metric_card(
        second_row[1], "추가로 관리하는 유지 고객 수", f"{selected_metrics['fp']:,}명",
        "실제로는 유지하지만 캠페인 대상에 포함되는 고객",
    )
    _metric_card(
        second_row[2], "발견·적중 균형 점수", f"{selected_metrics['f1']:.3f}",
        "발견률과 적중률을 함께 고려한 비교 점수",
    )

    st.markdown("#### 기준 변화에 따른 예상 결과")
    sweep_df = _build_sweep(y_val, probabilities)
    _render_threshold_chart(
        sweep_df,
        selected_threshold=selected_threshold,
        applied_threshold=applied_threshold,
        selected_metrics=selected_metrics,
    )
    st.markdown(
        f"현재 선택 기준에서는 관리 대상 고객 **{selected_metrics['predicted_risk']:,}명**, "
        f"놓칠 수 있는 이탈 고객 **{selected_metrics['fn']:,}명**, "
        f"추가로 관리하는 유지 고객 **{selected_metrics['fp']:,}명**으로 예상됩니다."
    )

    st.markdown("#### 고객 선정 결과")
    _render_selection_bars(selected_metrics)

    with st.expander("기준을 어떻게 정하면 좋을까요?"):
        st.markdown(
            f"- **적극 관리(30%)**: 더 많은 이탈 고객을 발견하지만 캠페인 대상과 비용이 증가합니다.\n"
            f"- **균형 관리({DEFAULT_THRESHOLD:.0%}, 권장)**: 발견률과 선정 고객 적중률의 균형을 고려합니다.\n"
            "- **선별 관리(65%)**: 이탈 가능성이 높은 고객에게 집중해 캠페인 비용을 줄입니다."
        )

    with st.expander("지표 용어는 무엇인가요?"):
        st.markdown(
            f"- **Threshold(캠페인 대상 선정 기준)**: `{selected_threshold:.2f}`\n"
            f"- **Recall(실제 이탈 고객 발견률)**: `{selected_metrics['recall']:.3f}`\n"
            f"- **Precision(선정 고객 적중률)**: `{selected_metrics['precision']:.3f}`\n"
            f"- **F1-score(발견·적중 균형 점수)**: `{selected_metrics['f1']:.3f}`\n"
            f"- **TP / FN / FP**: `{selected_metrics['tp']}`명 / "
            f"`{selected_metrics['fn']}`명 / `{selected_metrics['fp']}`명"
        )

    # 기존 캠페인 대상 선정 화면에서 이어지던 실제 적용 결과 영역을 같은 위치에 표시합니다.
    risk_segments.render_summary(model, preprocessor)

    # 기존 탭에서는 위에 있던 제어 영역을 이 탭에서는 결과 아래로 배치합니다.
    st.divider()
    st.markdown("#### 운영 전략 선택 · 필요할 때만 직접 조정")
    st.caption(
        "기본 권장값은 38%입니다. 운영 인력·예산 제약이 있을 때만 아래에서 다른 기준을 선택하세요."
    )

    preset_columns = st.columns(3)
    for column, (name, preset) in zip(preset_columns, PRESETS.items()):
        with column:
            is_selected = abs(selected_threshold - preset["threshold"]) < 1e-9
            st.button(
                f"{name} · {preset['threshold']:.0%}",
                key=f"recommended_preset_{preset['threshold']:.2f}",
                type="primary" if is_selected else "secondary",
                on_click=_select_recommended_preset,
                args=(preset["threshold"],),
                width="stretch",
            )
            st.caption(preset["description"])

    st.select_slider(
        "이탈 가능성이 몇 % 이상인 고객을 관리 대상으로 선정할까요?",
        options=[round(value, 2) for value in np.arange(0.10, 0.91, 0.01)],
        key="recommended_draft_threshold",
        format_func=lambda value: f"{value:.0%}",
        on_change=_mark_recommended_pending,
        help="기준을 바꾸면 위 결과와 그래프가 즉시 미리보기로 갱신됩니다. 적용 버튼을 눌러야 전체 대시보드에 반영됩니다.",
    )

    status_left, status_right = st.columns(2)
    status_left.metric("현재 선택 기준", f"{selected_threshold:.0%}")
    status_right.metric("전체 대시보드 적용 기준", f"{applied_threshold:.0%}")

    if abs(selected_threshold - applied_threshold) > 1e-9:
        st.warning("현재 선택 기준은 아직 전체 대시보드에 적용되지 않았습니다.")
    elif st.session_state["recommended_threshold_apply_success"]:
        st.success("선택한 기준이 위험 고객 목록, 개별 예측, ROI 화면에 적용되었습니다.")
    else:
        st.caption("현재 선택 기준과 전체 대시보드 적용 기준이 같습니다.")

    st.button(
        "기준 적용하기",
        type="primary",
        on_click=_apply_recommended_threshold,
        width="stretch",
    )
