"""최종 운영 기준 38% 캠페인 대상을 위한 ROI 시뮬레이터.

[Doo 작업]
- 고객 선정 기준은 38%로 고정하고 비용·성공률·이익률 시나리오를 비교
- 보수적·기준·낙관적 시나리오와 사용자 설정 상태 추가
- 매출총이익률을 반영한 유지 이익·순이익·ROI 및 손익분기 지표 추가
- K%별/Threshold별 순이익, 시나리오 비교, 성공률 민감도 분석 구현
- 핵심 KPI 중심으로 화면을 압축하고 보조 분석은 접힌 영역으로 재배치
- 캠페인 예산 한도와 K% 방식의 예산 내 자동 조정 기능 추가
- 프리셋 자동 판별, 중복 시나리오 제거와 최대 순이익 지점 해석 강화

기존 XGBoost 모델, 현재 고객 스냅샷, 고객별 이탈 확률과 적용 Threshold는 변경하지 않는다.
"""

from __future__ import annotations

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from customer_scoring import load_customer_table, score_customers
from config import DEFAULT_THRESHOLD


SCENARIOS = {
    "보수적": {"cost": 8.0, "success_rate": 0.10, "margin_rate": 0.35},
    "기준": {"cost": 5.0, "success_rate": 0.15, "margin_rate": 0.45},
    "낙관적": {"cost": 3.0, "success_rate": 0.20, "margin_rate": 0.55},
}

SCENARIO_DESCRIPTIONS = {
    "보수적": "비용은 높고 캠페인 효과와 이익률은 낮게 가정합니다.",
    "기준": "현재 의사결정을 위한 중간 수준의 예시 가정입니다.",
    "낙관적": "비용은 낮고 캠페인 효과와 이익률은 높게 가정합니다.",
}

PERIOD_OPTIONS = ["1개월", "3개월", "6개월", "12개월"]
PERIOD_DAYS = {"1개월": 30, "3개월": 90, "6개월": 180, "12개월": 365}


def _project_revenue_per_customer(targeted: pd.DataFrame, analysis_period: str) -> float:
    """과거 고객별 일평균 순매출을 선택 기간으로 환산한 평균 유지 매출 참고값."""
    if targeted.empty:
        return 0.0
    # 첫 구매 직후 고객의 하루 매출이 과도하게 연환산되지 않도록 최소 30일의 관찰 노출을 둔다.
    exposure_days = (targeted["tenure_days"].astype(float) + 1.0).clip(lower=30.0)
    daily_net_revenue = targeted["net_revenue"].clip(lower=0.0) / exposure_days
    return float((daily_net_revenue * PERIOD_DAYS[analysis_period]).mean())


def _apply_scenario(name: str, reference_revenue: float) -> None:
    """[Doo 작업] 선택한 프리셋 값을 입력 위젯의 세션 상태에 반영한다."""
    values = SCENARIOS[name]
    st.session_state["roi_cost"] = values["cost"]
    st.session_state["roi_success"] = values["success_rate"]
    st.session_state["roi_margin"] = values["margin_rate"]
    st.session_state["roi_retained_revenue"] = float(max(reference_revenue, 0.0))
    st.session_state["roi_active_scenario"] = name


def _mark_custom() -> None:
    """[Doo 작업] 사용자가 프리셋 값을 직접 바꾸면 사용자 설정으로 표시한다."""
    st.session_state["roi_active_scenario"] = "사용자 설정"


def _activate_custom() -> None:
    st.session_state["roi_active_scenario"] = "사용자 설정"


def _use_reference_revenue(value: float) -> None:
    """현재 타겟의 과거 평균 순매출을 유지 매출 입력의 참고값으로 복원한다."""
    st.session_state["roi_retained_revenue"] = float(max(value, 0.0))
    st.session_state["roi_active_scenario"] = "사용자 설정"


def _set_budget_k(k_percent: int) -> None:
    """[Doo 작업] 예산 안에서 가능한 K%를 선택하되 공통 Threshold는 변경하지 않는다."""
    st.session_state["roi_k"] = int(k_percent)


def _scenario_name(
    campaign_cost: float,
    success_rate: float,
    margin_rate: float,
    retained_revenue: float,
    reference_revenue: float,
    *,
    tolerance: float = 1e-6,
) -> str:
    """[Doo 작업] 현재 값이 프리셋과 같으면 프리셋 이름으로 자동 복원한다."""
    if not math.isclose(retained_revenue, reference_revenue, rel_tol=tolerance, abs_tol=tolerance):
        return "사용자 설정"
    for name, values in SCENARIOS.items():
        if (
            math.isclose(campaign_cost, values["cost"], rel_tol=tolerance, abs_tol=tolerance)
            and math.isclose(success_rate, values["success_rate"], rel_tol=tolerance, abs_tol=tolerance)
            and math.isclose(margin_rate, values["margin_rate"], rel_tol=tolerance, abs_tol=tolerance)
        ):
            return name
    return "사용자 설정"


def _initialize_state(reference_revenue: float) -> None:
    defaults = {
        "roi_cost": SCENARIOS["기준"]["cost"],
        "roi_success": SCENARIOS["기준"]["success_rate"],
        "roi_margin": SCENARIOS["기준"]["margin_rate"],
        "roi_retained_revenue": float(max(reference_revenue, 0.0)),
        "roi_period": "3개월",
        "roi_k": 20,
        "roi_budget_enabled": False,
        "roi_total_budget": 5000.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    previous_reference = st.session_state.get("roi_reference_revenue")
    if previous_reference is not None and math.isclose(
        float(st.session_state["roi_retained_revenue"]), float(previous_reference),
        rel_tol=1e-6, abs_tol=1e-6,
    ):
        # 사용자가 참고값을 그대로 쓰는 동안에는 분석 기간·타겟 변경을 입력값에 반영한다.
        st.session_state["roi_retained_revenue"] = float(max(reference_revenue, 0.0))
    st.session_state["roi_reference_revenue"] = float(max(reference_revenue, 0.0))
    if "roi_active_scenario" not in st.session_state:
        current = {
            "cost": float(st.session_state["roi_cost"]),
            "success_rate": float(st.session_state["roi_success"]),
            "margin_rate": float(st.session_state["roi_margin"]),
        }
        matched = next(
            (
                name
                for name, values in SCENARIOS.items()
                if all(abs(current[key] - values[key]) < 1e-9 for key in current)
            ),
            "사용자 설정",
        )
        st.session_state["roi_active_scenario"] = matched


def _select_target(
    scored_customers: pd.DataFrame,
    target_mode: str,
    k_percent: int,
    threshold: float,
) -> pd.DataFrame:
    """[Doo 작업] 선택 방식에 맞는 고객만 반환한다."""
    if target_mode == "이탈 확률 상위 K%":
        count = max(1, int(len(scored_customers) * k_percent / 100))
        return scored_customers.iloc[:count].copy()
    return scored_customers[scored_customers["이탈확률"] >= threshold].copy()


def _calculate_roi(
    targeted: pd.DataFrame,
    campaign_cost: float,
    success_rate: float,
    margin_rate: float,
    retained_revenue_per_customer: float,
) -> dict:
    """[Doo 작업] 매출총이익률과 손익분기를 포함한 ROI 지표를 계산한다."""
    target_count = len(targeted)
    expected_churners = float(targeted["이탈확률"].sum()) if target_count else 0.0
    expected_saved = expected_churners * success_rate
    retained_revenue = expected_saved * retained_revenue_per_customer
    retained_profit = retained_revenue * margin_rate
    total_cost = target_count * campaign_cost
    net_profit = retained_profit - total_cost
    roi_pct = (net_profit / total_cost * 100) if total_cost > 0 else None

    break_even_denominator = expected_churners * retained_revenue_per_customer * margin_rate
    if break_even_denominator > 0:
        break_even_success_rate = total_cost / break_even_denominator
    else:
        break_even_success_rate = math.inf

    if target_count > 0:
        break_even_cost_per_customer = retained_profit / target_count
    else:
        break_even_cost_per_customer = 0.0

    # 기존 구현의 평균 순매출 기반 결과는 비교용 전문 지표로만 유지한다.
    historical_average_revenue = float(targeted["net_revenue"].mean()) if target_count else 0.0
    legacy_retained_revenue = expected_saved * historical_average_revenue
    legacy_net_benefit = legacy_retained_revenue - total_cost

    return {
        "target_count": target_count,
        "expected_churners": expected_churners,
        "expected_saved": expected_saved,
        "retained_revenue_per_customer": retained_revenue_per_customer,
        "retained_revenue": retained_revenue,
        "retained_profit": retained_profit,
        "total_cost": total_cost,
        "net_profit": net_profit,
        "roi_pct": roi_pct,
        "break_even_success_rate": break_even_success_rate,
        "break_even_cost_per_customer": break_even_cost_per_customer,
        "historical_average_revenue": historical_average_revenue,
        "legacy_retained_revenue": legacy_retained_revenue,
        "legacy_net_benefit": legacy_net_benefit,
    }


def _metric_card(column, label: str, value: str, description: str, *, negative: bool = False) -> None:
    """[Doo 작업] CRM KPI를 동일한 카드 형식으로 표시한다."""
    with column:
        with st.container(border=True):
            st.caption(label)
            color = "#C62828" if negative else "#1A1A2E"
            st.markdown(
                f'<div style="font-size:26px;font-weight:700;color:{color};margin:2px 0 8px 0;min-height:38px;">'
                f"{value}</div>",
                unsafe_allow_html=True,
            )
            st.caption(description)


def _format_roi(value: float | None) -> str:
    return "계산 불가" if value is None else f"{value:,.1f}%"


def _build_k_sweep(
    scored_customers: pd.DataFrame,
    campaign_cost: float,
    success_rate: float,
    margin_rate: float,
    retained_revenue: float,
) -> pd.DataFrame:
    rows = []
    for k in range(5, 105, 5):
        group = _select_target(scored_customers, "이탈 확률 상위 K%", k, 0.0)
        metrics = _calculate_roi(group, campaign_cost, success_rate, margin_rate, retained_revenue)
        rows.append({
            "상위 K%": k,
            "타겟 고객 수": metrics["target_count"],
            "예상 이탈 고객": metrics["expected_churners"],
            "예상 방지 고객": metrics["expected_saved"],
            "예상 유지 이익": metrics["retained_profit"],
            "캠페인 총비용": metrics["total_cost"],
            "예상 순이익": metrics["net_profit"],
            "ROI": metrics["roi_pct"],
        })
    return pd.DataFrame(rows)


def _build_threshold_sweep(
    scored_customers: pd.DataFrame,
    applied_threshold: float,
    campaign_cost: float,
    success_rate: float,
    margin_rate: float,
    retained_revenue: float,
) -> pd.DataFrame:
    thresholds = sorted(set(np.round(np.arange(0.10, 0.91, 0.05), 2).tolist() + [round(applied_threshold, 2)]))
    rows = []
    for threshold in thresholds:
        group = _select_target(scored_customers, "현재 캠페인 선정 기준 이상", 0, threshold)
        metrics = _calculate_roi(group, campaign_cost, success_rate, margin_rate, retained_revenue)
        rows.append({
            "Threshold": threshold,
            "Threshold 표시": f"{threshold:.0%}",
            "타겟 고객 수": metrics["target_count"],
            "예상 이탈 고객": metrics["expected_churners"],
            "예상 방지 고객": metrics["expected_saved"],
            "예상 유지 이익": metrics["retained_profit"],
            "캠페인 총비용": metrics["total_cost"],
            "예상 순이익": metrics["net_profit"],
            "ROI": metrics["roi_pct"],
        })
    return pd.DataFrame(rows)


def _profit_chart(
    sweep_df: pd.DataFrame,
    x_field: str,
    current_value: float,
    x_title: str,
) -> tuple[alt.Chart, pd.Series]:
    """현재 선택·최대 순이익·손익분기선을 표시한 공통 차트."""
    best_row = sweep_df.loc[sweep_df["예상 순이익"].idxmax()]
    tooltip = [
        alt.Tooltip(f"{x_field}:Q", title=x_title, format=".0%" if x_field == "Threshold" else ".0f"),
        alt.Tooltip("타겟 고객 수:Q", format=",.0f"),
        alt.Tooltip("예상 이탈 고객:Q", format=",.1f"),
        alt.Tooltip("예상 방지 고객:Q", format=",.1f"),
        alt.Tooltip("예상 유지 이익:Q", title="예상 유지 이익 (GBP)", format=",.0f"),
        alt.Tooltip("캠페인 총비용:Q", title="캠페인 총비용 (GBP)", format=",.0f"),
        alt.Tooltip("예상 순이익:Q", title="예상 순이익 (GBP)", format=",.0f"),
        alt.Tooltip("ROI:Q", format=",.1f"),
    ]
    x_encoding = alt.X(
        f"{x_field}:Q",
        title=x_title,
        axis=alt.Axis(format="%" if x_field == "Threshold" else "d"),
    )
    base = alt.Chart(sweep_df).encode(x=x_encoding)
    line = base.mark_line(color="#2F80ED", point=True).encode(
        y=alt.Y("예상 순이익:Q", title="예상 순이익 (GBP)"),
        tooltip=tooltip,
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="#8A8F98", strokeDash=[5, 4], size=1,
    ).encode(y="y:Q")
    current = alt.Chart(pd.DataFrame({x_field: [current_value]})).mark_rule(
        color="#2F80ED", strokeDash=[5, 4], size=2,
    ).encode(x=f"{x_field}:Q")
    current_point = alt.Chart(
        sweep_df.iloc[(sweep_df[x_field] - current_value).abs().argsort()[:1]]
    ).mark_point(color="#2F80ED", shape="circle", filled=True, size=130).encode(
        x=f"{x_field}:Q", y="예상 순이익:Q", tooltip=tooltip,
    )
    best_point = alt.Chart(pd.DataFrame([best_row])).mark_point(
        color="#E56A00", shape="diamond", filled=True, size=170,
    ).encode(x=f"{x_field}:Q", y="예상 순이익:Q", tooltip=tooltip)
    return (line + zero + current + current_point + best_point).properties(height=330), best_row


def _scenario_comparison(
    targeted: pd.DataFrame,
    reference_revenue: float,
    current_values: dict,
    active_scenario: str,
) -> pd.DataFrame:
    rows = []
    scenarios = {
        name: {**values, "retained_revenue": reference_revenue}
        for name, values in SCENARIOS.items()
    }
    if active_scenario == "사용자 설정":
        scenarios["사용자 설정"] = current_values
    for name, values in scenarios.items():
        metrics = _calculate_roi(
            targeted,
            values["cost"],
            values["success_rate"],
            values["margin_rate"],
            values["retained_revenue"],
        )
        rows.append({
            "현재": "●" if name == active_scenario else "",
            "시나리오": name,
            "가정": (
                f"비용 £{values['cost']:,.0f} · 성공률 {values['success_rate']:.0%} · "
                f"이익률 {values['margin_rate']:.0%} · 유지매출 £{values['retained_revenue']:,.0f}"
            ),
            "예상 방지 고객": metrics["expected_saved"],
            "예상 유지 이익": metrics["retained_profit"],
            "캠페인 총비용": metrics["total_cost"],
            "예상 순이익": metrics["net_profit"],
            "ROI": metrics["roi_pct"],
        })
    return pd.DataFrame(rows)


def _render_sensitivity(
    targeted: pd.DataFrame,
    campaign_cost: float,
    current_success_rate: float,
    margin_rate: float,
    retained_revenue: float,
) -> None:
    rows = []
    for success_rate in np.arange(0.0, 1.01, 0.05):
        metrics = _calculate_roi(targeted, campaign_cost, success_rate, margin_rate, retained_revenue)
        rows.append({"성공률": success_rate, "예상 순이익": metrics["net_profit"]})
    sensitivity_df = pd.DataFrame(rows)
    line = alt.Chart(sensitivity_df).mark_line(color="#2F80ED", point=True).encode(
        x=alt.X("성공률:Q", axis=alt.Axis(format="%")),
        y=alt.Y("예상 순이익:Q", title="예상 순이익 (GBP)"),
        tooltip=[
            alt.Tooltip("성공률:Q", format=".0%"),
            alt.Tooltip("예상 순이익:Q", title="예상 순이익 (GBP)", format=",.0f"),
        ],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#8A8F98").encode(y="y:Q")
    current = alt.Chart(pd.DataFrame({"성공률": [current_success_rate]})).mark_rule(
        color="#E56A00", strokeDash=[5, 4], size=2,
    ).encode(x="성공률:Q")
    st.altair_chart((line + zero + current).properties(height=300), width="stretch")


def render(model, preprocessor):
    """[Doo 작업] 핵심 의사결정이 먼저 보이는 압축형 ROI 화면을 렌더링한다."""
    st.markdown("### 💰 캠페인 ROI 시뮬레이터")
    st.markdown("이탈 위험 고객에게 캠페인을 진행할 때 예상 비용, 유지 효과, 순이익과 ROI를 비교합니다.")
    st.caption(
        "모델 이탈 확률과 입력 가정에 따른 예상치이며 실제 성과를 보장하지 않습니다. "
        "금액은 영국 이커머스 원본 데이터에 맞춰 GBP로 표시합니다."
    )

    scored_customers = score_customers(load_customer_table(), model, preprocessor)
    scored_customers = scored_customers.sort_values("이탈확률", ascending=False).reset_index(drop=True)
    applied_threshold = DEFAULT_THRESHOLD

    st.markdown("#### 1. 캠페인 대상")
    target_mode = "현재 캠페인 선정 기준 이상"
    k_percent = 0
    preliminary_target = _select_target(scored_customers, target_mode, k_percent, applied_threshold)
    threshold_columns = st.columns(2)
    threshold_columns[0].metric("최종 운영 기준", f"{applied_threshold:.0%}")
    threshold_columns[1].metric("캠페인 대상 고객", f"{len(preliminary_target):,}명")
    st.caption("ROI는 다른 Threshold나 상위 K%를 적용하지 않고 최종 운영 기준 38% 이상 고객으로 계산합니다.")

    if preliminary_target.empty:
        st.warning("최종 운영 기준에 해당하는 고객이 없어 ROI를 계산할 수 없습니다.")
        return

    selected_period = st.session_state.get("roi_period", "3개월")
    reference_revenue = _project_revenue_per_customer(preliminary_target, selected_period)
    _initialize_state(reference_revenue)
    st.session_state["roi_active_scenario"] = _scenario_name(
        float(st.session_state["roi_cost"]),
        float(st.session_state["roi_success"]),
        float(st.session_state["roi_margin"]),
        float(st.session_state["roi_retained_revenue"]),
        reference_revenue,
    )

    st.markdown("#### 2. 시나리오 프리셋")
    preset_columns = st.columns(3)
    for column, name in zip(preset_columns, SCENARIOS):
        values = SCENARIOS[name]
        with column:
            st.button(
                f"{name} 시나리오",
                key=f"roi_preset_{name}",
                type="primary" if st.session_state["roi_active_scenario"] == name else "secondary",
                on_click=_apply_scenario,
                args=(name, reference_revenue),
                width="stretch",
            )
            st.caption(
                f"비용 £{values['cost']:.0f} · 성공률 {values['success_rate']:.0%} · "
                f"이익률 {values['margin_rate']:.0%} — {SCENARIO_DESCRIPTIONS[name]}"
            )

    st.markdown("#### 3. 주요 가정값 설정")
    assumptions_column, operations_column = st.columns([3, 2], gap="large")
    with assumptions_column:
        with st.container(border=True):
            first_inputs = st.columns(4)
            with first_inputs[0]:
                campaign_cost = st.number_input(
                    "1인당 캠페인 비용 (GBP)", min_value=0.0, step=0.5, key="roi_cost",
                    on_change=_mark_custom,
                    help="쿠폰, 포인트, 발송과 운영비를 포함한 고객 1명당 예상 비용입니다.",
                )
            with first_inputs[1]:
                success_rate = st.select_slider(
                    "이탈 방지 성공률",
                    options=[round(value, 2) for value in np.arange(0.0, 1.01, 0.01)],
                    key="roi_success", format_func=lambda value: f"{value:.0%}",
                    on_change=_mark_custom,
                    help="캠페인을 받은 예상 이탈 고객 중 이탈을 막을 수 있다고 가정하는 비율입니다.",
                )
            with first_inputs[2]:
                margin_rate = st.select_slider(
                    "매출총이익률",
                    options=[round(value, 2) for value in np.arange(0.0, 1.01, 0.01)],
                    key="roi_margin", format_func=lambda value: f"{value:.0%}",
                    on_change=_mark_custom,
                    help="유지 매출 중 원가 등을 제외하고 이익으로 남는 비율입니다.",
                )
            with first_inputs[3]:
                retained_revenue_per_customer = st.number_input(
                    "고객 1인당 예상 유지 매출 (GBP)", min_value=0.0, step=10.0,
                    key="roi_retained_revenue", on_change=_mark_custom,
                    help="이탈을 방지한 고객이 분석 기간에 발생시킬 것으로 가정한 매출입니다.",
                )

            active_scenario = _scenario_name(
                campaign_cost, success_rate, margin_rate,
                retained_revenue_per_customer, reference_revenue,
            )
            st.session_state["roi_active_scenario"] = active_scenario

            second_inputs = st.columns(4)
            with second_inputs[0]:
                analysis_period = st.selectbox(
                    "분석 기간", PERIOD_OPTIONS, key="roi_period",
                    help="고객 1인당 예상 유지 매출을 계산하는 기준 기간입니다.",
                )
            with second_inputs[1]:
                st.markdown("**유지 매출 참고값**")
                st.button(
                    "현재 대상 평균값 사용", on_click=_use_reference_revenue,
                    args=(reference_revenue,), width="stretch",
                )
            with second_inputs[2]:
                st.markdown("**현재 시나리오**")
                st.markdown(f"`{active_scenario}`")
                st.caption("입력값과 프리셋을 자동 비교")
            with second_inputs[3]:
                st.markdown("**현재 타겟 고객**")
                st.markdown(f"**{len(preliminary_target):,}명**")
                st.caption(f"과거 일평균 순매출을 {analysis_period}로 환산: £{reference_revenue:,.0f}")

    with operations_column:
        with st.container(border=True):
            st.markdown("**현재 시나리오 및 예산 상태**")
            st.markdown(f"현재 시나리오: **{active_scenario}**")
            budget_enabled = st.toggle("예산 한도 사용", key="roi_budget_enabled")
            if budget_enabled:
                total_budget = st.number_input(
                    "총 캠페인 예산 (GBP)", min_value=0.0, step=100.0,
                    key="roi_total_budget",
                    help="기존 ROI 계산을 바꾸지 않고 실제 발송 가능 범위를 점검하는 운영 제약입니다.",
                )
            else:
                total_budget = float(st.session_state.get("roi_total_budget", 5000.0))
                st.caption("예산 한도를 사용하지 않아 기존 계산 방식이 그대로 적용됩니다.")
            st.caption(f"입력 유지 매출: £{retained_revenue_per_customer:,.0f} · {analysis_period}")

    if campaign_cost < 0 or not 0 <= success_rate <= 1 or not 0 <= margin_rate <= 1 or retained_revenue_per_customer < 0:
        st.error("비용·유지 매출은 0 이상, 성공률·이익률은 0%에서 100% 사이여야 합니다.")
        return

    targeted = _select_target(scored_customers, target_mode, k_percent, applied_threshold)
    metrics = _calculate_roi(
        targeted, campaign_cost, success_rate, margin_rate, retained_revenue_per_customer,
    )

    budget_exceeded = False
    max_sendable = len(scored_customers)
    if budget_enabled:
        max_sendable = (
            math.floor(total_budget / campaign_cost)
            if campaign_cost > 0 else len(scored_customers)
        )
        max_sendable = min(max_sendable, len(scored_customers))
        budget_difference = total_budget - metrics["total_cost"]
        budget_exceeded = metrics["target_count"] > max_sendable
        budget_columns = st.columns(4)
        budget_columns[0].metric("최대 발송 가능", f"{max_sendable:,}명")
        budget_columns[1].metric("현재 타겟", f"{metrics['target_count']:,}명")
        budget_columns[2].metric("예산 상태", "초과" if budget_exceeded else "예산 내")
        budget_columns[3].metric(
            "초과 금액" if budget_exceeded else "남은 예산",
            f"£{abs(budget_difference):,.0f}",
        )
        if budget_exceeded:
            st.warning(
                "현재 선택한 타겟 고객 수가 캠페인 예산을 초과합니다. "
                f"예산 내에서 발송 가능한 최대 고객은 {max_sendable:,}명입니다."
            )
            st.caption("예산이 부족해도 최종 운영 기준 38%는 자동으로 변경하지 않습니다.")

    st.markdown("#### 4. 핵심 KPI")
    first_kpis = st.columns(3)
    _metric_card(first_kpis[0], "타겟 고객 수", f"{metrics['target_count']:,}명", "캠페인 발송 대상")
    _metric_card(
        first_kpis[1], "예상 방지 이탈 고객", f"약 {round(metrics['expected_saved']):,}명",
        f"계산 원값 {metrics['expected_saved']:.1f}명",
    )
    _metric_card(first_kpis[2], "예상 유지 이익", f"£{metrics['retained_profit']:,.0f}", "유지 매출 × 이익률")
    second_kpis = st.columns(3)
    _metric_card(second_kpis[0], "캠페인 총비용", f"£{metrics['total_cost']:,.0f}", "타겟 수 × 1인당 비용")
    _metric_card(
        second_kpis[1], "예상 순이익", f"£{metrics['net_profit']:,.0f}",
        "유지 이익 - 캠페인 비용", negative=metrics["net_profit"] < 0,
    )
    _metric_card(second_kpis[2], "예상 ROI", _format_roi(metrics["roi_pct"]), "가정 기반 예상치")

    break_even_rate = metrics["break_even_success_rate"]
    if math.isfinite(break_even_rate):
        break_even_gap = success_rate - break_even_rate
        break_even_message = (
            f"현재 손익분기 성공률 **{break_even_rate:.1%}** · 입력 성공률 **{success_rate:.1%}** · "
            f"손익분기점 대비 **{break_even_gap * 100:+.1f}%p** · "
            f"감당 가능한 최대 1인당 비용 **£{metrics['break_even_cost_per_customer']:,.2f}**"
        )
        if break_even_gap < 0:
            st.warning(f"**손익분기 기준**  \n{break_even_message}")
        else:
            st.info(f"**손익분기 기준**  \n{break_even_message}")
    else:
        st.warning("**손익분기 기준**  \n고객 가치 또는 이익률이 0이어서 손익분기 성공률을 계산할 수 없습니다.")

    with st.container(border=True):
        support_columns = st.columns(5)
        support_columns[0].caption("예상 이탈 고객")
        support_columns[0].markdown(f"**약 {round(metrics['expected_churners']):,}명**")
        support_columns[1].caption("1인당 예상 유지 매출")
        support_columns[1].markdown(f"**£{retained_revenue_per_customer:,.0f}**")
        support_columns[2].caption("예상 유지 매출")
        support_columns[2].markdown(f"**£{metrics['retained_revenue']:,.0f}**")
        support_columns[3].caption("손익분기 성공률")
        support_columns[3].markdown(
            "**계산 불가**" if not math.isfinite(break_even_rate) else f"**{break_even_rate:.1%}**"
        )
        support_columns[4].caption("최대 1인당 캠페인 비용")
        support_columns[4].markdown(f"**£{metrics['break_even_cost_per_customer']:,.2f}**")

    st.markdown("#### 5. 현재 결과 해석")
    plan_column, profit_column = st.columns(2)
    with plan_column:
        st.markdown("**현재 캠페인 계획**")
        st.markdown(
            f"타겟 고객 **{metrics['target_count']:,}명**  \n"
            f"예상 이탈 고객 **약 {round(metrics['expected_churners']):,}명**  \n"
            f"예상 방지 이탈 고객 **약 {round(metrics['expected_saved']):,}명**"
        )
    with profit_column:
        st.markdown("**예상 수익성**")
        st.markdown(
            f"예상 유지 이익 **£{metrics['retained_profit']:,.0f}**  \n"
            f"캠페인 총비용 **£{metrics['total_cost']:,.0f}**  \n"
            f"예상 순이익 **£{metrics['net_profit']:,.0f}** · ROI **{_format_roi(metrics['roi_pct'])}**"
        )
    if metrics["net_profit"] >= 0:
        st.info("현재 설정에서는 캠페인 비용보다 예상 유지 이익이 높아 수익이 발생하는 것으로 계산됩니다.")
    else:
        st.error("현재 설정에서는 캠페인 비용이 예상 유지 이익보다 높아 손실이 발생하는 것으로 계산됩니다.")
    if metrics["roi_pct"] is not None and metrics["roi_pct"] >= 500:
        st.warning("높은 고객 가치 또는 성공률 가정의 영향을 크게 받고 있습니다. 실제 캠페인 자료로 가정을 점검해주세요.")

    with st.expander("시나리오별 결과 비교"):
        st.markdown("동일한 타겟 고객을 기준으로 비용, 성공률과 이익률 가정에 따른 결과를 비교합니다.")
        current_values = {
            "cost": campaign_cost,
            "success_rate": success_rate,
            "margin_rate": margin_rate,
            "retained_revenue": retained_revenue_per_customer,
        }
        comparison_df = _scenario_comparison(
            targeted, reference_revenue, current_values, active_scenario,
        )
        display_comparison = comparison_df.copy()
        display_comparison["시나리오"] = (
            display_comparison["시나리오"] + " · " + display_comparison["가정"]
        )
        display_comparison = display_comparison.drop(columns="가정")
        display_comparison["예상 방지 고객"] = display_comparison["예상 방지 고객"].map(
            lambda value: f"약 {round(value):,}명"
        )
        for column in ["예상 유지 이익", "캠페인 총비용", "예상 순이익"]:
            display_comparison[column] = display_comparison[column].map(lambda value: f"£{value:,.0f}")
        display_comparison["ROI"] = display_comparison["ROI"].map(_format_roi)
        st.dataframe(
            display_comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "현재": st.column_config.TextColumn(width="small"),
                "시나리오": st.column_config.TextColumn(width="large"),
            },
        )
        st.caption("● 표시는 현재 선택된 시나리오입니다. 프리셋과 같은 사용자 설정 행은 자동으로 숨깁니다.")

    with st.expander("가정 민감도 분석"):
        st.markdown("다른 조건을 고정하고 이탈 방지 성공률만 변경했을 때의 예상 순이익입니다.")
        _render_sensitivity(
            targeted, campaign_cost, success_rate, margin_rate, retained_revenue_per_customer,
        )

    with st.expander("계산식과 가정 자세히 보기"):
        st.markdown(
            """
            - **예상 이탈 고객 수** = 캠페인 대상 고객들의 이탈 확률 합계
            - **예상 방지 이탈 고객 수** = 예상 이탈 고객 수 × 이탈 방지 성공률
            - **예상 유지 매출** = 예상 방지 이탈 고객 수 × 고객 1인당 예상 유지 매출
            - **예상 유지 이익** = 예상 유지 매출 × 매출총이익률
            - **캠페인 총비용** = 캠페인 대상 고객 수 × 1인당 캠페인 비용
            - **예상 순이익** = 예상 유지 이익 - 캠페인 총비용
            - **예상 ROI** = 예상 순이익 ÷ 캠페인 총비용 × 100
            - **손익분기 성공률** = 총비용 ÷ (예상 이탈 고객 수 × 1인당 유지 매출 × 이익률)
            - **최대 발송 가능 고객 수** = 총 캠페인 예산 ÷ 1인당 캠페인 비용(소수점 버림)

            예산 한도는 기존 ROI 계산을 대체하지 않고 운영 가능 여부만 점검합니다.
            Threshold 방식에서는 예산 초과 경고만 제공하고 실제 적용 Threshold를 변경하지 않습니다.
            """
        )
        st.markdown(
            """
            **기본 시나리오의 외부 참고 근거**

            - 캠페인 비용 £3–£8은 쿠폰·발송·운영비를 포함하는 민감도 범위입니다. 영국 Royal Mail의
              온라인 소형 소포 요금이 £3.95부터 시작한다는 점을 비용 규모의 참고선으로만 사용했습니다.
            - 성공률 10–20%는 온라인 리테일 현장실험에서 관측된 약 10%p의 이탈 감소와 14.6%의
              사이트 재방문 증가를 중심으로 설정한 탐색 범위이며, 이 프로젝트 기업의 실측 성공률은 아닙니다.
            - 이익률 35–55%는 영국 온라인 카드·선물 기업 Moonpig/Greetz가 공시한 46.1–57.0%의
              매출총이익률을 중심으로 보수적 할인폭을 둔 범위입니다.
            - 유지 매출 참고값은 각 고객의 과거 순매출을 `max(첫 구매 후 경과일+1, 30일)`로 나눈 뒤,
              선택한 30·90·180·365일로 환산한 고객 평균입니다.

            외부 벤치마크는 업종·시기·캠페인 구조가 다르므로 초기 시뮬레이션 범위를 정하는 용도로만 사용하며,
            실제 운영 후에는 A/B 테스트의 증분 이탈률·증분 이익으로 교체해야 합니다.

            자료: [Royal Mail 현재 요금](https://www.royalmail.com/current-postage-prices) ·
            [쿠폰과 이탈·CLV 현장실험](https://doi.org/10.1016/j.jretconser.2026.104798) ·
            [리타겟 광고 현장실험](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2852484) ·
            [Moonpig Group FY2025 공시](https://www.moonpig.group/media/stvd50fh/moonpig-group-plc-annual-report-2025-cfo-review.pdf)
            """
        )
        st.markdown(
            f"**과거 평균 순매출 기반 기존 참고값**  \n"
            f"과거 평균 순매출 £{metrics['historical_average_revenue']:,.0f}을 사용하면 유지 매출은 "
            f"£{metrics['legacy_retained_revenue']:,.0f}, 비용 차감 결과는 £{metrics['legacy_net_benefit']:,.0f}입니다."
        )
        st.warning(
            "결과는 이탈 확률과 입력 가정에 따른 시뮬레이션이며 실제 성과를 보장하지 않습니다. "
            "미래 구매액, 원가, 할인 사용률과 운영비에 따라 달라질 수 있으며 금액은 GBP 기준입니다."
        )
