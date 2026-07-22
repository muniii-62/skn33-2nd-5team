import json

import pandas as pd
import streamlit as st
from config import DEFAULT_THRESHOLD, FEATURE_ORDER, PROJECT_ROOT


def render(model, preprocessor):
    st.markdown("### 🔍 개별 고객 이탈 예측")
    st.caption("고객 정보를 입력하면 이탈 확률과 추천 액션을 실시간으로 확인할 수 있습니다.")

    with st.container(border=True):
        st.markdown("#### 📋 구매 이력")
        col1, col2 = st.columns(2)

        with col1:
            recency_days = st.slider(
                "🕒 최근 구매 후 경과일", 0, 365, 60, key="ind_recency",
                help="마지막 구매일로부터 오늘까지 지난 일수",
            )
            frequency = st.number_input(
                "🔁 구매 횟수", min_value=1, max_value=356, value=5, step=1,
                key="ind_frequency", help="관찰 기간 동안의 고유 주문 횟수",
            )
            distinct_products = st.number_input(
                "📦 구매 상품 종류 수", min_value=1, max_value=2183, value=50, step=1,
                key="ind_distinct",
                help="지금까지 구매한 서로 다른 상품(품목코드)의 가짓수",
            )
            net_revenue = st.number_input(
                "💰 순매출 (£)", 0.0, 482037.0, 1000.0, step=50.0, key="ind_revenue",
                help="관찰 기간 동안 이 고객이 실제로 지불한 금액의 합계 "
                     "(반품·취소분은 제외한 순수 매출). 고객의 구매 규모를 나타내는 지표입니다.",
            )
            st.caption("💡 순매출 = 취소·반품을 제외한 실제 구매 금액 합계")

            if "ind_tenure" in st.session_state:
                st.session_state["ind_tenure"] = min(
                    max(int(st.session_state["ind_tenure"]), int(recency_days)), 647,
                )
            tenure_days = st.number_input(
                "📅 첫 구매 후 경과일", min_value=int(recency_days), max_value=647,
                value=max(200, int(recency_days)), step=1, key="ind_tenure",
                help="첫 구매일부터 기준일까지 지난 고객 활동 기간입니다. 가입 기간이 아닙니다.",
            )

        with col2:
            avg_days_between_orders = float(tenure_days) / float(frequency)
            # [Doo 작업] 계산식과 모델 입력은 유지하고 사용자 노출 명칭과 설명만 정정했다.
            st.metric("↔️ 주문 1회당 활동 기간", f"{avg_days_between_orders:.1f}일")
            st.caption(
                "첫 구매 후 경과일 ÷ 구매 횟수로 계산하는 "
                "구매 간격 대리값입니다. 실제 주문 사이의 평균은 아닙니다."
            )
            has_return = st.selectbox(
                "↩️ 취소 경험 있음?", ["아니오", "예"], key="ind_has_return",
                help="관찰 기간 중 주문 취소/반품 이력이 한 번이라도 있었는지",
            )
            if "ind_activity_count" in st.session_state:
                st.session_state["ind_activity_count"] = min(
                    max(int(st.session_state["ind_activity_count"]), 0),
                    frequency,
                )

            recent_activity_count = st.slider(
                "📈 최근 90일 이내 구매 횟수",
                min_value=0,
                max_value=frequency,
                value=min(1, frequency),
                step=1,
                key="ind_activity_count",
                help="전체 구매 횟수 중 최근 90일 이내에 구매한 횟수를 정수로 선택합니다.",
            )
            recent_activity_ratio = recent_activity_count / frequency
            st.caption(
                f"💡 전체 {frequency}회 중 최근 {recent_activity_count}회 "
                f"→ 최근 구매 활동 비중 {recent_activity_ratio:.0%}"
            )

            average_order_value = float(net_revenue) / float(frequency)
            threshold_path = PROJECT_ROOT / "data" / "preprocessed" / "is_low_value_threshold.json"
            with open(threshold_path, encoding="utf-8") as threshold_file:
                low_value_threshold = float(json.load(threshold_file)["avg_order_value_q20"])
            is_low_value = average_order_value <= low_value_threshold
            st.metric("💸 평균 주문금액", f"£{average_order_value:,.2f}")
            st.caption(
                f"순매출 ÷ 구매 횟수로 자동 계산 · Train 하위 20% 기준 "
                f"£{low_value_threshold:,.2f} {'이하' if is_low_value else '초과'}"
            )

            with st.expander("부가 정보"):
                is_uk = st.selectbox(
                    "🇬🇧 UK 거주?", ["예", "아니오"], key="ind_is_uk",
                    help="영국 거주 고객인지 여부",
                )

        unusual_inputs = []
        if frequency > 54:
            unusual_inputs.append("구매 횟수가 학습 고객의 99% 구간(54회)을 초과합니다.")
        if distinct_products > 496:
            unusual_inputs.append("상품 종류 수가 학습 고객의 99% 구간(약 496개)을 초과합니다.")
        if net_revenue > 28416:
            unusual_inputs.append("순매출이 학습 고객의 99% 구간(약 £28,416)을 초과합니다.")
        if unusual_inputs:
            st.warning("학습 데이터에서 드문 입력 범위입니다. 예측 신뢰도가 낮을 수 있습니다.\n\n- " + "\n- ".join(unusual_inputs))

        if recency_days >= 250:
            st.info(
                "ℹ️ 최근 250일 이상 미구매 고객군은 실제 데이터에서도 이탈률이 70% 안팎에서 "
                "포화되는 경향이 있어, 이 구간에서는 확률의 세부 변동보다 '이미 고위험군'이라는 "
                "판정 자체를 참고하시는 것을 권장합니다."
            )

    input_df = pd.DataFrame([{
        "net_revenue": net_revenue,
        "recency_days": recency_days,
        "frequency": frequency,
        "distinct_products": distinct_products,
        "tenure_days": tenure_days,
        "avg_days_between_orders": avg_days_between_orders,
        "is_low_value": int(is_low_value),
        "is_uk": 1 if is_uk == "예" else 0,
        "has_return": 1 if has_return == "예" else 0,
        "recent_activity_ratio": recent_activity_ratio,
    }])
    input_df = input_df[FEATURE_ORDER]

    # 원본 스케일 입력을 preprocessor로 변환 후 예측 (모델은 스케일링된 데이터로 학습됨)
    input_processed = preprocessor.transform(input_df)
    input_processed_df = pd.DataFrame(input_processed, columns=FEATURE_ORDER)
    # XGBoost는 numpy.float32를 반환하지만 Streamlit의 st.progress는
    # Python 기본 float/int만 허용하므로 여기서 명시적으로 변환합니다.
    churn_proba = float(model.predict_proba(input_processed_df)[0, 1])

    st.write("")
    threshold = DEFAULT_THRESHOLD
    st.info(
        f"최종 운영 기준에 따라 이탈 가능성 **{threshold:.0%} 이상**인 고객을 캠페인 대상으로 선정합니다."
    )
    with st.expander("캠페인 대상 선정 기준이 뭔가요?"):
        st.markdown(
            "모델이 계산한 이탈 가능성을 바탕으로 **몇 % 이상인 고객부터 캠페인 대상으로 "
            "관리할지** 정하는 기준입니다.\n\n"
            "현재 기준 38%는 Validation에서 실제 이탈 고객 발견률 85% 이상을 만족하는 "
            "Threshold 중 F1이 가장 높은 지점입니다. 첫 화면의 기준 비교는 참고용이며 "
            "이 화면의 판정 기준은 변경되지 않습니다."
        )

    if churn_proba >= threshold:
        level, color, bg, action = (
            "🟠 캠페인 대상 고객", "#c45d16", "#fff4e8",
            "고객 특성에 맞는 재구매 유도 캠페인 검토",
        )
    else:
        level, color, bg, action = (
            "🟢 일반 관찰 고객", "#2e7d32", "#eaf7ea",
            "정기 고객 관리와 구매 주기 모니터링",
        )

    st.markdown(
        f"""
        <div style="background-color:{bg}; border-left:6px solid {color};
                    border-radius:8px; padding:20px 24px; margin-top:8px;">
            <div style="font-size:15px; color:{color}; font-weight:600;">{level}</div>
            <div style="font-size:36px; font-weight:700; color:{color}; margin:4px 0 8px 0;">
                {churn_proba:.1%}
            </div>
            <div style="font-size:14px; color:#444;">
                <b>추천 액션</b> — {action}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    progress_percent = int(round(min(max(churn_proba, 0.0), 1.0) * 100))
    st.progress(progress_percent)

    with st.expander("⚠️ 모델 한계 및 주의사항"):
        st.markdown("""
        - recency_days가 340일 이상인 구간은 실제 이탈률이 포화 상태(70% 안팎)라,
          예측 확률의 미세한 등락은 절대값보다 방향성으로 해석하는 것을 권장합니다.
        - 본 모델은 정형 데이터(구매 이력 기반)로 학습되었으며, 상품 리뷰·문의 등
          비정형 데이터는 반영되지 않았습니다.
        - 입력값이 학습 데이터의 일반적인 범위를 크게 벗어나면 예측 신뢰도가 낮아질 수 있습니다.
        - 예측 확률은 별도의 확률 보정 모델을 거친 실제 발생률이 아니라 위험 순위 판단용 값입니다.
        """)
