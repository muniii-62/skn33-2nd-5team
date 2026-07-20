import pandas as pd
import streamlit as st
from config import FEATURE_ORDER


def render(model, preprocessor):
    st.markdown("### 🔍 개별 고객 이탈 예측")
    st.caption("고객 정보를 입력하면 이탈 확률과 추천 액션을 실시간으로 확인할 수 있습니다.")

    with st.container(border=True):
        st.markdown("#### 📋 구매 이력")
        col1, col2 = st.columns(2)

        with col1:
            recency_days = st.slider(
                "🕒 최근 구매 후 경과일", 0, 400, 60, key="ind_recency",
                help="마지막 구매일로부터 오늘까지 지난 일수",
            )
            frequency = st.slider(
                "🔁 구매 횟수", 1, 50, 5, key="ind_frequency",
                help="관찰 기간 동안의 총 주문 건수",
            )
            distinct_products = st.slider(
                "📦 구매 상품 종류 수", 1, 200, 10, key="ind_distinct",
                help="지금까지 구매한 서로 다른 상품(품목코드)의 가짓수",
            )
            net_revenue = st.number_input(
                "💰 순매출 (£)", 0.0, 50000.0, 500.0, key="ind_revenue",
                help="관찰 기간 동안 이 고객이 실제로 지불한 금액의 합계 "
                     "(반품·취소분은 제외한 순수 매출). 고객의 구매 규모를 나타내는 지표입니다.",
            )
            st.caption("💡 순매출 = 취소·반품을 제외한 실제 구매 금액 합계")

            tenure_days = st.slider(
                "📅 가입 후 경과일", 0, 800, 200, key="ind_tenure",
                help="첫 구매일로부터 오늘까지 지난 일수 (고객으로 지낸 기간)",
            )

        with col2:
            avg_days_between_orders = st.slider(
                "↔️ 평균 구매 간격 (일)", 0.0, 400.0, 40.0, key="ind_avg_gap",
                help="가입 후 경과일 ÷ 구매 횟수 — 이 고객이 평소 며칠에 한 번꼴로 재구매하는지",
            )
            has_return = st.selectbox(
                "↩️ 취소 경험 있음?", ["아니오", "예"], key="ind_has_return",
                help="관찰 기간 중 주문 취소/반품 이력이 한 번이라도 있었는지",
            )
            recent_activity_ratio = st.slider(
                "📈 최근 구매 활동 비중", 0.0, 1.0, 0.1, key="ind_activity_ratio",
                help="전체 구매 횟수 중 '최근 90일 이내' 구매가 차지하는 비율. "
                     "예: 총 10번 구매 중 최근 90일 내 3번이면 0.3 — 값이 높을수록 "
                     "최근에도 활발히 구매 중이라는 뜻입니다.",
            )
            st.caption("💡 예: 총 10번 구매 중 최근 90일 내 3번이면 0.3")

            with st.expander("부가 정보 (예측 영향 적음)"):
                st.caption(
                    "아래 두 항목은 SHAP 분석 결과 이탈 예측에 미치는 영향이 "
                    "거의 없는 것으로 확인됐습니다(LightGBM 기준). 참고용으로만 남겨둡니다."
                )
                is_low_value = st.selectbox(
                    "💸 평균 주문금액 하위 20%?", ["아니오", "예"], key="ind_low_value",
                    help="이 고객의 평균 주문금액이 전체 고객 하위 20%에 속하는지",
                )
                is_uk = st.selectbox(
                    "🇬🇧 UK 거주?", ["예", "아니오"], key="ind_is_uk",
                    help="영국 거주 고객인지 여부",
                )

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
        "is_low_value": 1 if is_low_value == "예" else 0,
        "is_uk": 1 if is_uk == "예" else 0,
        "has_return": 1 if has_return == "예" else 0,
        "recent_activity_ratio": recent_activity_ratio,
    }])
    input_df = input_df[FEATURE_ORDER]

    # 원본 스케일 입력을 preprocessor로 변환 후 예측 (모델은 스케일링된 데이터로 학습됨)
    input_processed = preprocessor.transform(input_df)
    input_processed_df = pd.DataFrame(input_processed, columns=FEATURE_ORDER)
    churn_proba = model.predict_proba(input_processed_df)[0, 1]

    st.write("")
    threshold = st.slider(
        "⚙️ 판정 기준 (threshold)", 0.0, 1.0, 0.44, 0.01, key="ind_threshold",
        help="이탈확률이 이 값 이상이면 '중위험'으로 분류합니다. 값을 낮추면 더 많은 "
             "고객을 위험군으로 잡아내지만(Recall↑) 오탐도 늘어납니다(Precision↓).",
    )
    with st.expander("threshold가 뭔가요?"):
        st.markdown(
            "모델이 출력하는 건 0~1 사이의 '이탈 확률'이고, threshold는 그 확률을 "
            "**몇 % 이상부터 위험군으로 볼지** 정하는 기준선입니다.\n\n"
            "- **threshold를 낮추면** → 위험군으로 분류되는 고객이 늘어남 (덜 놓침, Recall↑) "
            "대신 실제로는 안 떠날 고객까지 위험군에 섞임 (Precision↓)\n"
            "- **threshold를 높이면** → 확실히 위험한 고객만 골라냄 (Precision↑) "
            "대신 놓치는 이탈 고객이 늘어남 (Recall↓)\n\n"
            "팀에서는 '이탈 고객을 놓치지 않는 것'을 우선순위로 두고, "
            "**Recall 0.80 이상을 유지하는 선에서 F1-score가 가장 높은 지점**을 "
            "기준값으로 선택했습니다 (RF 튜닝 결과 기준 약 0.40~0.44)."
        )

    if churn_proba >= 0.65:
        level, color, bg, action = (
            "🔴 고위험 고객", "#c0392b", "#fdecea",
            "전담 영업 담당자 즉시 연락, 맞춤 할인 오퍼 제공",
        )
    elif churn_proba >= threshold:
        level, color, bg, action = (
            "🟡 중위험 고객", "#b7791f", "#fff8e6",
            "재주문 리마인드 이메일 발송",
        )
    else:
        level, color, bg, action = (
            "🟢 저위험 고객", "#2e7d32", "#eaf7ea",
            "별도 조치 불필요, 정기 뉴스레터만 유지",
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
    st.progress(min(max(churn_proba, 0.0), 1.0))

    with st.expander("⚠️ 모델 한계 및 주의사항"):
        st.markdown("""
        - recency_days가 340일 이상인 구간은 실제 이탈률이 포화 상태(70% 안팎)라,
          예측 확률의 미세한 등락은 절대값보다 방향성으로 해석하는 것을 권장합니다.
        - 본 모델은 정형 데이터(구매 이력 기반)로 학습되었으며, 상품 리뷰·문의 등
          비정형 데이터는 반영되지 않았습니다.
        - is_uk, is_low_value, has_return 피처는 이탈 예측 기여도가 낮게 나타났습니다
          (SHAP 분석 결과, LightGBM과 RandomForest 양쪽에서 일관됨).
        """)