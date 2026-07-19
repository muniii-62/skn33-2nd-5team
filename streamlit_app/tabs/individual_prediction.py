import pandas as pd
import streamlit as st
from config import FEATURE_ORDER


def render(model, preprocessor):
    st.header("개별 고객 이탈 예측")

    col1, col2 = st.columns(2)

    with col1:
        recency_days = st.slider("최근 구매 후 경과일 (recency_days)", 0, 400, 60, key="ind_recency")
        frequency = st.slider("구매 횟수 (frequency)", 1, 50, 5, key="ind_frequency")
        distinct_products = st.slider("구매 상품 종류 수 (distinct_products)", 1, 200, 10, key="ind_distinct")
        net_revenue = st.number_input("순매출 (net_revenue, £)", 0.0, 50000.0, 500.0, key="ind_revenue")
        tenure_days = st.slider("가입 후 경과일 (tenure_days)", 0, 800, 200, key="ind_tenure")

    with col2:
        avg_days_between_orders = st.slider(
            "평균 구매 간격 (avg_days_between_orders)", 0.0, 400.0, 40.0, key="ind_avg_gap"
        )
        if recency_days >= 250:
            st.info(
                "ℹ️ 최근 250일 이상 미구매 고객군은 실제 데이터에서도 이탈률이 70% 안팎에서 "
                "포화되는 경향이 있어, 이 구간에서는 확률의 세부 변동보다 '이미 고위험군'이라는 "
                "판정 자체를 참고하시는 것을 권장합니다."
            )
        is_low_value = st.selectbox("평균 주문금액 하위 20%인가? (is_low_value)", [0, 1], key="ind_low_value")
        is_uk = st.selectbox("UK 거주인가? (is_uk)", [1, 0], key="ind_is_uk")
        has_return = st.selectbox("취소 경험이 있는가? (has_return)", [0, 1], key="ind_has_return")
        recent_activity_ratio = st.slider(
            "최근 90일 구매 비중 (recent_activity_ratio)", 0.0, 1.0, 0.1, key="ind_activity_ratio"
        )

    input_df = pd.DataFrame([{
        "net_revenue": net_revenue,
        "recency_days": recency_days,
        "frequency": frequency,
        "distinct_products": distinct_products,
        "tenure_days": tenure_days,
        "avg_days_between_orders": avg_days_between_orders,
        "is_low_value": is_low_value,
        "is_uk": is_uk,
        "has_return": has_return,
        "recent_activity_ratio": recent_activity_ratio,
    }])
    input_df = input_df[FEATURE_ORDER]

    # 원본 스케일 입력을 preprocessor로 변환 후 예측 (모델은 스케일링된 데이터로 학습됨)
    input_processed = preprocessor.transform(input_df)
    input_processed_df = pd.DataFrame(input_processed, columns=FEATURE_ORDER)

    churn_proba = model.predict_proba(input_processed_df)[0, 1]

    st.divider()

    threshold = st.slider("판정 기준(threshold)", 0.0, 1.0, 0.44, 0.01, key="ind_threshold")

    st.metric("이탈 확률", f"{churn_proba:.1%}")

    if churn_proba >= 0.65:
        st.error("🔴 고위험 고객")
        st.write("**추천 액션**: 전담 영업 담당자 즉시 연락, 맞춤 할인 오퍼 제공")
    elif churn_proba >= threshold:
        st.warning("🟡 중위험 고객")
        st.write("**추천 액션**: 재주문 리마인드 이메일 발송")
    else:
        st.success("🟢 저위험 고객")
        st.write("**추천 액션**: 별도 조치 불필요, 정기 뉴스레터만 유지")

    st.divider()
    with st.expander("모델 한계 및 주의사항"):
        st.markdown("""
        - recency_days가 340일 이상인 구간은 실제 이탈률이 포화 상태(70% 안팎)라,
          예측 확률의 미세한 등락은 절대값보다 방향성으로 해석하는 것을 권장합니다.
        - 본 모델은 정형 데이터(구매 이력 기반)로 학습되었으며, 상품 리뷰·문의 등
          비정형 데이터는 반영되지 않았습니다.
        - is_uk, is_low_value, has_return 피처는 이탈 예측 기여도가 낮게 나타났습니다
          (SHAP 분석 결과, LightGBM과 RandomForest 양쪽에서 일관됨).
        """)