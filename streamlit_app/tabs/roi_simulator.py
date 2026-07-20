"""캠페인 ROI 시뮬레이터 — 위험고객 세분화 탭과 동일한 활성 고객 모집단 사용.
(라벨이 없는 '현재' 고객이므로, 실제 이탈 여부 대신 모델의 이탈확률 기댓값으로 계산)"""

import pandas as pd
import streamlit as st

from customer_scoring import load_customer_table, score_customers


def render(model, preprocessor):
    st.markdown("### 💰 캠페인 ROI 시뮬레이터")
    st.caption(
        "이탈확률 상위 K% 고객에게 캠페인(쿠폰 등)을 돌렸을 때, "
        "예상되는 이탈 방지 효과와 비용 대비 이익을 계산합니다."
    )

    snap = score_customers(load_customer_table(), model, preprocessor)
    snap = snap.sort_values("이탈확률", ascending=False).reset_index(drop=True)
    total_customers = len(snap)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        k_percent = st.slider("🎯 상위 K% 타겟팅", 5, 100, 20, 5, key="roi_k",
                               help="이탈확률이 높은 상위 몇 %의 고객에게 캠페인을 보낼지")
    with col_b:
        coupon_cost = st.number_input("💸 1인당 캠페인 비용 (£)", 0.0, 100.0, 5.0, 0.5, key="roi_cost",
                                       help="쿠폰 할인액, 발송 비용 등 1인당 드는 비용")
    with col_c:
        success_rate = st.slider("✅ 캠페인 성공률", 0.0, 1.0, 0.3, 0.05, key="roi_success",
                                  help="캠페인을 받은 '진짜 이탈 위험' 고객 중 실제로 붙잡히는 비율 (가정값)")

    n_targeted = max(1, int(total_customers * k_percent / 100))
    targeted = snap.iloc[:n_targeted]

    # 라벨이 없는 미래 시점 예측이므로, 실제 이탈 수 대신 확률의 합(기댓값)을 사용
    expected_churners = targeted["이탈확률"].sum()
    expected_saved = expected_churners * success_rate
    avg_customer_value = targeted["net_revenue"].mean()
    expected_revenue_retained = expected_saved * avg_customer_value
    total_cost = n_targeted * coupon_cost
    net_benefit = expected_revenue_retained - total_cost
    roi_pct = (net_benefit / total_cost * 100) if total_cost > 0 else 0.0

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("타겟 고객 수", f"{n_targeted:,}명")
    m2.metric("예상 방지 이탈자", f"{expected_saved:.1f}명")
    m3.metric("캠페인 총비용", f"£{total_cost:,.0f}")
    m4.metric("순이익", f"£{net_benefit:,.0f}", delta=f"ROI {roi_pct:.0f}%")

    st.caption(
        f"※ 계산 가정: 타겟 그룹의 평균 순매출(£{avg_customer_value:,.0f})을 "
        "고객 1인당 가치로 사용, 캠페인 성공률은 사용자가 입력한 가정값입니다."
    )

    st.markdown("#### K%에 따른 순이익 변화")
    sweep_rows = []
    for k in range(5, 105, 5):
        n = max(1, int(total_customers * k / 100))
        grp = snap.iloc[:n]
        exp_saved = grp["이탈확률"].sum() * success_rate
        revenue = exp_saved * grp["net_revenue"].mean()
        cost = n * coupon_cost
        sweep_rows.append({"상위 K%": k, "순이익(£)": revenue - cost})
    sweep_df = pd.DataFrame(sweep_rows)
    st.line_chart(sweep_df.set_index("상위 K%"))
    st.caption(f"현재 설정: 상위 {k_percent}% 지점 (그래프에서 위치 확인)")
