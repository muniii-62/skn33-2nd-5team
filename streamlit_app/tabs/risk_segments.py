import pandas as pd
import streamlit as st

from customer_scoring import load_customer_table, score_customers, segment

BADGE_BY_TYPE = {
    "이탈 위험 높음": "리텐션 우선",
    "첫 구매 고객": "웰컴 캠페인",
    "장기 구매 주기": "리마인드 예정",
    "정상": "모니터링",
}


def _detail_row(label, value):
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:6px 0; border-bottom:1px solid #F0F0F0;">
            <span style="color:#666; font-size:14px;">{label}</span>
            <span style="font-weight:700; font-size:14px;">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(model, preprocessor):
    snap = score_customers(load_customer_table(), model, preprocessor)

    # [Doo 작업] 캠페인 기준 설정 화면에서 `기준 적용하기`로 확정한 값을 가져와
    # 고객별 모델 위험군 여부를 계산합니다. 기존 구매주기 기반 고객유형은 유지합니다.
    threshold = st.session_state["applied_threshold"]
    snap["고객유형"] = snap.apply(segment, axis=1)
    snap["모델판정"] = snap["이탈확률"].ge(threshold).map(
        {True: "모델 위험군", False: "모델 일반군"}
    )
    snap["평소_주기_대비"] = (snap["recency_days"] / snap["avg_days_between_orders"]).round(2)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("캠페인 대상 고객")
        filter_choice = st.radio(
            "필터", ["전체", "모델 위험군", "이탈 위험 높음", "첫 구매 고객", "장기 구매 주기"],
            horizontal=True, key="risk_filter", label_visibility="collapsed",
        )

        # [Doo 작업] 기존 룰 기반 필터에 `모델 위험군` 필터를 추가했습니다.
        if filter_choice == "전체":
            view = snap
        elif filter_choice == "모델 위험군":
            view = snap[snap["모델판정"] == "모델 위험군"]
        else:
            view = snap[snap["고객유형"] == filter_choice]
        view = view.sort_values("이탈확률", ascending=False)

        st.markdown("**우선순위 고객 목록**")
        st.caption(
            f"캠페인 선정 기준 {threshold:.0%} · 총 {len(view):,}명 중 이탈확률 높은 순으로 표시 "
            "(행을 클릭하면 오른쪽에 상세 정보가 뜹니다)"
        )
        display_df = view[["CustomerID", "고객유형", "이탈확률", "평소_주기_대비"]].reset_index(drop=True).copy()
        display_df["이탈확률"] = (display_df["이탈확률"] * 100).round(1).astype(str) + "%"
        display_df["평소_주기_대비"] = display_df["평소_주기_대비"].astype(str) + "배"

        table_event = st.dataframe(
            display_df, hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row", key="risk_table",
        )

        selected_rows = table_event.selection.rows if table_event.selection else []
        if selected_rows:
            selected_cid = int(view.reset_index(drop=True).iloc[selected_rows[0]]["CustomerID"])
            st.session_state["risk_search"] = str(selected_cid)

    with col2:
        st.subheader("CustomerID 검색")
        query = st.text_input("CustomerID", placeholder="예: 17850", key="risk_search",
                               label_visibility="collapsed")

        if query:
            try:
                cid = int(query)
                matched = snap[snap["CustomerID"] == cid]
            except ValueError:
                matched = pd.DataFrame()

            if matched.empty:
                st.warning("해당 CustomerID를 찾을 수 없습니다.")
            else:
                cust = matched.iloc[0]
                badge = BADGE_BY_TYPE.get(cust["고객유형"], "모니터링")

                with st.container(border=True):
                    st.markdown(
                        f"""
                        <span style="background:#E7F0FF; color:#2F80ED; font-size:12px;
                                     font-weight:700; padding:3px 10px; border-radius:12px;">
                            {badge}
                        </span>
                        <div style="font-size:20px; font-weight:700; margin-top:8px;">고객 상세</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    _detail_row("CustomerID", int(cust["CustomerID"]))
                    _detail_row("최근 구매 후", f"{int(cust['recency_days'])}일")
                    _detail_row("평균 구매 간격", f"{cust['avg_days_between_orders']:.0f}일")
                    _detail_row("구매 횟수", f"{int(cust['frequency'])}회")
                    _detail_row("이탈 확률", f"{cust['이탈확률']:.1%}")

                    st.write("")
                    if cust["고객유형"] == "이탈 위험 높음":
                        st.info(f"평소 구매 주기보다 {cust['평소_주기_대비']:.2f}배 오래 구매하지 않았습니다.")
                        st.markdown("- 개인화 쿠폰 즉시 발송\n- 이전 구매 기반 추천 상품 제안\n- 리텐션 메시지 우선 발송")
                    elif cust["고객유형"] == "첫 구매 고객":
                        st.markdown("- 웰컴 쿠폰 발송\n- 한 달 뒤 리텐션 메시지 예약")
                    elif cust["고객유형"] == "장기 구매 주기":
                        st.markdown("- 평소 주기에 맞춘 리마인드 메시지")
                    else:
                        st.markdown("- 별도 조치 불필요")
