"""마케팅·CRM 담당자를 위한 위험고객 세분화 운영 화면.

[Doo 작업]
- 적용 Threshold 기반 CRM 핵심 지표와 고객 우선순위 표시
- 관리 범위·고객 유형 복합 필터와 CustomerID 검색 구현
- 선택 고객의 거래 상태·위험 요인·추천 캠페인 상세 정보 연결
- 현재 필터 결과와 전체 캠페인 대상을 구분한 Excel·CSV 다운로드 구현

기존 XGBoost 모델, 전처리, 이탈 확률 및 고객 유형 계산 로직은 변경하지 않습니다.
"""

from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from customer_scoring import load_customer_table, score_customers, segment


BADGE_BY_TYPE = {
    "이탈 위험 높음": "리텐션 우선",
    "첫 구매 고객": "웰컴 캠페인",
    "장기 구매 주기": "리마인드 예정",
    "정상": "모니터링",
}

CAMPAIGN_BY_TYPE = {
    "이탈 위험 높음": "개인화 리텐션 쿠폰",
    "첫 구매 고객": "웰컴 캠페인",
    "장기 구매 주기": "구매 주기 리마인드",
    "정상": "일반 리텐션 캠페인",
}

PRIORITY_COLORS = {
    "긴급": ("#FDECEC", "#C62828", "🔴"),
    "높음": ("#FFF3E0", "#E56A00", "🟠"),
    "관리": ("#EAF3FF", "#1F6FCC", "🔵"),
    "관찰": ("#F2F4F7", "#667085", "⚪"),
}


def _priority_label(probability: float, threshold: float) -> str:
    """[Doo 작업] 적용 기준과 확률을 함께 사용해 CRM 우선순위를 정합니다."""
    if probability < threshold:
        return "관찰"
    if probability >= 0.80:
        return "긴급"
    if probability >= 0.60:
        return "높음"
    return "관리"


def _customer_type_group(row: pd.Series) -> str:
    """[Doo 작업] 기존 고객 유형을 바꾸지 않고 UI 필터용 범주만 추가합니다."""
    if row["고객유형"] == "첫 구매 고객":
        return "첫 구매 고객"
    if row["고객유형"] == "장기 구매 주기":
        return "장기 구매 주기 고객"
    if row["frequency"] > 1:
        return "반복 구매 고객"
    return "기타 관리 고객"


def _campaign_plan(customer: pd.Series) -> dict:
    """[Doo 작업] 기존 고객 유형 규칙을 실무형 추천 문구로 확장합니다."""
    probability = customer["이탈확률"]
    customer_type = customer["고객유형"]

    if customer_type == "이탈 위험 높음":
        return {
            "name": "개인화 리텐션 쿠폰",
            "reason": f"평소 구매 주기를 초과했고 현재 이탈 확률이 {probability:.1%}입니다.",
            "actions": [
                "7일 이내 사용 가능한 개인화 쿠폰 발송",
                "이전 구매 상품과 연관된 상품 추천",
                "이메일 또는 앱 푸시로 우선 안내",
            ],
        }
    if customer_type == "첫 구매 고객":
        return {
            "name": "웰컴 캠페인",
            "reason": f"첫 구매 이후 추가 구매가 확인되지 않았고 현재 이탈 확률이 {probability:.1%}입니다.",
            "actions": [
                "첫 구매 고객 전용 재구매 쿠폰 제공",
                "첫 구매 상품과 연관된 상품 추천",
                "한 달 이내 리텐션 메시지 예약",
            ],
        }
    if customer_type == "장기 구매 주기":
        return {
            "name": "구매 주기 리마인드",
            "reason": "평균 구매 간격이 긴 고객이므로 기존 구매 주기에 맞춘 접근이 적합합니다.",
            "actions": [
                "예상 재구매 시점에 맞춰 리마인드 발송",
                "과도한 조기 할인 발송은 제외",
                "관심 상품 재입고 또는 추천 상품 안내",
            ],
        }
    return {
        "name": "일반 리텐션 캠페인",
        "reason": f"거래 패턴과 현재 이탈 확률 {probability:.1%}를 함께 고려한 관리가 필요합니다.",
        "actions": [
            "일반 리텐션 메시지 발송",
            "최근 구매 상품 기반 추천 제공",
            "캠페인 반응 여부를 확인해 후속 조치 결정",
        ],
    }


def _risk_factors(customer: pd.Series) -> list[str]:
    """[Doo 작업] SHAP 대신 실제 거래 데이터에 해당하는 위험 요인만 설명합니다."""
    factors = []
    recency = customer.get("recency_days")
    cycle_ratio = customer.get("평소_주기_대비")
    frequency = customer.get("frequency")
    recent_ratio = customer.get("recent_activity_ratio")
    has_return = customer.get("has_return")

    if pd.notna(recency) and recency >= 60:
        factors.append(f"마지막 구매 후 {int(recency)}일이 경과했습니다.")
    if pd.notna(cycle_ratio) and cycle_ratio >= 1.2:
        factors.append(f"평균 구매 간격보다 {cycle_ratio:.1f}배 오래 구매하지 않았습니다.")
    if pd.notna(frequency) and int(frequency) == 1:
        factors.append("구매 횟수가 1회인 첫 구매 고객입니다.")
    if pd.notna(recent_ratio) and recent_ratio == 0:
        factors.append("최근 90일 동안 재구매 활동이 확인되지 않았습니다.")
    if has_return == 1:
        factors.append("취소 또는 반품 경험이 확인됩니다.")

    return factors


def _customer_interpretation(customer: pd.Series, threshold: float) -> str:
    """[Doo 작업] 모델 원인을 단정하지 않고 관찰된 거래 상태를 요약합니다."""
    customer_type = customer["고객유형"]
    cycle_ratio = customer.get("평소_주기_대비")
    probability = customer["이탈확률"]

    if customer_type == "첫 구매 고객":
        return "첫 구매 이후 추가 구매가 확인되지 않아 재구매 유도 캠페인을 검토할 수 있습니다."
    if customer_type == "장기 구매 주기":
        return "구매 주기가 긴 고객이므로 일반 고객보다 더 긴 관찰 기간을 고려해야 합니다."
    if pd.notna(cycle_ratio) and cycle_ratio >= 1.5:
        return "평균 구매 주기를 크게 초과했고 이탈 확률도 높아 빠른 캠페인 검토가 필요합니다."
    if probability >= threshold:
        return "현재 적용 기준 이상으로 분류되었습니다. 거래 패턴과 캠페인 비용을 함께 검토해주세요."
    return "현재 캠페인 대상 기준 미만입니다. 향후 구매 활동 변화를 관찰해주세요."


def _build_campaign_export(
    source_df: pd.DataFrame,
    threshold: float,
    *,
    campaign_only: bool = True,
    data_as_of=None,
) -> pd.DataFrame:
    """[Doo 작업] 전체 캠페인 대상 또는 현재 필터 결과를 다운로드 표로 만듭니다."""
    export_df = source_df.copy()
    if campaign_only:
        export_df = export_df[export_df["이탈확률"] >= threshold].copy()
    export_df = export_df.sort_values("이탈확률", ascending=False)

    generated_at = datetime.now().astimezone()
    if data_as_of is None:
        data_as_of = source_df["last_purchase"].max()
    data_as_of_text = data_as_of.date().isoformat() if pd.notna(data_as_of) else ""

    export_df["추천캠페인"] = export_df["고객유형"].map(CAMPAIGN_BY_TYPE).fillna("일반 리텐션 캠페인")
    # [Doo 작업] XGBoost float32 확률을 변환해 CSV의 소수 오차를 제거합니다.
    export_df["이탈확률(%)"] = (export_df["이탈확률"].astype(float) * 100).round(1)
    export_df["적용기준(%)"] = round(threshold * 100, 1)
    export_df["취소·반품경험"] = export_df["has_return"].map({1: "있음", 0: "없음"})
    export_df["데이터기준일"] = data_as_of_text
    export_df["파일생성시각"] = generated_at.strftime("%Y-%m-%d %H:%M:%S %Z")

    export_columns = {
        "CustomerID": "CustomerID",
        "이탈확률(%)": "이탈확률(%)",
        "적용기준(%)": "적용기준(%)",
        "모델판정": "모델판정",
        "고객유형": "고객유형",
        "추천캠페인": "추천캠페인",
        "recency_days": "최근구매후경과일",
        "avg_days_between_orders": "평균구매간격(일)",
        "평소_주기_대비": "평소주기대비(배)",
        "frequency": "구매횟수",
        "distinct_products": "구매상품종류수",
        "net_revenue": "순매출",
        "취소·반품경험": "취소·반품경험",
        "데이터기준일": "데이터기준일",
        "파일생성시각": "파일생성시각",
    }
    result = export_df[list(export_columns)].rename(columns=export_columns).reset_index(drop=True)
    if not result.empty:
        result["CustomerID"] = result["CustomerID"].astype(int)
    result["평균구매간격(일)"] = result["평균구매간격(일)"].round(1)
    result["순매출"] = result["순매출"].round(2)
    return result


def _build_campaign_excel(campaign_df: pd.DataFrame, threshold: float, scope_label: str) -> bytes:
    """[Doo 작업] 담당자 검토용 서식과 사용 안내를 포함한 Excel 파일을 만듭니다."""
    buffer = BytesIO()
    guide_df = pd.DataFrame(
        {
            "항목": ["파일 목적", "다운로드 범위", "선정 기준", "고객 수", "데이터 출처", "사용 안내"],
            "내용": [
                "마케팅·CRM 고객 목록 검토",
                scope_label,
                f"현재 적용된 캠페인 기준 {threshold:.0%}",
                f"{len(campaign_df):,}명",
                "online_retail_II 거래 데이터를 고객 단위로 집계한 스냅샷",
                "CustomerID를 사내 CRM 고객 정보와 연결한 뒤 캠페인 발송에 사용하세요.",
            ],
        }
    )

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        campaign_df.to_excel(writer, sheet_name="고객 목록", index=False)
        guide_df.to_excel(writer, sheet_name="사용 안내", index=False)

        target_sheet = writer.sheets["고객 목록"]
        guide_sheet = writer.sheets["사용 안내"]
        header_fill = PatternFill("solid", fgColor="2F80ED")
        header_font = Font(color="FFFFFF", bold=True)

        # [Doo 작업] Excel 검토 편의를 위해 서식·필터·첫 행 고정을 적용합니다.
        for sheet in (target_sheet, guide_sheet):
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions

        for sheet in (target_sheet, guide_sheet):
            for column_index, column_cells in enumerate(sheet.columns, start=1):
                max_length = max(len(str(cell.value or "")) for cell in column_cells)
                sheet.column_dimensions[get_column_letter(column_index)].width = min(max_length + 3, 45)

        if not campaign_df.empty:
            probability_column = campaign_df.columns.get_loc("이탈확률(%)") + 1
            threshold_column = campaign_df.columns.get_loc("적용기준(%)") + 1
            revenue_column = campaign_df.columns.get_loc("순매출") + 1
            for row_index in range(2, len(campaign_df) + 2):
                target_sheet.cell(row=row_index, column=probability_column).number_format = '0.0"%"'
                target_sheet.cell(row=row_index, column=threshold_column).number_format = '0.0"%"'
                target_sheet.cell(row=row_index, column=revenue_column).number_format = '#,##0.00'

    buffer.seek(0)
    return buffer.getvalue()


def _kpi_card(column, label: str, value: str, description: str):
    """[Doo 작업] CRM 핵심 수치를 동일한 카드 형식으로 표시합니다."""
    with column:
        with st.container(border=True):
            st.caption(label)
            st.markdown(f"<div style='font-size:28px;font-weight:750'>{value}</div>", unsafe_allow_html=True)
            st.caption(description)


def _detail_row(label, value):
    value = "정보 없음" if pd.isna(value) else value
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:7px 0; border-bottom:1px solid #F0F0F0; gap:12px;">
            <span style="color:#667085; font-size:13px;">{label}</span>
            <span style="font-weight:700; font-size:14px; text-align:right;">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _priority_badge(priority: str) -> str:
    background, color, icon = PRIORITY_COLORS[priority]
    return (
        f'<span style="background:{background};color:{color};font-weight:700;'
        f'padding:4px 10px;border-radius:14px;font-size:12px;">{icon} {priority}</span>'
    )


def _render_customer_detail(customer: pd.Series | None, threshold: float, data_as_of_text: str):
    """[Doo 작업] 선택 고객의 상태·위험 요인·추천 실행을 하나의 상세 카드로 연결합니다."""
    st.markdown("#### 선택 고객 상세 정보")
    if customer is None:
        st.info("고객 목록에서 고객을 선택하거나 CustomerID를 검색하면 상세 정보를 확인할 수 있습니다.")
        return

    priority = customer["우선순위"]
    campaign = _campaign_plan(customer)
    probability = customer["이탈확률"]
    customer_id = int(customer["CustomerID"])
    type_badge = BADGE_BY_TYPE.get(customer["고객유형"], "모니터링")

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
                <div>
                    <div style="color:#667085;font-size:12px;">CustomerID</div>
                    <div style="font-size:24px;font-weight:750;">{customer_id}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:30px;font-weight:750;">{probability:.1%}</div>
                    <div style="color:#667085;font-size:12px;">이탈 확률</div>
                </div>
            </div>
            <div style="margin:12px 0;display:flex;gap:8px;flex-wrap:wrap;">
                {_priority_badge(priority)}
                <span style="background:#F2F4F7;color:#344054;font-weight:650;
                             padding:4px 10px;border-radius:14px;font-size:12px;">
                    {customer['고객유형']}
                </span>
                <span style="background:#EEF4FF;color:#3538CD;font-weight:650;
                             padding:4px 10px;border-radius:14px;font-size:12px;">
                    {type_badge}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        detail_left, detail_right = st.columns(2)
        with detail_left:
            _detail_row("최근 구매 후", f"{int(customer['recency_days'])}일")
            _detail_row("평균 구매 간격", f"{customer['avg_days_between_orders']:.1f}일")
            cycle_ratio = customer["평소_주기_대비"]
            _detail_row("평소 주기 대비", f"{cycle_ratio:.1f}배" if pd.notna(cycle_ratio) else "정보 없음")
            _detail_row("총 구매 횟수", f"{int(customer['frequency'])}회")
        with detail_right:
            _detail_row("구매 상품 종류", f"{int(customer['distinct_products']):,}개")
            _detail_row("순매출", f"£{customer['net_revenue']:,.2f}")
            _detail_row("취소·반품 경험", "있음" if customer["has_return"] == 1 else "없음")
            _detail_row("데이터 기준일", data_as_of_text or "정보 없음")

        st.info(_customer_interpretation(customer, threshold))

        st.markdown("**주요 위험 요인**")
        factors = _risk_factors(customer)
        if factors:
            st.markdown("\n".join(f"- {factor}" for factor in factors))
        else:
            st.caption(
                "거래 데이터에서 뚜렷한 단일 위험 요인은 확인되지 않았습니다. "
                "현재 이탈 확률과 전체 구매 패턴을 함께 검토해주세요."
            )

        st.markdown("**추천 캠페인**")
        st.markdown(f"**{campaign['name']}**")
        st.caption(f"추천 이유 · {campaign['reason']}")
        st.markdown("**권장 실행 방법**")
        st.markdown("\n".join(f"- {action}" for action in campaign["actions"]))


def _clear_customer_search():
    """[Doo 작업] 검색어와 선택 고객을 함께 초기화합니다."""
    st.session_state["risk_search_input"] = ""
    st.session_state["selected_customer_id"] = None


def _render_download_group(
    title: str,
    description: str,
    export_df: pd.DataFrame,
    threshold: float,
    file_prefix: str,
    scope_label: str,
    key_prefix: str,
):
    """[Doo 작업] 다운로드 범위와 파일 용도를 분리해 혼동을 방지합니다."""
    st.markdown(f"**{title}**")
    st.caption(description)

    if export_df.empty:
        st.info("현재 조건에 해당하는 고객이 없습니다. 필터 조건이나 캠페인 기준을 조정해주세요.")
        disabled_left, disabled_right = st.columns(2)
        disabled_left.download_button(
            "담당자 검토용 Excel 다운로드", data=b"", file_name="empty.xlsx",
            disabled=True, width="stretch", key=f"{key_prefix}_excel_empty",
        )
        disabled_right.download_button(
            "CRM 업로드용 CSV 다운로드", data=b"", file_name="empty.csv",
            disabled=True, width="stretch", key=f"{key_prefix}_csv_empty",
        )
        return

    file_date = datetime.now().astimezone().strftime("%Y%m%d")
    threshold_label = int(round(threshold * 100))
    try:
        excel_bytes = _build_campaign_excel(export_df, threshold, scope_label)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
    except Exception as error:
        st.error("파일을 생성하지 못했습니다. 데이터 상태를 확인한 뒤 다시 시도해주세요.")
        with st.expander("개발자 확인용 오류 정보"):
            st.code(str(error))
        return

    excel_column, csv_column = st.columns(2)
    excel_column.download_button(
        "담당자 검토용 Excel 다운로드",
        data=excel_bytes,
        file_name=f"{file_prefix}_threshold_{threshold_label}pct_{file_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        width="stretch",
        key=f"{key_prefix}_excel",
    )
    csv_column.download_button(
        "CRM 업로드용 CSV 다운로드",
        data=csv_bytes,
        file_name=f"{file_prefix}_threshold_{threshold_label}pct_{file_date}.csv",
        mime="text/csv",
        width="stretch",
        key=f"{key_prefix}_csv",
    )


def render(model, preprocessor):
    """[Doo 작업] 위험고객 세분화 CRM 운영 화면을 렌더링합니다."""
    snap = score_customers(load_customer_table(), model, preprocessor)
    threshold = st.session_state["applied_threshold"]

    # 기존 고객 유형과 이탈 확률 계산은 그대로 유지하고 UI용 파생값만 추가합니다.
    snap["고객유형"] = snap.apply(segment, axis=1)
    snap["모델판정"] = snap["이탈확률"].ge(threshold).map(
        {True: "캠페인 대상 고객", False: "일반 관찰 고객"}
    )
    snap["평소_주기_대비"] = (
        snap["recency_days"] / snap["avg_days_between_orders"]
    ).replace([float("inf"), float("-inf")], pd.NA).round(2)
    snap["우선순위"] = snap["이탈확률"].apply(lambda value: _priority_label(value, threshold))
    snap["고객유형필터"] = snap.apply(_customer_type_group, axis=1)
    snap["추천캠페인"] = snap["고객유형"].map(CAMPAIGN_BY_TYPE).fillna("일반 리텐션 캠페인")

    if "selected_customer_id" not in st.session_state:
        st.session_state["selected_customer_id"] = None
    if "risk_search_input" not in st.session_state:
        st.session_state["risk_search_input"] = ""

    data_as_of = snap["last_purchase"].max()
    data_as_of_text = data_as_of.date().isoformat() if pd.notna(data_as_of) else ""
    campaign_target = snap[snap["이탈확률"] >= threshold]
    high_risk = snap[snap["이탈확률"] >= 0.80]

    st.markdown("### 🔎 위험고객 세분화")
    st.markdown(
        "현재 적용된 캠페인 기준 이상인 고객을 우선 관리 대상으로 분류합니다.  \n"
        "고객의 위험도와 구매 패턴을 확인하고 적합한 리텐션 캠페인을 선택할 수 있습니다."
    )
    st.markdown(
        f"""
        <div style="background:#EAF3FF;border:1px solid #BBD7FF;border-radius:10px;
                    padding:14px 18px;margin:12px 0 18px 0;">
            <div style="color:#1F6FCC;font-size:13px;font-weight:700;">현재 적용된 캠페인 기준</div>
            <div style="font-size:24px;font-weight:750;margin:4px 0;">{threshold:.0%}</div>
            <div style="color:#425466;font-size:13px;">
                이탈 확률이 {threshold:.0%} 이상인 고객이 캠페인 대상입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1) CRM 핵심 KPI
    first_kpi_row = st.columns(3)
    _kpi_card(first_kpi_row[0], "전체 고객", f"{len(snap):,}명", "현재 고객 스냅샷 전체")
    _kpi_card(
        first_kpi_row[1], "캠페인 대상 고객", f"{len(campaign_target):,}명",
        f"현재 적용 기준 {threshold:.0%} 이상",
    )
    _kpi_card(first_kpi_row[2], "고위험 고객", f"{len(high_risk):,}명", "이탈 확률 80% 이상")
    second_kpi_row = st.columns(3)
    first_purchase_count = int((snap["고객유형"] == "첫 구매 고객").sum())
    long_cycle_count = int((snap["고객유형"] == "장기 구매 주기").sum())
    average_target_probability = campaign_target["이탈확률"].mean() if not campaign_target.empty else 0
    _kpi_card(second_kpi_row[0], "첫 구매 고객", f"{first_purchase_count:,}명", "구매 횟수 1회")
    _kpi_card(second_kpi_row[1], "장기 구매 주기 고객", f"{long_cycle_count:,}명", "평균 구매 간격 90일 이상")
    _kpi_card(
        second_kpi_row[2], "대상 고객 평균 이탈 확률", f"{average_target_probability:.1%}",
        "현재 캠페인 대상 고객 평균",
    )

    # 2) CustomerID 검색 — 필터보다 먼저 제공하고 검색 결과는 필터와 무관하게 선택합니다.
    st.markdown("#### CustomerID 검색")
    search_column, clear_column = st.columns([5, 1])
    with search_column:
        with st.form("risk_customer_search_form"):
            search_query = st.text_input(
                "CustomerID",
                placeholder="예: 17850",
                key="risk_search_input",
                label_visibility="collapsed",
            )
            search_submitted = st.form_submit_button("고객 검색", type="primary", width="stretch")
    with clear_column:
        st.button("검색 초기화", on_click=_clear_customer_search, width="stretch")

    if search_submitted:
        try:
            searched_customer_id = int(search_query.strip())
            matched = snap[snap["CustomerID"] == searched_customer_id]
        except (ValueError, AttributeError):
            matched = pd.DataFrame()
        if matched.empty:
            st.error("해당 CustomerID를 찾을 수 없습니다. 입력한 번호를 다시 확인해주세요.")
        else:
            st.session_state["selected_customer_id"] = searched_customer_id
            st.success(f"CustomerID {searched_customer_id} 고객을 선택했습니다. 필터와 관계없이 상세 정보를 표시합니다.")

    # 3) 관리 범위와 고객 유형을 독립적으로 적용합니다.
    st.markdown("#### 고객 필터")
    management_counts = {
        "전체 고객": len(snap),
        "캠페인 대상 고객": len(campaign_target),
        "고위험 고객": len(high_risk),
    }
    type_options = ["전체 유형", "첫 구매 고객", "반복 구매 고객", "장기 구매 주기 고객", "기타 관리 고객"]

    management_column, type_column = st.columns(2)
    management_filter = management_column.selectbox(
        "관리 범위",
        list(management_counts),
        format_func=lambda option: f"{option}  {management_counts[option]:,}명",
        key="risk_management_filter",
    )
    type_filter = type_column.selectbox(
        "고객 유형",
        type_options,
        # [Doo 작업] 고객 유형 드롭다운은 인원 수 없이 유형명만 간결하게 표시합니다.
        key="risk_type_filter",
    )

    view = snap
    if management_filter == "캠페인 대상 고객":
        view = view[view["이탈확률"] >= threshold]
    elif management_filter == "고위험 고객":
        view = view[view["이탈확률"] >= 0.80]
    if type_filter != "전체 유형":
        view = view[view["고객유형필터"] == type_filter]
    view = view.sort_values("이탈확률", ascending=False).reset_index(drop=True)

    # 4) 목록과 상세 정보를 65:35로 연결합니다.
    list_column, detail_column = st.columns([1.65, 1], gap="large")
    with list_column:
        st.markdown(f"#### 우선순위 고객 목록 · {len(view):,}명")
        st.caption("이탈 확률이 높은 순으로 표시됩니다. 표의 열 제목을 눌러 다시 정렬할 수 있습니다.")
        if view.empty:
            st.info("현재 조건에 해당하는 고객이 없습니다. 필터 조건이나 캠페인 기준을 조정해주세요.")
        else:
            display_df = view[
                ["CustomerID", "고객유형", "이탈확률", "평소_주기_대비", "추천캠페인", "우선순위"]
            ].copy()
            display_df.insert(0, "등급", display_df["우선순위"].map(
                lambda priority: f"{PRIORITY_COLORS[priority][2]} {priority}"
            ))
            display_df = display_df.drop(columns="우선순위")
            display_df["이탈확률"] = (display_df["이탈확률"].astype(float) * 100).round(1)
            display_df["평소_주기_대비"] = display_df["평소_주기_대비"].astype(float).round(1)

            table_event = st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
                height=500,
                on_select="rerun",
                selection_mode="single-row",
                key="risk_customer_table",
                column_config={
                    "등급": st.column_config.TextColumn("우선순위", width="small"),
                    "CustomerID": st.column_config.NumberColumn("CustomerID", format="%d"),
                    "고객유형": st.column_config.TextColumn("고객 유형", width="medium"),
                    "이탈확률": st.column_config.ProgressColumn(
                        "이탈 확률", format="%.1f%%", min_value=0, max_value=100,
                    ),
                    "평소_주기_대비": st.column_config.NumberColumn("평소 주기 대비", format="%.1f배"),
                    "추천캠페인": st.column_config.TextColumn("추천 캠페인", width="medium"),
                },
            )
            selected_rows = table_event.selection.rows if table_event.selection else []
            if selected_rows:
                selected_customer_id = int(view.iloc[selected_rows[0]]["CustomerID"])
                st.session_state["selected_customer_id"] = selected_customer_id

    selected_customer = None
    selected_customer_id = st.session_state.get("selected_customer_id")
    if selected_customer_id is not None:
        matched_customer = snap[snap["CustomerID"] == int(selected_customer_id)]
        if not matched_customer.empty:
            selected_customer = matched_customer.iloc[0]
    with detail_column:
        _render_customer_detail(selected_customer, threshold, data_as_of_text)

    # 5) 다운로드 범위를 현재 필터 결과와 전체 캠페인 대상으로 명확히 분리합니다.
    st.divider()
    st.markdown("### 캠페인 실행용 고객 목록 다운로드")
    st.markdown(
        f"현재 적용 기준 **{threshold:.0%} 이상**, 전체 캠페인 대상 고객 **{len(campaign_target):,}명**입니다.  \n"
        "화면 필터 결과와 전체 캠페인 대상의 다운로드 범위를 구분해 사용하세요."
    )

    filtered_export = _build_campaign_export(
        view, threshold, campaign_only=False, data_as_of=data_as_of,
    )
    full_campaign_export = _build_campaign_export(
        snap, threshold, campaign_only=True, data_as_of=data_as_of,
    )
    download_summary = st.columns(4)
    download_summary[0].metric("현재 필터 결과", f"{len(filtered_export):,}명")
    download_summary[1].metric("전체 캠페인 대상", f"{len(full_campaign_export):,}명")
    download_summary[2].metric("적용 기준", f"{threshold:.0%}")
    download_summary[3].metric("포함 컬럼", f"{len(full_campaign_export.columns)}개")
    st.caption(f"데이터 기준일 {data_as_of_text or '정보 없음'} · Excel은 검토용, CSV는 시스템 업로드용")

    filtered_download_column, full_download_column = st.columns(2, gap="large")
    with filtered_download_column:
        with st.container(border=True):
            _render_download_group(
                "현재 필터 결과 다운로드",
                f"현재 관리 범위와 고객 유형 필터를 모두 만족하는 {len(filtered_export):,}명만 포함합니다.",
                filtered_export,
                threshold,
                "filtered_customers",
                "현재 화면 필터 결과",
                "filtered_customers",
            )
    with full_download_column:
        with st.container(border=True):
            _render_download_group(
                "전체 캠페인 대상 다운로드",
                f"화면 필터와 관계없이 현재 적용 기준 이상 고객 {len(full_campaign_export):,}명을 포함합니다.",
                full_campaign_export,
                threshold,
                "campaign_targets",
                "전체 캠페인 대상",
                "campaign_targets",
            )
