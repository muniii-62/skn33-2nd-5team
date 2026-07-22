"""데이터 현황·EDA와 모델 검증 결과를 한 화면에서 설명합니다."""

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PREPROCESSING_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "preprocessing"
TRAINING_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "training"


def _metric_card(column, label: str, value: str, description: str) -> None:
    with column:
        with st.container(border=True):
            st.caption(label)
            st.markdown(
                f"<div style='font-size:28px;font-weight:750'>{value}</div>",
                unsafe_allow_html=True,
            )
            st.caption(description)


@st.cache_data
def _load_metrics() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "metrics.csv")


def _show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"그래프 파일을 찾을 수 없습니다: {path.name}")


def _render_eda() -> None:
    st.markdown("### 데이터 현황·EDA")
    st.markdown(
        "UCI Online Retail II 거래 데이터를 고객 단위로 집계하고, 기준일 이후 "
        "90일 동안 정상 재구매가 없으면 `churn=1`로 정의했습니다."
    )
    st.info(
        "분석 데이터 기준 — 고객 스냅샷 4,320명입니다. 현재 고객 목록 4,295명과는 "
        "생성 시점과 용도가 달라 인원 수가 다를 수 있습니다."
    )

    summary = st.columns(4)
    _metric_card(summary[0], "원본 거래", "1,067,371행", "상품 한 줄이 원본 1행")
    _metric_card(summary[1], "분석 고객", "4,320명", "고객 1명 = 최종 1행")
    _metric_card(summary[2], "재구매 이탈", "2,134명", "90일 무구매 · 49.4%")
    _metric_card(summary[3], "재구매 고객", "2,186명", "90일 내 재구매 · 50.6%")

    st.markdown("#### Target 분포")
    target_df = pd.DataFrame(
        {"고객 수": [2_134, 2_186]},
        index=["재구매 이탈 (1)", "재구매 (0)"],
    )
    target_left, target_right = st.columns([1, 1.35], gap="large")
    with target_left:
        st.bar_chart(target_df, horizontal=True, color="#2F80ED")
    with target_right:
        with st.container(border=True):
            st.markdown("**클래스 상태**")
            st.markdown(
                "- 이탈 49.4%, 재구매 50.6%로 거의 균형입니다.\n"
                "- SMOTE·언더샘플링·오버샘플링은 적용하지 않았습니다.\n"
                "- Train·Validation·Test는 `stratify=y`로 비율을 유지했습니다."
            )

    st.markdown("#### 주요 Feature와 Target의 관계")
    relationship_left, relationship_right = st.columns(2, gap="large")
    with relationship_left:
        _show_image(
            PREPROCESSING_FIGURE_DIR / "target_feature_comparison.png",
            "재구매 고객과 이탈 고객의 주요 Feature 평균 비교",
        )
    with relationship_right:
        _show_image(
            PREPROCESSING_FIGURE_DIR / "feature_distributions.png",
            "주요 수치형 Feature 분포",
        )

    with st.expander("변수 간 관계와 데이터 품질 자세히 보기"):
        _show_image(
            PREPROCESSING_FIGURE_DIR / "correlation_heatmap.png",
            "Feature 및 Target 상관관계",
        )
        quality_df = pd.DataFrame(
            [
                ["CustomerID 결측", "243,007행 (22.8%)", "고객 추적이 불가능해 고객 단위 분석에서 제외"],
                ["완전 중복", "34,335행", "복수 수량 기록 가능성이 있어 일괄 삭제하지 않고 집계"],
                ["취소 송장", "19,494행", "정상 구매에서는 제외하고 반품 경험에는 활용"],
                ["Quantity ≤ 0", "22,950행", "정상 구매에서 제외"],
                ["Price ≤ 0", "6,207행", "정상 유상 구매가 아니므로 제외"],
            ],
            columns=["점검 항목", "규모", "처리 및 근거"],
        )
        st.dataframe(quality_df, hide_index=True, width="stretch")

    st.markdown("#### 핵심 인사이트")
    # [Doo 작업] avg_days_between_orders의 계산 의미에 맞춰 표시 명칭만 정정했다.
    st.markdown(
        "- 이탈 고객은 최근 거래 활동 후 경과일과 평균 구매 간격이 더 깁니다.\n"
        "- 이탈 고객은 구매 횟수·상품 다양성·과거 순매출이 상대적으로 낮습니다.\n"
        "- 순매출 등 일부 변수의 오른쪽 꼬리가 길어 로그 변환과 스케일링을 적용했습니다.\n"
        "- 위 결과는 집단 차이와 예측 신호이며 이탈의 인과 원인을 의미하지 않습니다."
    )


def _render_model_performance() -> None:
    metrics = _load_metrics()
    validation = metrics[metrics["split"] == "validation"].copy()
    final_test = metrics[
        (metrics["split"] == "test") & (metrics["model"] == "Final XGBoost")
    ].iloc[0]

    st.markdown("### 모델 성능")
    st.markdown(
        "동일한 Train·Validation·Test 분할에서 후보 모델을 비교하고, Validation에서 "
        "Recall 85% 이상인 Threshold 중 F1이 가장 높은 운영안을 선택했습니다."
    )
    st.info("검증 기준 — Train 2,592명 · Validation 864명 · Test 864명 · Random State 42")

    st.markdown("#### Validation 후보 비교")
    validation_display = validation[
        ["model", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    ].copy()
    validation_display.columns = [
        "모델", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"
    ]
    for column in ["Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
        validation_display[column] = validation_display[column].map(lambda value: f"{value:.3f}")
    st.dataframe(validation_display, hide_index=True, width="stretch")
    _show_image(
        TRAINING_FIGURE_DIR / "validation_model_comparison.png",
        "Validation 모델별 운영점 성능 비교",
    )

    st.markdown("#### 최종 Test 성능 — XGBoost, Threshold 38%")
    test_cards = st.columns(5)
    _metric_card(test_cards[0], "Precision", f"{final_test['precision']:.1%}", "선정 고객 적중률")
    _metric_card(test_cards[1], "Recall", f"{final_test['recall']:.1%}", "실제 이탈 고객 발견률")
    _metric_card(test_cards[2], "F1", f"{final_test['f1']:.3f}", "발견·적중 균형")
    _metric_card(test_cards[3], "ROC-AUC", f"{final_test['roc_auc']:.3f}", "순위 판별 성능")
    _metric_card(test_cards[4], "PR-AUC", f"{final_test['pr_auc']:.3f}", "이탈 탐지 품질")

    matrix_column, curve_column = st.columns(2, gap="large")
    with matrix_column:
        _show_image(
            TRAINING_FIGURE_DIR / "final_confusion_matrix.png",
            "최종 Test Confusion Matrix · TN 220 / FP 217 / FN 60 / TP 367",
        )
    with curve_column:
        _show_image(
            TRAINING_FIGURE_DIR / "test_roc_pr_curves.png",
            "최종 Test ROC 및 Precision-Recall Curve",
        )

    with st.container(border=True):
        st.markdown("**최종 XGBoost 선정 이유**")
        st.markdown(
            "Validation Recall 85% 조건을 만족하면서 후보 중 F1이 가장 높았고, "
            "Threshold 38%의 Validation Recall 85.7%가 Test에서 85.9%로 유사하게 재현됐습니다. "
            "다만 LightGBM의 Validation ROC-AUC·PR-AUC와 Logistic의 Test ROC-AUC·PR-AUC가 "
            "근소하게 높으므로 XGBoost가 모든 지표에서 우수하다는 뜻은 아닙니다."
        )

    st.markdown("#### 오류 분석과 모델 해석")
    error_left, importance_right = st.columns(2, gap="large")
    with error_left:
        _show_image(
            TRAINING_FIGURE_DIR / "error_group_profiles.png",
            "Test TP·TN·FP·FN 고객군의 평균 행동 비교",
        )
        st.caption(
            "FN은 비교적 활동적인 고객이 갑자기 중단한 사례, FP는 구매 주기가 긴 고객과 유사했습니다."
        )
    with importance_right:
        _show_image(
            TRAINING_FIGURE_DIR / "permutation_importance.png",
            "Permutation Importance · 예측 기여도",
        )
        st.caption(
            "중요도는 Feature를 섞었을 때 성능이 얼마나 감소하는지 보여주며, 이탈의 인과 원인을 뜻하지 않습니다."
        )

    with st.expander("확률 해석과 모델 한계"):
        _show_image(
            TRAINING_FIGURE_DIR / "calibration_curve.png",
            "Test Calibration Curve",
        )
        st.markdown(
            "- XGBoost Brier Score는 0.2007로 Logistic 0.2003과 유사하지만 별도 확률 보정은 하지 않았습니다.\n"
            "- 따라서 38%는 실제 이탈률 38%가 아니라 캠페인 대상 판정을 위한 운영 기준입니다.\n"
            "- 실제 탈퇴가 아닌 90일 무구매 대리 Target이며, 현재 평가는 무작위 층화 분할입니다.\n"
            "- 운영 전 시간순 검증, 데이터 드리프트 점검과 실제 캠페인 A/B 테스트가 필요합니다."
        )


def render() -> None:
    st.markdown("## 데이터 분석 및 모델 성능")
    st.caption("데이터 근거부터 최종 모델 선정까지 프로젝트의 분석 흐름을 확인합니다.")
    eda_tab, performance_tab = st.tabs(["데이터 현황·EDA", "모델 성능"])
    with eda_tab:
        _render_eda()
    with performance_tab:
        _render_model_performance()
