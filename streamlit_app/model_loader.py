import joblib
import streamlit as st
from config import MODEL_PATH, PREPROCESSOR_PATH, FEATURE_ORDER


def _get_transformer_column_order(preprocessor):
    """학습된 ColumnTransformer의 선언 순서대로 출력 피처명을 반환한다."""
    order = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if not isinstance(columns, (list, tuple)):
            raise RuntimeError(
                f"전처리 단계 '{name}'의 컬럼 정의를 해석할 수 없습니다: {columns!r}"
            )
        order.extend(columns)
    return order


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # 전처리기 실제 출력 순서와 FEATURE_ORDER가 어긋나면, 에러 없이 조용히
    # 잘못된 예측이 나갈 수 있으므로 앱 시작 시점에 강제로 검증한다.
    # 기존 저장 모델의 FunctionTransformer는 get_feature_names_out()을 지원하지
    # 않을 수 있어, 학습된 ColumnTransformer의 컬럼 선언을 직접 확인한다.
    actual_order = _get_transformer_column_order(preprocessor)
    if actual_order != FEATURE_ORDER:
        raise RuntimeError(
            "컬럼 순서 불일치! preprocessor 실제 출력 순서와 config.FEATURE_ORDER가 다릅니다.\n"
            f"  실제:   {actual_order}\n"
            f"  설정값: {FEATURE_ORDER}\n"
            "전처리 파이프라인이 바뀌었다면 config.py의 FEATURE_ORDER를 이 순서로 갱신하세요."
        )

    return model, preprocessor
