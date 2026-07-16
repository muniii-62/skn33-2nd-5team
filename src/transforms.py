"""전처리 파이프라인에서 사용하는 변환 함수들.
별도 모듈로 분리한 이유: pickle로 저장된 Pipeline을 다른 스크립트/노트북에서
불러올 때, 함수가 __main__ 모듈에 있으면 참조를 못 찾는 문제가 생기기 때문."""

import numpy as np


def clip_negative(x):
    """음수를 0으로 클리핑 (net_revenue 예외 처리, 07 EDA 근거)"""
    return np.clip(x, 0, None)


def log1p_transform(x):
    return np.log1p(x)