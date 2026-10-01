import numpy as np

from app.ml.dropout_constants import FEATURE_COLUMNS


def explain_prediction(explainer: object, matrix: np.ndarray) -> dict[str, float]:
    raw_values = explainer.shap_values(matrix)
    values = np.asarray(raw_values)
    if values.ndim == 2:
        values = values[0]
    if values.ndim != 1 or values.shape[0] != len(FEATURE_COLUMNS):
        raise ValueError("Invalid SHAP values shape")

    return {
        feature: round(float(value), 4)
        for feature, value in zip(FEATURE_COLUMNS, values, strict=True)
    }


def top_feature_from_shap(shap_values: dict[str, float]) -> str:
    positive_values = {
        feature: value for feature, value in shap_values.items() if value > 0
    }
    if positive_values:
        return max(positive_values.items(), key=lambda item: item[1])[0]

    return max(shap_values.items(), key=lambda item: abs(item[1]))[0]
