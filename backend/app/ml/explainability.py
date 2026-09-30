"""
SHAP-based Model Explainability

For tree-based models (Random Forest, XGBoost), we use SHAP TreeExplainer
which is fast and exact. For linear models, LinearExplainer is used.

SHAP values quantify each feature's marginal contribution to a single
prediction — they are an EXPLANATION of model behavior, not causal claims.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from app.logging_config import get_logger

logger = get_logger(__name__)


def explain_prediction(
    model,
    X: np.ndarray,
    feature_names: list[str],
    model_type: str,
    max_features: int = 15,
) -> dict:
    """
    Compute SHAP values for the given model and input X (last row for inference).

    Returns a dict with:
      - shap_values: {feature: shap_value}
      - feature_importance: {feature: mean |shap|} over all training rows
      - top_positive_features: list of (feature, shap, feature_value)
      - top_negative_features: list of (feature, shap, feature_value)
      - explanation_text: human-readable summary
    """
    try:
        import shap

        if model_type in ("xgboost", "random_forest"):
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X)
        else:
            logger.warning("SHAP not supported for model_type", model_type=model_type)
            return _empty_explanation(feature_names)

        # Compute SHAP for the last row (the prediction row)
        sample = X[-1:] if X.ndim == 2 else X.reshape(1, -1)
        shap_vals = explainer.shap_values(sample)

        # shap_vals may be list (for classifiers) — take class 1
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_vals = np.array(shap_vals).ravel()
        feature_vals = sample.ravel()

        shap_dict = {fn: float(sv) for fn, sv in zip(feature_names, shap_vals)}
        fv_dict = {fn: float(fv) for fn, fv in zip(feature_names, feature_vals)}

        sorted_by_abs = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        top_pos = [
            {"feature": k, "shap_value": v, "feature_value": fv_dict[k], "direction": "positive"}
            for k, v in sorted_by_abs if v > 0
        ][:max_features]

        top_neg = [
            {"feature": k, "shap_value": v, "feature_value": fv_dict[k], "direction": "negative"}
            for k, v in sorted_by_abs if v < 0
        ][:max_features]

        # Feature importance: mean absolute SHAP over all rows in X
        all_shap = explainer.shap_values(X)
        if isinstance(all_shap, list):
            all_shap = all_shap[1]
        mean_abs_shap = np.abs(np.array(all_shap)).mean(axis=0)
        importance = {fn: float(v) for fn, v in zip(feature_names, mean_abs_shap)}

        # Generate a human-readable explanation
        explanation_text = _generate_text(top_pos, top_neg, model_type)

        return {
            "shap_values": shap_dict,
            "feature_importance": importance,
            "top_positive_features": top_pos,
            "top_negative_features": top_neg,
            "explanation_text": explanation_text,
        }

    except Exception as exc:
        logger.error("SHAP computation failed", error=str(exc))
        return _empty_explanation(feature_names)


def _generate_text(top_pos: list, top_neg: list, model_type: str) -> str:
    pos_names = [item["feature"] for item in top_pos[:3]]
    neg_names = [item["feature"] for item in top_neg[:3]]

    text_parts = []
    if pos_names:
        text_parts.append(
            f"The strongest factors supporting the prediction were: {', '.join(pos_names)}."
        )
    if neg_names:
        text_parts.append(
            f"Factors that pushed the prediction in the opposite direction: {', '.join(neg_names)}."
        )
    text_parts.append(
        "Note: These are model feature contributions, not causal relationships. "
        "Stock prices are influenced by many factors not captured here."
    )
    return " ".join(text_parts)


def _empty_explanation(feature_names: list[str]) -> dict:
    return {
        "shap_values": {},
        "feature_importance": {fn: 0.0 for fn in feature_names},
        "top_positive_features": [],
        "top_negative_features": [],
        "explanation_text": "Explainability is not available for this model type.",
    }
