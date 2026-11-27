"""Explainable AI: SHAP values, permutation importance and partial dependence
for the XGBoost forecaster. Explanations are phrased in terms of the features
the model actually uses (past levels, changes and calendar position). Rainfall
and extraction are not in the data, so they cannot appear as explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence, permutation_importance

from . import config
from .features import make_features

PLAIN_NAMES = {
    "lag_1": "level one month ago",
    "lag_2": "level two months ago",
    "lag_3": "level three months ago",
    "lag_6": "level six months ago",
    "lag_12": "level twelve months ago (same month last year)",
    "roll_mean_3": "average level over the previous 3 months",
    "roll_mean_6": "average level over the previous 6 months",
    "roll_std_6": "recent variability (6-month std)",
    "diff_1": "last month's change",
    "diff_12": "change over the last 12 months",
    "month_sin": "seasonal timing (sine)",
    "month_cos": "seasonal timing (cosine)",
}


def explain_model(tree_forecaster, monthly_levels: pd.Series, test_start: pd.Timestamp,
                  eval_forecaster=None) -> dict:
    """Explain `tree_forecaster` (fitted on all data, used for the forecast).

    Permutation importance is measured with `eval_forecaster` (the model fitted
    WITHOUT the test block), so the test period never informs its own explanation.
    """
    cols = tree_forecaster.feature_columns
    X_all = make_features(monthly_levels)[cols]
    mask = X_all.notna().all(axis=1)
    X = X_all[mask]
    est = tree_forecaster.model_
    eval_est = (eval_forecaster or tree_forecaster).model_

    X_arr = X.to_numpy()
    explainer = shap.TreeExplainer(est)
    sv = explainer.shap_values(X_arr)
    shap_df = pd.DataFrame(sv, index=X.index, columns=cols)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)

    # Explanation for the most recent observed month
    latest = X.index[-1]
    contrib = shap_df.loc[latest].sort_values(key=np.abs, ascending=False)
    top = contrib.head(3)
    pieces = []
    for feat, val in top.items():
        pieces.append(f"{PLAIN_NAMES.get(feat, feat)} ({val:+.2f} m)")
    base = float(explainer.expected_value) if np.ndim(explainer.expected_value) == 0 else float(explainer.expected_value[0])
    predicted = float(est.predict(X_arr[-1:])[0])
    sentence = (f"For {latest:%B %Y}, the model predicts {predicted:.2f} m against an average of "
                f"{base:.2f} m. The largest contributions are: " + "; ".join(pieces) + ".")

    # Permutation importance on the held-out test period
    test_mask = X.index >= test_start
    y_test = monthly_levels.loc[X.index].to_numpy()[test_mask]
    perm = permutation_importance(eval_est, X_arr[test_mask], y_test,
                                  n_repeats=5, random_state=config.SEED, n_jobs=1)
    perm_df = pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False)

    # Partial dependence for the three most important features
    pd_out = {}
    for feat in mean_abs.index[:3]:
        res = partial_dependence(est, X_arr, features=[cols.index(feat)], grid_resolution=30)
        pd_out[feat] = {"grid": res["grid_values"][0].tolist(),
                        "average_prediction": res["average"][0].tolist()}

    return {
        "model": tree_forecaster.name,
        "explained_month": f"{latest:%Y-%m}",
        "explanation": sentence,
        "latest_contributions": {k: round(float(v), 3) for k, v in contrib.items()},
        "mean_abs_shap": {k: round(float(v), 4) for k, v in mean_abs.items()},
        "permutation_importance_test": {k: round(float(v), 4) for k, v in perm_df.items()},
        "partial_dependence": pd_out,
        "shap_frame": shap_df,
    }
