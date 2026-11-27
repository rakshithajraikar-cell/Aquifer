"""Evaluation protocol shared by every model.

Two kinds of error are measured on the held-out data:
  * one-step-ahead: each month predicted from the real history before it.
    This is the usual leaderboard number, but it is optimistic because real
    previous values are always available.
  * multi-horizon: from each forecast origin, predict 1..12 months ahead using
    only the model's own forecasts. This is what a deployed forecaster faces.
Model selection uses multi-horizon error on the validation block. The test
block is only used to report the final result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def regression_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    err = y - yhat
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    nonzero = np.abs(y) > 1e-9
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_pct": float(np.mean(np.abs(err[nonzero]) / np.abs(y[nonzero])) * 100),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def one_step_predictions(model, series: pd.Series, positions) -> pd.DataFrame:
    rows = []
    for t in positions:
        pred = float(model.forecast(series.iloc[:t], 1)[0])
        rows.append({"month": series.index[t], "actual": float(series.iloc[t]),
                     "predicted": pred, "horizon": 1})
    return pd.DataFrame(rows)


def multi_horizon_errors(model, series: pd.Series, origins, horizon: int = config.MAX_HORIZON) -> pd.DataFrame:
    rows = []
    n = len(series)
    for t in origins:
        if t + horizon > n:
            break
        preds = model.forecast(series.iloc[:t], horizon)
        for h in range(horizon):
            rows.append({
                "origin": series.index[t - 1],
                "month": series.index[t + h],
                "horizon": h + 1,
                "actual": float(series.iloc[t + h]),
                "predicted": float(preds[h]),
            })
    return pd.DataFrame(rows)


def horizon_table(errors: pd.DataFrame) -> pd.DataFrame:
    def _agg(g):
        m = regression_metrics(g["actual"], g["predicted"])
        return pd.Series({"MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE_pct": m["MAPE_pct"], "n": len(g)})
    return errors.groupby("horizon").apply(_agg, include_groups=False)
