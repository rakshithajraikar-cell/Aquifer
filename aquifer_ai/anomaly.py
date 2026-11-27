"""Groundwater anomaly detection: sudden jumps, unusual seasonal residuals,
repeated (flat) readings, and an Isolation Forest over the same signals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config

Z_THRESHOLD = 3.5       # robust z-score (modified z, 0.6745 * (x - median) / MAD)
FLAT_RUN_MIN = 6        # identical consecutive monthly values -> suspicious
CONTAMINATION = 0.03    # ASSUMPTION: expected share of unusual months for Isolation Forest


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=x.index)
    return 0.6745 * (x - med) / mad


def detect_anomalies(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return one row per flagged month with the methods that flagged it."""
    level = monthly["level"]
    observed = ~monthly["missing"]
    diff = level.diff()
    seasonal_resid = level - level.groupby(level.index.month).transform("median")

    flags: dict[pd.Timestamp, dict] = {}

    def add(ts, method, text):
        entry = flags.setdefault(ts, {"methods": [], "reasons": []})
        entry["methods"].append(method)
        entry["reasons"].append(text)

    # 1. Sudden jumps in month-to-month change
    z_diff = _robust_z(diff[observed])
    for ts, z in z_diff.items():
        if abs(z) > Z_THRESHOLD:
            direction = "jump up" if z > 0 else "jump down"
            add(ts, "statistical_jump",
                f"Sudden {direction} of {diff[ts]:+.2f} in one month (robust z = {z:.1f})")

    # 2. Unusual departure from the usual level for that calendar month
    z_resid = _robust_z(seasonal_resid[observed])
    for ts, z in z_resid.items():
        if abs(z) > Z_THRESHOLD:
            add(ts, "seasonal_residual",
                f"Level is unusual for this calendar month (robust z = {z:.1f})")

    # 3. Repeated identical readings (possible sensor or data-entry problem)
    run_id = (level != level.shift()).cumsum()
    run_len = level.groupby(run_id).transform("size")
    for ts in level[(run_len >= FLAT_RUN_MIN) & observed].index:
        add(ts, "repeated_value",
            f"Same reading ({level[ts]:.2f}) repeated for {int(run_len[ts])} or more consecutive months")

    # 4. Isolation Forest on level, change and seasonal residual
    feats = pd.DataFrame({"level": level, "diff": diff, "resid": seasonal_resid}).dropna()
    feats = feats[observed.reindex(feats.index)]
    if len(feats) > 20:
        iso = IsolationForest(contamination=CONTAMINATION, random_state=config.SEED)
        labels = iso.fit_predict(feats.to_numpy())
        for ts in feats.index[labels == -1]:
            add(ts, "isolation_forest", "Unusual combination of level, monthly change and seasonal residual")

    rows = []
    for ts, e in sorted(flags.items()):
        rows.append({
            "month": ts,
            "level": float(level[ts]),
            "methods": ";".join(sorted(set(e["methods"]))),
            "reason": " | ".join(e["reasons"]),
            "is_repeated_value": "repeated_value" in e["methods"],
        })
    return pd.DataFrame(rows, columns=["month", "level", "methods", "reason", "is_repeated_value"])


def anomaly_rate(anomalies: pd.DataFrame, monthly: pd.DataFrame, years: int = 5) -> float:
    """Share of observed months in the last `years` years that carry any flag."""
    cutoff = monthly.index.max() - pd.DateOffset(years=years)
    window = monthly[(monthly.index > cutoff) & (~monthly["missing"])]
    if len(window) == 0:
        return 0.0
    flagged = anomalies[anomalies["month"] > cutoff]["month"].nunique()
    return float(flagged / len(window))
