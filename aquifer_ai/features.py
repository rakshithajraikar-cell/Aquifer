"""Feature engineering for supervised forecasting.

Every feature for month t is computed from values up to t-1 only, so the
target (level at t) never leaks into its own inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LAGS

LAG_COLUMNS = [f"lag_{l}" for l in LAGS]
ROLLING_COLUMNS = ["roll_mean_3", "roll_mean_6", "roll_std_6", "diff_1", "diff_12"]
CALENDAR_COLUMNS = ["month_sin", "month_cos"]
FEATURE_COLUMNS = LAG_COLUMNS + ROLLING_COLUMNS + CALENDAR_COLUMNS

# Feature groups used by the ablation study (see run_pipeline.py)
FEATURE_GROUPS = {
    "History only (lagged levels)": LAG_COLUMNS,
    "History + rolling & change features": LAG_COLUMNS + ROLLING_COLUMNS,
    "History + rolling + seasonal calendar": FEATURE_COLUMNS,
}


def make_features(level: pd.Series) -> pd.DataFrame:
    """Build the feature matrix. Row t describes the prediction target at t."""
    past = level.shift(1)
    X = pd.DataFrame(index=level.index)
    for lag in LAGS:
        X[f"lag_{lag}"] = level.shift(lag)
    X["roll_mean_3"] = past.rolling(3).mean()
    X["roll_mean_6"] = past.rolling(6).mean()
    X["roll_std_6"] = past.rolling(6).std()
    X["diff_1"] = level.shift(1) - level.shift(2)
    X["diff_12"] = level.shift(1) - level.shift(13)
    month = level.index.month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    return X[FEATURE_COLUMNS]
