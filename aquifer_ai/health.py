"""Aquifer Health Score (0-100). A PROJECT-DEVELOPED INDICATOR, not an official
scientific or regulatory measurement. Higher is healthier.

Each component is scored 0-100 (100 = healthiest) from the project's own
data, then combined with the weights in config.HEALTH_WEIGHTS. Components that
need rainfall or extraction data are excluded and the other weights are renormalised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def long_term_trend_m_per_decade(monthly: pd.DataFrame) -> float:
    obs = monthly[~monthly["missing"]]
    years = (obs.index - obs.index[0]).days / 365.25
    slope_per_year = np.polyfit(years, obs["level"].to_numpy(), 1)[0]
    if not config.LEVEL_IS_DEPTH:
        slope_per_year = -slope_per_year
    return float(slope_per_year * 10)  # positive = deepening (worse)


def health_score(monthly: pd.DataFrame, current_percentile: float, recharge_ratio: float | None,
                 anomaly_rate: float, forecast_risk_mean: float | None) -> dict:
    trend = long_term_trend_m_per_decade(monthly)
    comps = {
        "current_level": 100 - current_percentile,
        "long_term_trend": float(np.clip(1 - trend / config.TREND_WORST_M_PER_DECADE, 0, 1) * 100),
        "anomaly_frequency": float(np.clip(1 - anomaly_rate / 0.10, 0, 1) * 100),
    }
    if recharge_ratio is not None:
        comps["recharge"] = float(np.clip(recharge_ratio, 0, 1) * 100)
    if forecast_risk_mean is not None:
        comps["forecast_condition"] = 100 - forecast_risk_mean

    used = {k: config.HEALTH_WEIGHTS[k] for k in comps}
    total = sum(used.values())
    score = sum(comps[k] * used[k] for k in comps) / total
    contributions = {k: round(comps[k] * used[k] / total, 1) for k in comps}
    excluded = [k for k in config.HEALTH_WEIGHTS if k not in comps] + ["extraction_pressure", "rainfall_condition"]
    return {
        "score": round(float(score), 1),
        "label": "Project-developed indicator (not an official measurement)",
        "components": {k: round(v, 1) for k, v in comps.items()},
        "weights_used": {k: round(w / total, 3) for k, w in used.items()},
        "points_contributed": contributions,
        "excluded_components": excluded,
        "trend_m_per_decade": round(trend, 3),
        "anomaly_rate_last_5y": round(anomaly_rate, 3),
    }
