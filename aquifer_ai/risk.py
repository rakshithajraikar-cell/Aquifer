"""Groundwater Risk Index (0-100), early-warning alerts and rule-based recommendations.

Every component is computed from the observed history or from the forecast.
Components that need rainfall or extraction data are NOT computed; their weights
are dropped and the rest are renormalised. The output lists them as unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

RISK_CLASSES = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
UNAVAILABLE_COMPONENTS = {
    "rainfall_deficit": "No rainfall series in the dataset",
    "temperature": "No temperature series in the dataset",
    "extraction_rate": "No extraction series in the dataset",
}


def risk_class(score: float) -> str:
    lo, mid, hi = config.RISK_CLASS_EDGES
    if score < lo:
        return RISK_CLASSES[0]
    if score < mid:
        return RISK_CLASSES[1]
    if score < hi:
        return RISK_CLASSES[2]
    return RISK_CLASSES[3]


def _score_percentile_same_month(value: float, month: int, hist: pd.Series) -> float:
    same = hist[hist.index.month == month]
    return float((same <= value).mean() * 100)


class RiskContext:
    """Statistics learned from the observed history, reused for every month scored."""

    def __init__(self, observed: pd.Series, recharge_ratio: float | None):
        self.observed = observed
        self.recharge_ratio = recharge_ratio
        change12 = observed.diff(12).dropna()
        positive = change12[change12 > 0]
        self.decline_scale = float(np.quantile(positive, 0.95)) if len(positive) else 1.0
        self.decline_scale = self.decline_scale or 1.0
        self.month_mean = observed.groupby(observed.index.month).mean()
        self.month_std = observed.groupby(observed.index.month).std(ddof=0).replace(0, np.nan)
        self.critical_depth = float(np.percentile(observed, config.CRITICAL_PERCENTILE))

    def components(self, series_all: pd.Series, t: pd.Timestamp) -> dict:
        value = float(series_all[t])
        m = t.month
        comps = {}
        comps["level"] = _score_percentile_same_month(value, m, self.observed)
        prev_t = t - pd.DateOffset(months=12)
        if prev_t in series_all.index and not np.isnan(series_all[prev_t]):
            change = value - float(series_all[prev_t])
            comps["decline"] = float(np.clip(change / self.decline_scale, 0, 1) * 100) if change > 0 else 0.0
        z = (value - self.month_mean.get(m, value)) / (self.month_std.get(m, np.nan) or np.nan)
        if not np.isnan(z):
            comps["seasonal"] = float(np.clip((z + 2) / 4, 0, 1) * 100)
        if self.recharge_ratio is not None:
            comps["recharge"] = float((1 - np.clip(self.recharge_ratio, 0, 1)) * 100)
        return comps

    def score(self, series_all: pd.Series, t: pd.Timestamp) -> dict:
        comps = self.components(series_all, t)
        used = {k: w for k, w in config.RISK_WEIGHTS.items() if k in comps}
        total_w = sum(used.values())
        value = sum(comps[k] * w for k, w in used.items()) / total_w
        return {
            "month": t.strftime("%Y-%m"),
            "level": float(series_all[t]),
            "risk_index": round(float(value), 1),
            "risk_class": risk_class(value),
            "components": {k: round(v, 1) for k, v in comps.items()},
            "weights_used": {k: round(w / total_w, 3) for k, w in used.items()},
        }


def score_timeline(ctx: RiskContext, series_all: pd.Series, months) -> list[dict]:
    return [ctx.score(series_all, t) for t in months]


def early_warnings(ctx: RiskContext, combined: pd.Series, forecast_months, anomalies: pd.DataFrame,
                   horizon_label: str) -> list[dict]:
    alerts = []
    fc = combined.loc[forecast_months]
    last_obs = ctx.observed.index.max()
    above = fc[fc >= ctx.critical_depth] if config.LEVEL_IS_DEPTH else fc[fc <= ctx.critical_depth]
    if len(above):
        first, last = above.index[0], above.index[-1]
        alerts.append({
            "type": "critical_level",
            "severity": "critical",
            "message": (f"Groundwater level may reach a critical level between "
                        f"{first:%b %Y} and {last:%b %Y} (forecast {fc[first]:.2f} m, "
                        f"critical threshold {ctx.critical_depth:.2f} m)."),
            "window_start": first.strftime("%Y-%m"),
            "window_end": last.strftime("%Y-%m"),
        })

    change12_hist = ctx.observed.diff(12).dropna()
    rapid_threshold = float(np.quantile(change12_hist[change12_hist > 0], 0.95)) if (change12_hist > 0).any() else np.inf
    comb_change = combined.diff(12)
    rapid = [t for t in forecast_months if comb_change.get(t, 0) > rapid_threshold]
    if rapid:
        alerts.append({
            "type": "rapid_decline",
            "severity": "high",
            "message": (f"Rapid decline is forecast: the 12-month change exceeds the historical "
                        f"95th percentile ({rapid_threshold:.2f} m) in {len(rapid)} forecast month(s), "
                        f"starting {rapid[0]:%b %Y}."),
            "window_start": rapid[0].strftime("%Y-%m"),
            "window_end": rapid[-1].strftime("%Y-%m"),
        })

    cur = ctx.observed.iloc[-1]
    cur_pct = _score_percentile_same_month(cur, ctx.observed.index[-1].month, ctx.observed)
    recent_change = float(ctx.observed.iloc[-1] - ctx.observed.iloc[-13]) if len(ctx.observed) > 13 else 0.0
    if cur_pct >= config.CRITICAL_PERCENTILE and (recent_change > 0) == config.LEVEL_IS_DEPTH:
        alerts.append({
            "type": "possible_drought_stress",
            "severity": "high",
            "message": (f"Possible drought stress: the latest level is at the {cur_pct:.0f}th percentile "
                        f"for its calendar month and has moved toward lower water availability "
                        f"over the last 12 months."),
            "window_start": last_obs.strftime("%Y-%m"),
            "window_end": last_obs.strftime("%Y-%m"),
        })

    recent = anomalies[anomalies["month"] > last_obs - pd.DateOffset(months=12)] if len(anomalies) else anomalies
    if len(recent) >= 2:
        alerts.append({
            "type": "unusual_fluctuations",
            "severity": "medium",
            "message": f"{len(recent)} unusual month(s) flagged in the last 12 months; check the sensor and data quality.",
            "window_start": str(recent["month"].min().date())[:7],
            "window_end": str(recent["month"].max().date())[:7],
        })

    for key, reason in UNAVAILABLE_COMPONENTS.items():
        if key in ("rainfall_deficit",):
            alerts.append({"type": "low_predicted_rainfall", "severity": "not_evaluated",
                           "message": f"Not evaluated: {reason}.", "window_start": None, "window_end": None})
        if key == "extraction_rate":
            alerts.append({"type": "excessive_extraction", "severity": "not_evaluated",
                           "message": f"Not evaluated: {reason}.", "window_start": None, "window_end": None})
    return alerts


def recommendations(current: dict, forecast_12m_change: float | None, recharge_ratio: float | None,
                    scenario_gain_m: float | None, alerts: list[dict], n_repeated: int) -> list[dict]:
    """Rules that read the model outputs. Each recommendation quotes the figure it depends on."""
    recs = []
    cls = current["risk_class"]
    if cls in ("High Risk", "Critical Risk"):
        recs.append({
            "category": "extraction",
            "action": "Reduce groundwater extraction, starting with the 20% step in the what-if simulator.",
            "evidence": f"Current Groundwater Risk Index is {current['risk_index']} ({cls}).",
        })
    if forecast_12m_change is not None and forecast_12m_change > 0:
        recs.append({
            "category": "artificial_recharge",
            "action": ("Plan artificial recharge (percolation ponds, recharge pits or check dams) "
                       "to be in place before the next monsoon."),
            "evidence": f"Best model forecasts the level {forecast_12m_change:+.2f} m deeper over 12 months.",
        })
    if recharge_ratio is not None and recharge_ratio < 1:
        recs.append({
            "category": "rainwater_harvesting",
            "action": "Prioritise rooftop rainwater harvesting and recharge wells in the catchment.",
            "evidence": (f"The last completed year's seasonal recovery was {recharge_ratio:.2f}x the "
                         f"historical median, so natural recharge was weaker than usual."),
        })
    if scenario_gain_m is not None and scenario_gain_m > 0:
        recs.append({
            "category": "conservation",
            "action": "Set a water-conservation target and track progress against the 10–30% reduction scenarios.",
            "evidence": f"Under the assumed extraction response, a 20% cut lowers the 12-month level by about {scenario_gain_m:.2f} m.",
        })
    if any(a["type"] == "critical_level" for a in alerts):
        recs.append({
            "category": "urgent",
            "action": "Verify the station reading and review extraction permits within the critical window.",
            "evidence": next(a["message"] for a in alerts if a["type"] == "critical_level"),
        })
    if n_repeated:
        recs.append({
            "category": "data_quality",
            "action": "Calibrate or replace the sensor and confirm the monitoring-data records with the agency.",
            "evidence": f"{n_repeated} month(s) show the same reading repeated for 6 or more months.",
        })
    recs.append({
        "category": "monitoring",
        "action": "Keep monthly measurements and re-run the pipeline each time new readings arrive.",
        "evidence": "The forecast is monthly; longer or more frequent records improve every model.",
    })
    return recs
