"""Seasonal recharge and recovery analysis, estimated from the observed record.

For each calendar year:
  pre   = deepest (highest-depth) reading in Apr-Jun  (end of dry season)
  post  = shallowest (lowest-depth) reading in Sep-Dec (after the monsoon)
  rise  = pre - post   (how much the level recovered in that year)
  response_months  = months from pre to post
  recession_months = months after post until the level falls back to pre

Rainfall is not in the dataset, so this measures the groundwater response
itself, not its rainfall driver.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _depth_from_level(level: pd.Series) -> pd.Series:
    """Return a series where larger means more water in the aquifer."""
    return -level if config.LEVEL_IS_DEPTH else level


def recovery_events(monthly: pd.DataFrame) -> pd.DataFrame:
    level = monthly["level"]
    rows = []
    for year in sorted(set(level.index.year)):
        dry = level[(level.index.year == year) & level.index.month.isin([4, 5, 6])]
        wet = level[(level.index.year == year) & level.index.month.isin([9, 10, 11, 12])]
        if dry.empty or wet.empty:
            continue
        if config.LEVEL_IS_DEPTH:
            pre_t, post_t = dry.idxmax(), wet.idxmin()   # deepest before, shallowest after
        else:
            pre_t, post_t = dry.idxmin(), wet.idxmax()
        pre_v, post_v = float(level[pre_t]), float(level[post_t])
        rise = (pre_v - post_v) if config.LEVEL_IS_DEPTH else (post_v - pre_v)

        # Months until the level falls back to its pre-monsoon state
        recession = np.nan
        after = level[post_t:]
        if config.LEVEL_IS_DEPTH:
            back = after[after >= pre_v - 1e-9]
        else:
            back = after[after <= pre_v + 1e-9]
        if len(back) > 0:
            recession = (back.index[0].year - post_t.year) * 12 + (back.index[0].month - post_t.month)

        response = (post_t.year - pre_t.year) * 12 + (post_t.month - pre_t.month)
        rows.append({
            "year": year,
            "pre_month": pre_t.strftime("%Y-%m"),
            "pre_level": pre_v,
            "post_month": post_t.strftime("%Y-%m"),
            "post_level": post_v,
            "rise": float(rise),
            "response_months": int(response),
            "rise_rate_per_month": float(rise / response) if response > 0 else np.nan,
            "recession_months": recession,
        })
    return pd.DataFrame(rows)


def recovery_summary(events: pd.DataFrame) -> dict:
    ev = events.dropna(subset=["rise"])
    if ev.empty:
        return {}
    last = ev.iloc[-1]
    rises = ev[ev["rise"] > 0]["rise"]
    return {
        "years_analysed": int(len(ev)),
        "median_rise_m": float(ev["rise"].median()),
        "median_response_months": float(ev["response_months"].median()),
        "median_rise_rate_m_per_month": float(ev["rise_rate_per_month"].median()),
        "median_recession_months": float(ev["recession_months"].median()) if ev["recession_months"].notna().any() else None,
        "recession_months_iqr": [float(ev["recession_months"].quantile(0.25)), float(ev["recession_months"].quantile(0.75))]
        if ev["recession_months"].notna().any() else None,
        "share_of_years_with_recovery": float((ev["rise"] > 0).mean()),
        "latest_year": int(last["year"]),
        "latest_rise_m": float(last["rise"]),
        "latest_recovery_ratio": None,  # filled in by risk.py using the median rise
        "median_positive_rise_m": float(rises.median()) if len(rises) else 0.0,
    }


def latest_recharge_ratio(events: pd.DataFrame) -> float | None:
    """Last completed year's rise divided by the historical median rise.
    1.0 means the typical seasonal recovery; below 1 means a weaker recovery."""
    ev = events.dropna(subset=["rise"])
    if len(ev) < 3:
        return None
    median_rise = ev["rise"].median()
    if median_rise <= 0:
        return None
    return float(ev.iloc[-1]["rise"] / median_rise)
