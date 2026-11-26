"""What-if water-management simulator.

IMPORTANT: the dataset has no extraction series, so the extraction response is
NOT learned from data. The effect of a cut in extraction is:

    effect(h) = reduction * EXTRACTION_SHARE_OF_TREND * max(trend_per_year, 0) * h / 12

where trend_per_year is the observed long-term deepening of the record (a
data-derived number) and EXTRACTION_SHARE_OF_TREND is an explicit assumption.
Every scenario curve is labelled as assumption-driven.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

REDUCTIONS = {"A": 0.0, "B": 0.10, "C": 0.20, "D": 0.30}
LABELS = {
    "A": "Current extraction",
    "B": "10% reduction in extraction",
    "C": "20% reduction in extraction",
    "D": "30% reduction in extraction",
}


def trend_per_year(monthly: pd.DataFrame) -> float:
    obs = monthly[~monthly["missing"]]
    years = (obs.index - obs.index[0]).days / 365.25
    slope = float(np.polyfit(years, obs["level"].to_numpy(), 1)[0])
    return slope if config.LEVEL_IS_DEPTH else -slope


def simulate(baseline: pd.Series, trend: float, share: float = config.EXTRACTION_SHARE_OF_TREND) -> pd.DataFrame:
    """baseline: forecast Series indexed by month. Returns one column per scenario."""
    h = np.arange(1, len(baseline) + 1)
    out = {}
    for key, red in REDUCTIONS.items():
        effect = red * share * max(trend, 0) * h / 12
        adjusted = baseline.to_numpy() - effect if config.LEVEL_IS_DEPTH else baseline.to_numpy() + effect
        out[key] = adjusted
    frame = pd.DataFrame(out, index=baseline.index)
    frame.columns = [f"{k}: {LABELS[k]}" for k in frame.columns]
    return frame
