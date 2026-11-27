"""Tests for properties the project depends on: no leakage, chronological splits,
bounded scores, and correct baseline behaviour. Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aquifer_ai import config  # noqa: E402
from aquifer_ai import models as M  # noqa: E402
from aquifer_ai.data import chronological_split  # noqa: E402
from aquifer_ai.evaluate import regression_metrics  # noqa: E402
from aquifer_ai.features import FEATURE_COLUMNS, make_features  # noqa: E402
from aquifer_ai.recovery import recovery_events  # noqa: E402
from aquifer_ai.risk import RiskContext, risk_class  # noqa: E402
from aquifer_ai.scenarios import simulate  # noqa: E402


def _series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
    trend = np.linspace(0, 3, n)
    return pd.Series(10 + seasonal + trend + rng.normal(0, 0.2, n), index=idx, name="level")


def test_features_do_not_use_current_value():
    s = _series()
    base = make_features(s)
    changed = s.copy()
    t = 80
    changed.iloc[t] += 50.0  # perturb the target at t
    after = make_features(changed)
    # Features describing month t must not change when only level[t] changes
    assert np.allclose(base.iloc[t].fillna(-1).to_numpy(), after.iloc[t].fillna(-1).to_numpy())
    assert list(base.columns) == FEATURE_COLUMNS


def test_chronological_split_is_ordered_and_disjoint():
    sp = chronological_split(588)
    assert sp["train"].stop == sp["val"].start
    assert sp["val"].stop == sp["test"].start
    assert sp["test"].stop == 588
    assert sp["train"].start == 0


def test_seasonal_naive_repeats_last_year():
    s = _series()
    f = M.SeasonalNaiveForecaster().fit(s).forecast(s, 15)
    assert np.allclose(f[:12], s.values[-12:])
    assert np.allclose(f[12:], s.values[-12:][:3])


def test_naive_forecast_is_last_value():
    s = _series()
    assert np.allclose(M.NaiveForecaster().forecast(s, 3), s.iloc[-1])


def test_tree_forecaster_is_recursive_and_finite():
    s = _series(200)
    m = M.TreeForecaster("XGBoost", M.make_xgboost(2, 0.1, 50)).fit(s.iloc[:150])
    f = m.forecast(s.iloc[:150], 12)
    assert len(f) == 12 and np.all(np.isfinite(f))


def test_risk_index_bounded_and_classes():
    s = _series(240)
    ctx = RiskContext(observed=s, recharge_ratio=0.8)
    for t in s.index[-30:]:
        r = ctx.score(s, t)
        assert 0 <= r["risk_index"] <= 100
        assert r["risk_class"] in ("Low Risk", "Moderate Risk", "High Risk", "Critical Risk")
        assert abs(sum(r["weights_used"].values()) - 1) < 1e-6


def test_risk_class_edges():
    lo, mid, hi = config.RISK_CLASS_EDGES
    assert risk_class(lo - 0.1) == "Low Risk"
    assert risk_class(mid) == "High Risk"  # a score equal to an edge belongs to the higher class
    assert risk_class(hi) == "Critical Risk"
    assert risk_class(99) == "Critical Risk"


def test_scenarios_reduce_depth_when_extraction_is_cut():
    idx = pd.date_range("2021-01-01", periods=12, freq="MS")
    base = pd.Series(np.linspace(15, 16, 12), index=idx)
    out = simulate(base, trend=0.5, share=0.5)
    vals = out.iloc[-1].to_numpy()
    # Higher cut -> lower (shallower) depth; the current-extraction scenario equals the baseline
    assert np.isclose(vals[0], base.iloc[-1])
    assert vals[0] > vals[1] > vals[2] > vals[3]


def test_recovery_events_on_synthetic_monsoon_cycle():
    idx = pd.date_range("2000-01-01", periods=36, freq="MS")
    vals = []
    for m in idx.month:
        if m in (4, 5, 6):
            vals.append(12.0)   # deep before monsoon
        elif m in (9, 10, 11, 12):
            vals.append(8.0)    # shallower after monsoon
        else:
            vals.append(10.0)
    monthly = pd.DataFrame({"level": vals, "missing": False}, index=idx)
    ev = recovery_events(monthly)
    assert len(ev) == 3
    assert np.allclose(ev["rise"], 4.0)
    assert (ev["response_months"] > 0).all()


def test_regression_metrics_known_values():
    m = regression_metrics([1, 2, 3], [1, 2, 5])
    assert np.isclose(m["MAE"], 2 / 3)
    assert np.isclose(m["RMSE"], np.sqrt(4 / 3))
