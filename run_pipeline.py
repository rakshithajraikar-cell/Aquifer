"""End-to-end pipeline: data -> features -> model comparison -> best model ->
forecast -> risk -> early warning -> anomalies -> recovery -> health score ->
explainability -> what-if scenarios -> recommendations -> reports and figures.

Usage:  python run_pipeline.py            (about 5-10 minutes on 2 CPU cores)

Every number the dashboard shows is written to reports/ by this script.
Nothing is hard-coded or randomly generated.
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aquifer_ai import config  # noqa: E402
from aquifer_ai import models as M  # noqa: E402
from aquifer_ai.anomaly import anomaly_rate, detect_anomalies  # noqa: E402
from aquifer_ai.data import chronological_split, load_station_monthly  # noqa: E402
from aquifer_ai.evaluate import (horizon_table, multi_horizon_errors,  # noqa: E402
                                 one_step_predictions, regression_metrics)
from aquifer_ai.explain import explain_model  # noqa: E402
from aquifer_ai.features import FEATURE_COLUMNS, FEATURE_GROUPS  # noqa: E402
from aquifer_ai.health import health_score  # noqa: E402
from aquifer_ai.recovery import latest_recharge_ratio, recovery_events, recovery_summary  # noqa: E402
from aquifer_ai.risk import (RiskContext, early_warnings, recommendations,  # noqa: E402
                             score_timeline)
from aquifer_ai.scenarios import LABELS, simulate, trend_per_year  # noqa: E402
from aquifer_ai.stations import station_overview  # noqa: E402
from aquifer_ai.risk import _score_percentile_same_month  # noqa: E402

warnings.filterwarnings("ignore")
H = config.MAX_HORIZON


def log(msg):
    print(msg, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict(orient="records"))
    return o


# ------------------------------------------------------------------ candidates
def candidate_groups():
    rf = {}
    for depth in (None, 6, 10):
        for leaf in (1, 3, 5):
            rf[f"RF max_depth={depth}, min_leaf={leaf}"] = (
                lambda d=depth, l=leaf: M.TreeForecaster("Random Forest", M.make_random_forest(d, l)))
    xgb = {}
    for depth in (2, 3, 4):
        for lr in (0.03, 0.1):
            for n in (200, 400):
                xgb[f"XGB max_depth={depth}, lr={lr}, n={n}"] = (
                    lambda d=depth, r=lr, k=n: M.TreeForecaster("XGBoost", M.make_xgboost(d, r, k)))
    return {
        "Naive": {"Naive": M.NaiveForecaster},
        "Seasonal naive": {"Seasonal naive (t-12)": M.SeasonalNaiveForecaster},
        "ARIMA": {"ARIMA AIC-selected": M.ArimaForecaster},
        "SARIMA": {"SARIMA(1,1,1)(1,0,1)[12]": M.SarimaForecaster},
        "Random Forest": rf,
        "XGBoost": xgb,
        "LSTM": {"LSTM": lambda: M.RecurrentForecaster("lstm")},
        "GRU": {"GRU": lambda: M.RecurrentForecaster("gru")},
    }


def clone_for_full_fit(chosen_entry, group):
    """New, unfitted model with the selected configuration (keeps RNN epoch count)."""
    _, _, val_model, factory = chosen_entry
    fresh = factory()
    if getattr(val_model, "best_epochs_", None):
        fresh.best_epochs_ = val_model.best_epochs_
    return fresh


def main():
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.DATA_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------- 1. data
    log("1. Loading and cleaning station data")
    monthly, meta = load_station_monthly()
    monthly.to_csv(config.DATA_DIR / f"clean_station_{meta['station']}_monthly.csv")
    s = monthly["level"].asfreq("MS")
    n = len(s)
    sp = chronological_split(n)
    val_start, val_stop = sp["val"].start, sp["val"].stop
    test_start = sp["test"].start
    dates = {k: (s.index[v.start].strftime("%Y-%m"), s.index[v.stop - 1].strftime("%Y-%m"))
             for k, v in sp.items()}
    log(f"   {n} months, {dates['train'][0]}..{dates['test'][1]}; splits {dates}")

    anomalies = detect_anomalies(monthly)
    flat_runs = int(anomalies["is_repeated_value"].sum())

    # ---------------------------------------------- 2. model selection (validation)
    log("2. Tuning and selecting models on the validation block (test block untouched)")
    groups = candidate_groups()
    val_rows, chosen = [], {}
    for group, configs in groups.items():
        best = None
        for cfg_name, factory in configs.items():
            model = factory()
            t0 = time.time()
            model.fit(s.iloc[sp["train"]])
            fit_t = time.time() - t0
            err = multi_horizon_errors(model, s.iloc[:val_stop], range(val_start, val_stop))
            m = regression_metrics(err["actual"], err["predicted"])
            row = {"group": group, "config": cfg_name, "family": model.family,
                   "val_MAE": m["MAE"], "val_RMSE": m["RMSE"], "val_fit_time_s": fit_t,
               }
            val_rows.append(row)
            log(f"   {cfg_name:<42} val MAE {m['MAE']:.3f}  ({fit_t:.1f}s)")
            if best is None or m["MAE"] < best[0]:
                best = (m["MAE"], cfg_name, model, factory)
        chosen[group] = best  # (val MAE, config name, validation-fitted model, factory)
    val_table = pd.DataFrame(val_rows)
    val_table.to_csv(config.REPORTS_DIR / "validation_tuning.csv", index=False)

    # ---------------------------------------------- 3. final test evaluation
    log("3. Refitting each chosen model on train+validation and scoring the test block")
    trainval = s.iloc[:val_stop]
    test_positions = range(test_start, n)
    comparison, test_errors, final_models = [], {}, {}
    for group, (val_mae, cfg_name, val_model, factory) in chosen.items():
        model = val_model  # RNNs keep best_epochs_ from validation for the refit
        t0 = time.time()
        model.fit(trainval)
        train_t = time.time() - t0
        one = one_step_predictions(model, s, test_positions)
        one_m = regression_metrics(one["actual"], one["predicted"])
        err = multi_horizon_errors(model, s, range(test_start, n))
        multi_m = regression_metrics(err["actual"], err["predicted"])
        ht = horizon_table(err)
        test_errors[group] = err
        final_models[group] = model
        comparison.append({
            "Model": group,
            "Selected configuration": cfg_name,
            "Family": model.family,
            "Val MAE (multi-horizon)": round(val_mae, 4),
            "Test MAE (1-step)": round(one_m["MAE"], 4),
            "Test RMSE (1-step)": round(one_m["RMSE"], 4),
            "Test MAPE % (1-step)": round(one_m["MAPE_pct"], 3),
            "Test R2 (1-step)": round(one_m["R2"], 4),
            "Test MAE (multi-horizon)": round(multi_m["MAE"], 4),
            "Test RMSE (multi-horizon)": round(multi_m["RMSE"], 4),
            "Test MAPE % (multi-horizon)": round(multi_m["MAPE_pct"], 3),
            "Test R2 (multi-horizon)": round(multi_m["R2"], 4),
            "MAE h=1": round(float(ht.loc[1, "MAE"]), 4),
            "MAE h=3": round(float(ht.loc[3, "MAE"]), 4),
            "MAE h=6": round(float(ht.loc[6, "MAE"]), 4),
            "MAE h=12": round(float(ht.loc[12, "MAE"]), 4),
            "Training time (s)": round(train_t, 2),
        })
        log(f"   {group:<15} test MAE(1-step) {one_m['MAE']:.3f}  multi-horizon {multi_m['MAE']:.3f}")

    comp = pd.DataFrame(comparison).sort_values("Val MAE (multi-horizon)").reset_index(drop=True)
    comp.insert(0, "Rank (validation)", range(1, len(comp) + 1))
    comp.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    best_group = comp.iloc[0]["Model"]
    best_model = final_models[best_group]
    test_rank_by_multi = comp.sort_values("Test MAE (multi-horizon)")["Model"].tolist()
    log(f"   BEST (lowest validation multi-horizon MAE): {best_group}")

    # ---------------------------------------------- 4. ablation (XGBoost)
    log("4. Ablation study (XGBoost, feature groups)")
    xgb_factory = chosen["XGBoost"][3]
    ablation = []
    for label, cols in FEATURE_GROUPS.items():
        m_val = M.TreeForecaster("XGBoost", xgb_factory().estimator, feature_columns=cols)
        m_val.fit(s.iloc[sp["train"]])
        err_v = multi_horizon_errors(m_val, s.iloc[:val_stop], range(val_start, val_stop))
        m_test = M.TreeForecaster("XGBoost", xgb_factory().estimator, feature_columns=cols).fit(trainval)
        err_t = multi_horizon_errors(m_test, s, range(test_start, n))
        mt = regression_metrics(err_t["actual"], err_t["predicted"])
        mv = regression_metrics(err_v["actual"], err_v["predicted"])
        ablation.append({"Feature set": label, "Features": len(cols),
                         "Val MAE": round(mv["MAE"], 4), "Test MAE (multi-horizon)": round(mt["MAE"], 4),
                         "Test RMSE (multi-horizon)": round(mt["RMSE"], 4),
                         "Test R2 (multi-horizon)": round(mt["R2"], 4)})
    ablation.append({"Feature set": "+ rainfall, temperature, extraction", "Features": 0,
                     "Val MAE": None, "Test MAE (multi-horizon)": None, "Test RMSE (multi-horizon)": None,
                     "Test R2 (multi-horizon)": None,
                     "Note": "Not run: these series are not in the dataset"})
    ablation_df = pd.DataFrame(ablation)
    ablation_df.to_csv(config.REPORTS_DIR / "ablation_study.csv", index=False)

    # ---------------------------------------------- 5. future forecast (all data)
    log("5. Forecasting the next 12 months from all observed data")
    last = s.index[-1]
    future_idx = pd.date_range(last + pd.offsets.MonthBegin(1), periods=H, freq="MS")
    all_forecasts = {}
    full_models = {}
    for group in chosen:
        mdl = clone_for_full_fit(chosen[group], group)
        mdl.fit(s)
        full_models[group] = mdl
        all_forecasts[group] = mdl.forecast(s, H)
    best_fc = pd.Series(all_forecasts[best_group], index=future_idx)

    best_err = test_errors[best_group]
    half_width = best_err.assign(abs_err=(best_err.actual - best_err.predicted).abs()).groupby("horizon")["abs_err"].quantile(0.8)
    band = np.array([float(half_width.loc[h]) for h in range(1, H + 1)])
    fc_table = pd.DataFrame({
        "month": future_idx.strftime("%Y-%m"),
        "horizon_months": range(1, H + 1),
        "forecast_level": best_fc.to_numpy(),
        "band_low": best_fc.to_numpy() - band,
        "band_high": best_fc.to_numpy() + band,
    })
    for group in all_forecasts:
        fc_table[f"forecast: {group}"] = all_forecasts[group]
    fc_table.to_csv(config.REPORTS_DIR / "forecasts.csv", index=False)
    joblib.dump(full_models[best_group], config.MODELS_DIR / "best_model.joblib")

    # ---------------------------------------------- 6. recovery, risk, alerts
    log("6. Recovery, risk index, early warnings")
    events = recovery_events(monthly)
    events.to_csv(config.REPORTS_DIR / "recovery_events.csv", index=False)
    rec = recovery_summary(events)
    ratio = latest_recharge_ratio(events)
    rec["latest_recovery_ratio"] = ratio
    recent = events[events["year"] >= events["year"].max() - 4]["rise"]
    earlier = events[events["year"] < events["year"].max() - 4]["rise"]
    regime = {
        "median_seasonal_rise_last_5_years_m": float(recent.median()),
        "median_seasonal_rise_earlier_m": float(earlier.median()),
        "note": ("Recent seasonal swings are much larger than the earlier record. Confirm the sensor, "
                 "station and unit have not changed before using recent values for decisions.")
        if recent.median() > 3 * max(earlier.median(), 0.1) else "No marked change in seasonal swing.",
    }

    ctx = RiskContext(observed=s, recharge_ratio=ratio)
    combined = pd.concat([s, best_fc])
    hist_timeline = score_timeline(ctx, combined, s.index[-120:])  # last 10 years of history
    fc_timeline = score_timeline(ctx, combined, future_idx)
    timeline_df = pd.DataFrame(hist_timeline + fc_timeline)
    timeline_df["kind"] = ["observed"] * len(hist_timeline) + ["forecast"] * len(fc_timeline)
    timeline_df.to_csv(config.REPORTS_DIR / "risk_timeline.csv", index=False)

    current = ctx.score(combined, last)
    fc_risk = pd.DataFrame(fc_timeline)
    alerts = early_warnings(ctx, combined, future_idx, anomalies, "12 months")
    forecast_change_12 = float(best_fc.iloc[-1] - s.iloc[-1])

    # ---------------------------------------------- 7. anomalies, health
    anomalies.to_csv(config.REPORTS_DIR / "anomalies.csv", index=False)
    a_rate = anomaly_rate(anomalies, monthly, years=5)
    cur_pct = _score_percentile_same_month(float(s.iloc[-1]), last.month, s)
    health = health_score(monthly, cur_pct, ratio, a_rate,
                          float(fc_risk["risk_index"].mean()) if len(fc_risk) else None)

    # ---------------------------------------------- 8. explainability (XGBoost)
    log("7. Explainability (SHAP, permutation importance, partial dependence)")
    xgb_full = full_models["XGBoost"]
    expl = explain_model(xgb_full, s, test_start=s.index[test_start],
                         eval_forecaster=final_models["XGBoost"])  # test-stage model for permutation importance
    joblib.dump(xgb_full, config.MODELS_DIR / "xgboost_full.joblib")
    shap_frame = expl.pop("shap_frame")
    shap_frame.tail(24).to_csv(config.REPORTS_DIR / "shap_values_last24.csv")

    # ---------------------------------------------- 9. what-if scenarios
    trend = trend_per_year(monthly)
    scen = simulate(best_fc, trend)
    scen.insert(0, "month", future_idx.strftime("%Y-%m"))
    scen.to_csv(config.REPORTS_DIR / "scenarios.csv", index=False)
    scenario_20 = [c for c in scen.columns if c.startswith("C:")][0]
    gain_20 = float(best_fc.iloc[-1] - scen[scenario_20].iloc[-1]) if config.LEVEL_IS_DEPTH else None

    # ---------------------------------------------- 10. recommendations
    recs = recommendations(current, forecast_change_12 if forecast_change_12 else None,
                           ratio, gain_20, alerts, flat_runs)

    # ---------------------------------------------- 11. stations (map)
    log("8. Station overview for the map")
    stations = station_overview()
    stations.to_csv(config.REPORTS_DIR / "station_overview.csv", index=False)

    # ---------------------------------------------- 12. figures
    log("9. Figures")
    make_figures(monthly, s, anomalies, chosen, test_errors, best_group, best_fc, half_width,
                 comp, ablation_df, expl, fc_timeline, hist_timeline, scen, trend)

    # ---------------------------------------------- 13. summary
    summary = {
        "station": meta,
        "trained_on": {"train": dates["train"], "validation": dates["val"], "test": dates["test"],
                       "note": "Final models for the forecast are refit on all observed data after selection."},
        "data_quality": {
            "months": n, "missing_months_interpolated": meta["missing_months_interpolated"],
            "months_with_several_readings": meta["months_with_several_readings"],
            "repeated_value_months": flat_runs,
            "anomaly_months_flagged": int(anomalies["month"].nunique()),
            "unit_note": "Unit is unverified; the code assumes depth to water in metres (higher = deeper).",
            "seasonal_regime_check": regime,
        },
        "features_unavailable": {"rainfall": "not in dataset", "temperature": "not in dataset",
                                 "extraction": "not in dataset", "evapotranspiration": "not in dataset",
                                 "recharge_indicators": "derived from the groundwater record only",
                                 "geographic_data_used": "latitude/longitude for the station map only"},
        "horizons_supported_months": [1, 3, 6, 12],
        "horizons_not_supported": {"7_days": "monthly data", "30_days": "monthly data (1 month is the nearest)"},
        "model_selection": {"rule": "lowest multi-horizon MAE on the validation block; test block only for reporting",
                            "best_model": best_group, "best_configuration": chosen[best_group][1],
                            "test_ranking_by_multi_horizon_MAE": test_rank_by_multi},
        "model_comparison": comp.to_dict(orient="records"),
        "ablation": ablation,
        "forecast": {"best_model": best_group, "months": fc_table.to_dict(orient="records")},
        "risk": {"current": current, "weights_used": current["weights_used"],
                 "class_edges": config.RISK_CLASS_EDGES, "critical_depth_m": ctx.critical_depth,
                 "forecast_mean_risk_index": float(fc_risk["risk_index"].mean())},
        "alerts": alerts,
        "anomalies": anomalies.to_dict(orient="records"),
        "recovery": {"summary": rec, "events": events.to_dict(orient="records")},
        "health": health,
        "trend_m_per_decade": round(health["trend_m_per_decade"], 3),
        "explainability": expl,
        "scenarios": {"trend_m_per_year": trend, "share_assumption": config.EXTRACTION_SHARE_OF_TREND,
                      "labels": LABELS, "rows": scen.to_dict(orient="records")},
        "recommendations": recs,
        "stations_count": int(len(stations)),
        "run_info": {"seed": config.SEED, "horizon_months": H, "feature_columns": FEATURE_COLUMNS},
    }
    with open(config.REPORTS_DIR / "summary.json", "w") as f:
        json.dump(jsonable(summary), f, indent=2)
    with open(config.REPORTS_DIR / "model_comparison.md", "w") as f:
        f.write(comp.to_string(index=False))
    log(f"Done. Best model: {best_group}. Summary: reports/summary.json")


def make_figures(monthly, s, anomalies, chosen, test_errors, best_group, best_fc, band,
                 comp, ablation_df, expl, fc_timeline, hist_timeline, scen, trend):
    fig_dir = config.FIGURES_DIR
    # 1 history + anomalies
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(s.index, s.values, lw=0.9, label="Observed level")
    if len(anomalies):
        a = anomalies.set_index("month")
        ax.scatter(a.index, a["level"], s=18, c="red", label="Anomaly", zorder=3)
    ax.set_title("Observed groundwater level with flagged anomalies")
    ax.set_ylabel("Level (unit unverified; depth shown inverted)" if config.LEVEL_IS_DEPTH else "Level")
    if config.LEVEL_IS_DEPTH:
        ax.invert_yaxis()
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "01_history_anomalies.png", dpi=150); plt.close(fig)

    # 2 test actual vs predicted (best model at horizons 1, 6, 12)
    err = test_errors[best_group]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(err[err.horizon == 1]["month"], err[err.horizon == 1]["actual"], label="Actual", lw=1.4)
    for h, style in ((1, "-"), (6, "--"), (12, ":")):
        sub = err[err.horizon == h]
        ax.plot(sub["month"], sub["predicted"], style, label=f"Predicted, {h}-month-ahead")
    ax.set_title(f"Test period: {best_group}"); ax.legend(); ax.grid(alpha=0.3)
    ax.invert_yaxis() if config.LEVEL_IS_DEPTH else None
    fig.tight_layout(); fig.savefig(fig_dir / "02_test_actual_vs_predicted.png", dpi=150); plt.close(fig)

    # 3 model comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(comp["Model"], comp["Test MAE (multi-horizon)"], color="#3b7bbf", label="Test MAE, multi-horizon")
    ax.barh(comp["Model"], comp["Test MAE (1-step)"], color="#9bc1e6", alpha=0.7, label="Test MAE, 1-step")
    ax.invert_yaxis(); ax.set_xlabel("MAE (metres, if depth)"); ax.set_title("Model comparison on the test block")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "03_model_comparison.png", dpi=150); plt.close(fig)

    # 4 residuals of best model (multi-horizon h=1)
    resid = err[err.horizon == 1]
    res = resid["actual"] - resid["predicted"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(resid["month"], res, lw=1); axes[0].axhline(0, c="k", lw=0.8)
    axes[0].set_title("Residuals over time (1-step)")
    axes[1].hist(res, bins=20, color="#3b7bbf"); axes[1].set_title("Residual distribution")
    fig.tight_layout(); fig.savefig(fig_dir / "04_residuals.png", dpi=150); plt.close(fig)

    # 5 forecast horizon + what-if scenarios
    fig, ax = plt.subplots(figsize=(12, 5))
    tail = s.iloc[-36:]
    ax.plot(tail.index, tail.values, c="k", label="Observed (last 36 months)")
    ax.plot(best_fc.index, best_fc.values, c="#d1495b", lw=2, label=f"Forecast: {best_group}")
    ax.fill_between(best_fc.index, best_fc.values - band, best_fc.values + band, alpha=0.2, color="#d1495b",
                    label="Approx. 80% band (test errors)")
    for col in scen.columns[1:]:
        ax.plot(pd.to_datetime(scen["month"]), scen[col], "--", lw=1, label=col)
    ax.set_title("12-month forecast and what-if extraction scenarios (assumption-driven)")
    if config.LEVEL_IS_DEPTH:
        ax.invert_yaxis()
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "05_forecast_scenarios.png", dpi=150); plt.close(fig)

    # 6 risk trend
    rt = pd.read_csv(config.REPORTS_DIR / "risk_timeline.csv")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(pd.to_datetime(rt["month"]), rt["risk_index"], lw=1.5, label="Groundwater Risk Index")
    for edge, lab in zip(config.RISK_CLASS_EDGES, ["Moderate", "High", "Critical"]):
        ax.axhline(edge, c="grey", ls=":", lw=0.8)
    ax.axvline(pd.to_datetime(rt[rt["kind"] == "forecast"]["month"].iloc[0]), c="k", ls="--", lw=0.8)
    ax.set_ylim(0, 100); ax.set_title("Groundwater Risk Index (observed, then forecast)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "06_risk_trend.png", dpi=150); plt.close(fig)

    # 7 feature importance (SHAP mean absolute)
    shap_series = pd.Series(expl["mean_abs_shap"]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(shap_series.index, shap_series.values, color="#3b7bbf")
    ax.set_title("XGBoost: mean |SHAP| by feature (m)")
    fig.tight_layout(); fig.savefig(fig_dir / "07_feature_importance.png", dpi=150); plt.close(fig)

    # 8 error by horizon
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for group, e in test_errors.items():
        ht = e.groupby("horizon").apply(lambda g: (g.actual - g.predicted).abs().mean(), include_groups=False)
        ax.plot(ht.index, ht.values, marker="o", ms=3, label=group)
    ax.set_xlabel("Forecast horizon (months)"); ax.set_ylabel("MAE on test block")
    ax.set_title("Error growth with forecast horizon"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "08_error_by_horizon.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
