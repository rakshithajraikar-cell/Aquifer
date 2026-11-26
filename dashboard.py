"""Streamlit dashboard for the Aquifer decision-support system.

Run:  streamlit run dashboard.py

Reads only the files written by run_pipeline.py (reports/, data/). The what-if
simulator recalculates scenario curves from the saved forecast using the
sliders in the sidebar. Nothing is generated randomly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
LEVEL_IS_DEPTH = True  # must match aquifer_ai/config.py

st.set_page_config(page_title="Aquifer Groundwater Dashboard", layout="wide", page_icon="💧")


# ----------------------------------------------------------------- loading
@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    path = REPORTS / "summary.json"
    if not path.exists():
        st.error("reports/summary.json not found. Run `python run_pipeline.py` first.")
        st.stop()
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_frames():
    station = load_summary()["station"]["station"]
    history = pd.read_csv(DATA / f"clean_station_{station}_monthly.csv", parse_dates=["month"])
    forecast = pd.read_csv(REPORTS / "forecasts.csv")
    forecast["month"] = pd.to_datetime(forecast["month"])
    risk = pd.read_csv(REPORTS / "risk_timeline.csv")
    risk["month"] = pd.to_datetime(risk["month"])
    stations = pd.read_csv(REPORTS / "station_overview.csv")
    events = pd.read_csv(REPORTS / "recovery_events.csv")
    scenarios_base = pd.read_csv(REPORTS / "scenarios.csv")
    return history, forecast, risk, stations, events, scenarios_base


def whatif_curves(baseline: np.ndarray, trend: float, share: float, cuts: list[float]) -> dict:
    h = np.arange(1, len(baseline) + 1)
    curves = {}
    for cut in cuts:
        effect = (cut / 100) * share * max(trend, 0) * h / 12
        adjusted = baseline - effect if LEVEL_IS_DEPTH else baseline + effect
        label = "Current extraction" if cut == 0 else f"{cut:g}% reduction in extraction"
        curves[label] = adjusted
    return curves


def fmt(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


CLASS_COLOURS = {"Low Risk": "#2e9e5b", "Moderate Risk": "#d9a520",
                 "High Risk": "#e0702b", "Critical Risk": "#c8323c"}


def depth_axis(fig):
    if LEVEL_IS_DEPTH:
        fig.update_yaxes(autorange="reversed")
    return fig


# ----------------------------------------------------------------- data
S = load_summary()
history, forecast, risk, stations, events, _ = load_frames()
station = S["station"]
risk_now = S["risk"]["current"]
health = S["health"]
recs = S["recommendations"]
alerts = S["alerts"]
anomalies = pd.DataFrame(S["anomalies"])
if len(anomalies):
    anomalies["month"] = pd.to_datetime(anomalies["month"])
trend = float(S["scenarios"]["trend_m_per_year"])

# ----------------------------------------------------------------- header
st.title("💧 AI-Powered Aquifer Groundwater Forecasting & Decision Support")
st.caption(
    f"Station {station['station']} · {station['block']}, {station['district']} · "
    f"test period {S['trained_on']['test'][0]} to {S['trained_on']['test'][1]} · "
    f"{S['stations_count']} monitoring wells in the dataset"
)
st.info(
    "**Groundwater unit is unverified.** Values are treated as depth to water in metres (higher = deeper). "
    "Rainfall, temperature, extraction and evapotranspiration are not in the dataset, so the models use "
    "groundwater history only. Forecasts are monthly."
)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("What-if simulator")
    st.caption("Assumption-driven: the dataset has no extraction series.")
    share = st.slider("Share of observed deepening caused by extraction", 0.0, 1.0,
                      float(S["scenarios"]["share_assumption"]), 0.05)
    cuts = st.multiselect("Extraction reductions (%)", [0, 10, 20, 30, 40, 50], default=[0, 10, 20, 30])
    if 0 not in cuts:
        cuts = [0] + cuts
    cuts = sorted(cuts)
    st.divider()
    st.caption(f"Best model: **{S['model_selection']['best_model']}** "
               f"({S['model_selection']['best_configuration']})")

tabs = st.tabs(["Overview", "Forecast", "Risk & Alerts", "What-if Simulator",
                "Explainable AI", "Recommendations", "Model Comparison", "Map"])

# ----------------------------------------------------------------- overview
with tabs[0]:
    last = history.iloc[-1]
    fc_last = forecast.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current level (m)", fmt(last["level"]), f"observed {last['month']:%b %Y}", delta_color="off")
    c2.metric("Forecast in 12 months (m)", fmt(fc_last["forecast_level"]),
              f"band {fmt(fc_last['band_low'])}–{fmt(fc_last['band_high'])}", delta_color="off")
    c3.metric("Groundwater Risk Index", fmt(risk_now["risk_index"], 1), risk_now["risk_class"], delta_color="off")
    c4.metric("Aquifer Health Score", f"{fmt(health['score'], 1)}/100",
              "project-developed indicator", delta_color="off")
    c5.metric("Long-term trend (m/decade)", fmt(S["trend_m_per_decade"]), "positive = deepening", delta_color="off")

    recent = history.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["month"], y=recent["level"], name="Observed (last 10 years)",
                             line=dict(color="#1f4f86")))
    fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["forecast_level"], name="Forecast",
                             line=dict(color="#d1495b", dash="dash")))
    shown = anomalies[anomalies["month"].isin(recent["month"])] if len(anomalies) else anomalies
    if len(shown):
        fig.add_trace(go.Scatter(x=shown["month"], y=shown["level"], mode="markers", name="Anomaly",
                                 marker=dict(color="red", size=7)))
    fig.update_layout(height=420, yaxis_title="Depth (m)", legend=dict(orientation="h"))
    st.plotly_chart(depth_axis(fig), width="stretch")

    dq = S["data_quality"]
    st.markdown(
        f"**Trend:** observed deepening of {fmt(S['trend_m_per_decade'])} m per decade. "
        f"**Data quality:** {dq['months']} months · {dq['missing_months_interpolated']} missing months interpolated "
        f"(flagged) · {dq['months_with_several_readings']} months averaged from several readings · "
        f"{dq['repeated_value_months']} repeated-value months · {dq['anomaly_months_flagged']} anomalous months."
    )
    regime = dq.get("seasonal_regime_check", {})
    if regime.get("note") and "No marked" not in regime["note"]:
        st.warning(f"Seasonal regime check: median seasonal rise was {fmt(regime['median_seasonal_rise_earlier_m'], 1)} m "
                   f"earlier and {fmt(regime['median_seasonal_rise_last_5_years_m'], 1)} m in the last five years. "
                   f"{regime['note']}")

# ----------------------------------------------------------------- forecast
with tabs[1]:
    st.markdown("Monthly data, so the nearest horizons are **1, 3, 6 and 12 months**. "
                "7-day and 30-day forecasts are not possible from monthly readings. Bands are approximate 80% "
                "intervals from test-period errors at each horizon.")
    pick = forecast[forecast["horizon_months"].isin([1, 3, 6, 12])][
        ["horizon_months", "month", "forecast_level", "band_low", "band_high"]].copy()
    pick["month"] = pick["month"].dt.strftime("%b %Y")
    pick.columns = ["Horizon (months)", "Month", "Forecast (m)", "Band low (m)", "Band high (m)"]
    st.dataframe(pick.round(2), hide_index=True, width="stretch")

    H = history.tail(48)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=H["month"], y=H["level"], name="Observed", line=dict(color="#1f4f86")))
    fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["band_high"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["band_low"], fill="tonexty",
                             fillcolor="rgba(209,73,91,0.15)", line=dict(width=0), name="Approx. 80% band"))
    fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["forecast_level"], name="Forecast",
                             line=dict(color="#d1495b", width=3)))
    fig.update_layout(height=460, yaxis_title="Depth (m)", legend=dict(orientation="h"))
    st.plotly_chart(depth_axis(fig), width="stretch")

    with st.expander("All models' forecasts (12 months)"):
        cols = [c for c in forecast.columns if c.startswith("forecast: ")]
        table = forecast[["month"] + cols].copy()
        table["month"] = table["month"].dt.strftime("%b %Y")
        table.columns = ["Month"] + [c.replace("forecast: ", "") for c in cols]
        st.dataframe(table.round(2), hide_index=True, width="stretch")

# ----------------------------------------------------------------- risk
with tabs[2]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Index now", fmt(risk_now["risk_index"], 1), risk_now["risk_class"], delta_color="off")
    c2.metric("Mean risk over forecast", fmt(S["risk"]["forecast_mean_risk_index"], 1), delta_color="off")
    c3.metric("Critical depth threshold (m)", fmt(S["risk"]["critical_depth_m"]),
              "90th percentile of history", delta_color="off")
    ratio = S["recovery"]["summary"].get("latest_recovery_ratio")
    c4.metric("Last seasonal rise vs median", f"{fmt(ratio)}×" if ratio is not None else "–",
              "above 1 = stronger than usual", delta_color="off")

    st.subheader("Early-warning alerts")
    if not alerts:
        st.success("No alerts.")
    for a in alerts:
        text = a["message"]
        if a.get("window_start"):
            text += f"  \n*Window: {a['window_start']} to {a['window_end']}*"
        if a["severity"] == "critical":
            st.error(f"**{a['type'].replace('_', ' ')}** — {text}")
        elif a["severity"] == "high":
            st.warning(f"**{a['type'].replace('_', ' ')}** — {text}")
        elif a["severity"] == "not_evaluated":
            st.caption(f"{a['type'].replace('_', ' ')}: {a['message']}")
        else:
            st.info(f"**{a['type'].replace('_', ' ')}** — {text}")

    st.subheader("Risk Index over time")
    fig = go.Figure()
    for kind, colour in (("observed", "#2f6fb3"), ("forecast", "#d1495b")):
        sub = risk[risk["kind"] == kind]
        fig.add_trace(go.Scatter(x=sub["month"], y=sub["risk_index"], name=kind.capitalize(),
                                 line=dict(color=colour)))
    for edge in S["risk"]["class_edges"]:
        fig.add_hline(y=edge, line_dash="dot", line_color="grey")
    fig.update_layout(height=380, yaxis=dict(range=[0, 100], title="Risk Index (0–100)"),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Components: level percentile for the calendar month, 12-month decline, seasonal deviation, "
               "recharge. Rainfall and extraction components are not computed because those series are not in the dataset.")

    st.subheader("Recovery after the monsoon")
    rs = S["recovery"]["summary"]
    st.markdown(
        f"Across {rs['years_analysed']} years: median seasonal rise **{fmt(rs['median_rise_m'])} m**, reached in a median of "
        f"**{fmt(rs['median_response_months'], 0)} months** from the pre-monsoon low. The level returned to its "
        f"pre-monsoon depth after a median of **{fmt(rs['median_recession_months'], 0)} months**. "
        f"The level recovered in {fmt(100 * rs['share_of_years_with_recovery'], 0)}% of years."
    )
    fig = go.Figure(go.Bar(x=events["year"], y=events["rise"], marker_color="#2f6fb3"))
    fig.update_layout(height=300, yaxis_title="Seasonal rise (m)", xaxis_title="Year")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Anomalies")
    if len(anomalies):
        view = anomalies.sort_values("month", ascending=False).head(25).copy()
        view["month"] = view["month"].dt.strftime("%Y-%m")
        st.dataframe(view[["month", "level", "methods", "reason"]].round(2), hide_index=True,
                     width="stretch")
        if len(anomalies) > 25:
            st.caption(f"Showing latest 25 of {len(anomalies)}.")
    else:
        st.write("No anomalies flagged.")

# ----------------------------------------------------------------- what-if
with tabs[3]:
    st.warning("These scenarios are **assumption-driven**, not learned from data. The effect of an extraction "
               "cut is the observed long-term deepening × the share set in the sidebar × the cut. "
               "Under the default share, the effect is small; calibrate it with extraction records.")
    base = forecast["forecast_level"].to_numpy()
    curves = whatif_curves(base, trend, share, cuts)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["month"], y=base, name="Baseline forecast",
                             line=dict(color="#666", dash="dot")))
    for label, vals in curves.items():
        fig.add_trace(go.Scatter(x=forecast["month"], y=vals, name=label))
    fig.update_layout(height=460, yaxis_title="Depth (m)", legend=dict(orientation="h"))
    st.plotly_chart(depth_axis(fig), width="stretch")

    rows = []
    for label, vals in curves.items():
        rows.append({"Scenario": label, "Level at 12 months (m)": round(float(vals[-1]), 3),
                     "Change vs baseline (m)": round(float(vals[-1] - base[-1]), 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Trend used: {trend:.3f} m per year · extraction share: {share:.2f} (assumption).")

# ----------------------------------------------------------------- XAI
with tabs[4]:
    e = S["explainability"]
    st.subheader(f"Why the model predicts what it does ({e['explained_month']})")
    st.info(e["explanation"])
    c1, c2 = st.columns(2)
    shap_items = sorted(e["mean_abs_shap"].items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in shap_items], y=[k for k, _ in shap_items], orientation="h",
                           marker_color="#2f6fb3"))
    fig.update_layout(height=420, title="Mean |SHAP| by feature (m)", margin=dict(l=160))
    c1.plotly_chart(fig, width="stretch")
    perm_items = sorted(e["permutation_importance_test"].items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in perm_items], y=[k for k, _ in perm_items], orientation="h",
                           marker_color="#d1495b"))
    fig.update_layout(height=420, title="Permutation importance (test period)", margin=dict(l=160))
    c2.plotly_chart(fig, width="stretch")

    st.subheader("Partial dependence")
    fig = go.Figure()
    for feat, pd_data in e["partial_dependence"].items():
        fig.add_trace(go.Scatter(x=pd_data["grid"], y=pd_data["average_prediction"], name=feat))
    fig.update_layout(height=380, xaxis_title="Feature value", yaxis_title="Average prediction (m)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Features are past levels, changes and calendar position only. Rainfall, temperature and "
               "extraction are not in the dataset, so they cannot appear here.")

# ----------------------------------------------------------------- recommendations
with tabs[5]:
    st.caption("Each recommendation quotes the model output it depends on. Advisory only; needs local hydrogeological review.")
    for r in recs:
        st.markdown(f"**{r['action']}**  \n*{r['category'].replace('_', ' ')}* · Evidence: {r['evidence']}")

# ----------------------------------------------------------------- models
with tabs[6]:
    m = S["model_selection"]
    st.info(f"**Selection rule:** {m['rule']}. Best model: **{m['best_model']}** ({m['best_configuration']}). "
            f"Test ranking by multi-horizon MAE: {' › '.join(m['test_ranking_by_multi_horizon_MAE'])}.")
    comp = pd.DataFrame(S["model_comparison"])
    show = ["Rank (validation)", "Model", "Family", "Val MAE (multi-horizon)", "Test MAE (1-step)",
            "Test RMSE (1-step)", "Test R2 (1-step)", "Test MAE (multi-horizon)", "Test RMSE (multi-horizon)",
            "Test R2 (multi-horizon)", "MAE h=1", "MAE h=3", "MAE h=6", "MAE h=12", "Training time (s)"]
    st.dataframe(comp[show].round(3), hide_index=True, width="stretch")
    st.caption("Tuned and ranked on the validation block. Test block used only for this table. "
               "'1-step' uses real previous values; 'multi-horizon' uses the model's own forecasts for 1–12 months.")

    fig = go.Figure()
    for _, r in comp.iterrows():
        fig.add_trace(go.Scatter(x=[1, 3, 6, 12], y=[r["MAE h=1"], r["MAE h=3"], r["MAE h=6"], r["MAE h=12"]],
                                 mode="lines+markers", name=r["Model"]))
    fig.update_layout(height=420, xaxis_title="Horizon (months)", yaxis_title="MAE on test (m)",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Ablation study (XGBoost feature groups)")
    ab = pd.DataFrame(S["ablation"])
    st.dataframe(ab.round(3), hide_index=True, width="stretch")
    st.caption("Rainfall, temperature and extraction ablations are not run because those series are not in the dataset.")

# ----------------------------------------------------------------- map
with tabs[7]:
    st.caption("Each monitoring well is coloured by where its latest reading ranks among all wells. "
               "Full risk analysis is available only for the modelled station.")
    fig = go.Figure(go.Scattermap(
        lat=stations["latitude"], lon=stations["longitude"], mode="markers",
        marker=dict(size=7, color=stations["level_percentile_among_stations"], colorscale="RdYlGn",
                    reversescale=True, cmin=0, cmax=100, colorbar=dict(title="Percentile")),
        text=[f"{s} · {b}, {d}<br>Latest: {l:.2f} m ({dt})<br>Percentile: {p:.0f}"
              for s, b, d, l, dt, p in zip(stations["station"], stations["block"], stations["district"],
                                           stations["latest_level"], stations["latest_date"],
                                           stations["level_percentile_among_stations"])],
        hoverinfo="text",
    ))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                      map=dict(style="open-street-map", center=dict(lat=14.5, lon=76.0), zoom=5.3))
    st.plotly_chart(fig, width="stretch")
