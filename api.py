"""FastAPI backend for the Aquifer dashboard.

Run:  uvicorn api:app --reload
Then open http://127.0.0.1:8000

Every endpoint reads the artifacts written by run_pipeline.py (reports/ and data/),
except /api/whatif, which recomputes the scenario curves from the saved forecast
using the parameters you pass in. Nothing is invented at request time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aquifer_ai import config

app = FastAPI(title="AI-Powered Aquifer Groundwater Forecasting & Decision Support", version="1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _summary() -> dict:
    path = config.REPORTS_DIR / "summary.json"
    if not path.exists():
        raise HTTPException(503, "Run `python run_pipeline.py` first to generate reports.")
    with open(path) as f:
        return json.load(f)


def _csv(name: str) -> pd.DataFrame:
    path = config.REPORTS_DIR / name
    if not path.exists():
        raise HTTPException(503, f"Missing {name}. Run `python run_pipeline.py`.")
    return pd.read_csv(path)


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/summary")
def summary():
    s = _summary()
    return {
        "station": s["station"],
        "trained_on": s["trained_on"],
        "data_quality": s["data_quality"],
        "features_unavailable": s["features_unavailable"],
        "model_selection": s["model_selection"],
        "risk": s["risk"],
        "health": s["health"],
        "trend_m_per_decade": s["trend_m_per_decade"],
        "horizons_supported_months": s["horizons_supported_months"],
        "horizons_not_supported": s["horizons_not_supported"],
        "alerts_count": len(s["alerts"]),
        "stations_count": s["stations_count"],
        "latest_observation": _records(_history_frame().tail(1)),
    }


def _history_frame() -> pd.DataFrame:
    station = _summary()["station"]["station"]
    path = config.DATA_DIR / f"clean_station_{station}_monthly.csv"
    if not path.exists():
        raise HTTPException(503, "Missing cleaned data. Run `python run_pipeline.py`.")
    return pd.read_csv(path)


@app.get("/api/history")
def history(from_year: int | None = None):
    df = _history_frame()
    if from_year:
        df = df[pd.to_datetime(df["month"]).dt.year >= from_year]
    return _records(df)


@app.get("/api/forecast")
def forecast():
    s = _summary()
    return {"best_model": s["forecast"]["best_model"], "months": _records(_csv("forecasts.csv"))}


@app.get("/api/models")
def models():
    s = _summary()
    return {
        "selection_rule": s["model_selection"]["rule"],
        "best_model": s["model_selection"]["best_model"],
        "best_configuration": s["model_selection"]["best_configuration"],
        "test_ranking_by_multi_horizon_MAE": s["model_selection"]["test_ranking_by_multi_horizon_MAE"],
        "comparison": s["model_comparison"],
        "ablation": s["ablation"],
    }


@app.get("/api/risk")
def risk():
    s = _summary()
    return {"current": s["risk"]["current"], "class_edges": s["risk"]["class_edges"],
            "critical_depth_m": s["risk"]["critical_depth_m"],
            "forecast_mean_risk_index": s["risk"]["forecast_mean_risk_index"],
            "timeline": _records(_csv("risk_timeline.csv")),
            "unavailable_components": s["features_unavailable"]}


@app.get("/api/alerts")
def alerts():
    return _summary()["alerts"]


@app.get("/api/anomalies")
def anomalies():
    return _summary()["anomalies"]


@app.get("/api/recovery")
def recovery():
    return _summary()["recovery"]


@app.get("/api/health")
def health():
    return _summary()["health"]


@app.get("/api/explain")
def explain():
    e = _summary()["explainability"]
    return e


@app.get("/api/recommendations")
def recommendations():
    return _summary()["recommendations"]


@app.get("/api/stations")
def stations():
    return _records(_csv("station_overview.csv"))


@app.get("/api/whatif")
def whatif(
    extraction_share: float = Query(config.EXTRACTION_SHARE_OF_TREND, ge=0, le=1,
                                    description="ASSUMPTION: share of observed deepening caused by extraction"),
    reduction_pct: str = Query("0,10,20,30", description="Comma-separated extraction cuts in percent"),
):
    """Scenario curves = best-model forecast minus the assumed extraction effect.
    Labelled as assumption-driven: the dataset has no extraction series."""
    s = _summary()
    fc = _csv("forecasts.csv")
    base = fc["forecast_level"].to_numpy()
    trend = float(s["scenarios"]["trend_m_per_year"])
    h = np.arange(1, len(base) + 1)
    try:
        cuts = [float(x) for x in reduction_pct.split(",") if x.strip() != ""]
    except ValueError:
        raise HTTPException(400, "reduction_pct must be comma-separated numbers")
    curves = {}
    for cut in cuts:
        effect = (cut / 100) * extraction_share * max(trend, 0) * h / 12
        adjusted = base - effect if config.LEVEL_IS_DEPTH else base + effect
        label = "Current extraction" if cut == 0 else f"{cut:g}% reduction in extraction"
        curves[label] = [round(float(v), 3) for v in adjusted]
    return JSONResponse({
        "label": "ASSUMPTION-DRIVEN SCENARIOS, not learned from data (no extraction series in the dataset)",
        "months": fc["month"].tolist(),
        "baseline_forecast": [round(float(v), 3) for v in base],
        "trend_m_per_year": trend,
        "extraction_share_assumption": extraction_share,
        "curves": curves,
    })


@app.get("/api/health-check")
def health_check():
    return {"status": "ok", "reports_ready": (config.REPORTS_DIR / "summary.json").exists()}
