# AI-Powered Aquifer Groundwater Level Forecasting, Risk Assessment and Smart Water Management

An integrated decision-support system. It forecasts groundwater levels, scores risk, explains
predictions, simulates water-management scenarios and suggests mitigation actions. Every figure on
the dashboard is produced by `run_pipeline.py` and stored in `reports/`. Nothing is hard-coded or random.

## What is in the repository

| Path | Purpose |
|---|---|
| `run_pipeline.py` | Runs the whole experiment and writes all reports, figures and models |
| `aquifer_ai/` | Python package: data, features, models, evaluation, risk, anomalies, recovery, health, explainability, scenarios |
| `api.py` | FastAPI backend (JSON endpoints and the dashboard page) |
| `static/index.html` | Interactive dashboard (Plotly, no build step) |
| `tests/` | Unit tests for leakage, splits, bounds and baselines (`pytest -q tests`) |
| `docs/architecture.md` | Architecture, data-flow and use-case diagrams |
| `docs/TESTING.md` | Test documentation |
| `reports/` | Generated results: `model_comparison.csv`, `ablation_study.csv`, `forecasts.csv`, `summary.json`, figures |
| `models/` | Saved models (the ARIMA model fitted on all data, and the XGBoost model used for explanations) |
| `*.py` (original) | Earlier scripts from the first part of the project, left unchanged |

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py            # about 35 minutes on 2 CPU cores (LSTM/GRU and tree tuning dominate)
pytest -q tests
uvicorn api:app --reload          # then open http://127.0.0.1:8000
```

## Method

- **Data.** Station 70202 (Chitradurga, Karnataka), monthly, January 1972 to December 2020: 588 months.
  The readings are averaged per month. No months were missing, so nothing was interpolated.
- **Split.** Chronological only, never shuffled: train 1972-01 to 2006-03, validation 2006-04 to 2013-07,
  test 2013-08 to 2020-12.
- **Selection.** Each candidate is tuned and ranked on validation data, using 1–12 month forecasts made
  from the model's own predictions. The test block is used once, to report results, after refitting on train+validation.
- **Models.** Naive and seasonal-naive baselines, ARIMA (order chosen by AIC), SARIMA(1,1,1)(1,0,1)[12],
  Random Forest and XGBoost (grid-tuned), LSTM and GRU (PyTorch, early stopping on validation).
- **Metrics.** MAE, RMSE, MAPE, R², error by horizon, and training time.

## Results (test period, 2013-08 to 2020-12)

| Model | Val MAE (multi-h) | Test MAE 1-step | Test MAE multi-h | Test RMSE multi-h | Test R² multi-h | MAE at 12 months | Train time (s) |
|---|---|---|---|---|---|---|---|
| **ARIMA (selected)** | **4.20** | **2.03** | 5.84 | **7.38** | **0.40** | 8.43 | 0.48 |
| SARIMA | 4.32 | 2.06 | 6.36 | 8.12 | 0.27 | 9.04 | 0.23 |
| Naive | 4.62 | 2.33 | 6.76 | 8.58 | 0.19 | 8.44 | 0.00 |
| Random Forest | 4.71 | 2.58 | 6.52 | 8.28 | 0.24 | 7.81 | 0.50 |
| XGBoost | 4.86 | 2.78 | 7.31 | 9.29 | 0.05 | 9.61 | 0.10 |
| GRU | 5.98 | 2.12 | **5.78** | 7.61 | 0.36 | **7.82** | 23.1 |
| LSTM | 6.28 | 2.31 | 6.26 | 8.12 | 0.27 | 8.66 | 10.1 |
| Seasonal naive | 6.65 | 7.87 | 8.27 | 9.62 | -0.02 | 8.44 | 0.00 |

**Best model.** ARIMA has the lowest validation error, the lowest one-step test error and the highest multi-horizon
test R². It is the selected model. The result is not clear-cut, though. GRU has a marginally lower test error
at multi-horizon (5.78 vs 5.84 m) and at 12 months (7.82 vs 8.43 m), but it ranked last on validation (5.98 m).
The gap is small, the test window covers only 89 months, and no significance test was run. ARIMA is the defensible
default. GRU is the strongest challenger and should be re-checked when more data arrives.

**Ablation (XGBoost).** On validation, the full set (history, rolling and change, and calendar features) had the lowest
error at 4.86 m, against 5.05 m for history alone. Adding rolling and change features without calendar features was worse
at 5.27 m. On test the pattern reverses: history alone scored 7.08 m and the full set 7.31 m. The ablation does not
show a consistent benefit from any feature group, and the test result favours the simplest set.
Rainfall, temperature and extraction ablations were not run because those series are not in the dataset.

## Dashboard features and what each is grounded in

| Feature | Implementation | Grounded in |
|---|---|---|
| Forecasts (1, 3, 6, 12 months) | Best model refit on all data; approximate 80% band from test errors per horizon | `forecasts.csv` |
| Groundwater Risk Index (0–100, four classes) | Level percentile for the calendar month, 12-month decline, seasonal deviation, recharge ratio; weights renormalised over available parts | `risk_timeline.csv` |
| Early-warning alerts | Critical level (90th percentile of history), rapid decline (95th percentile of 12-month rises), possible drought stress, unusual fluctuations | `summary.json → alerts` |
| What-if simulator | Forecast minus an assumed extraction effect; the share is adjustable in the UI | `/api/whatif` (**assumption-driven**) |
| Recharge and recovery | Seasonal rise, response time, return-to-pre-monsoon time per year | `recovery_events.csv` |
| Anomaly detection | Robust z-score on jumps and seasonal residuals, repeated-value runs, Isolation Forest | `anomalies.csv` |
| Explainable AI | SHAP, permutation importance on the test period, partial dependence | `summary.json → explainability` |
| Aquifer Health Score | Weighted project indicator, with the excluded parts listed | `summary.json → health` |
| Recommendations | Rules that quote the figure each recommendation depends on | `summary.json → recommendations` |
| Map | Latest reading per monitoring well (1,518 wells), coloured by rank | `station_overview.csv` |

## Limitations (please read before presenting)

1. **No rainfall, temperature, extraction, evapotranspiration or recharge data.** These are the main drivers in the
   brief, and the dataset does not contain them. The two sources I tried (Open-Meteo, NASA POWER) were not reachable from
   the build environment. The models use groundwater history only, and the risk, health and explanation features
   cover only what the data supports. Rainfall deficit, temperature, extraction rate and climate-change scenarios
   are **not implemented**. The dashboard and summary say so explicitly. Adding a CSV with these columns is the next step.
2. **Recent data looks like a regime change.** The median seasonal rise was about 3 m in the earlier record and
   about 19 m over the last five years. Also, the 2017–2020 readings are much deeper than before. Before using recent values
   for decisions, confirm that the sensor, station and unit have not changed. The pipeline records this check in
   `summary.json → data_quality.seasonal_regime_check`.
3. **Unit and direction are unverified.** The code assumes depth to water in metres, so higher means deeper. Set
   `LEVEL_IS_DEPTH` in `aquifer_ai/config.py` if that is wrong.
4. **Monthly data.** The 7-day and 30-day horizons are not possible. The nearest horizon is one month.
5. **Forecast accuracy is modest.** Multi-horizon R² is 0.40 at best and 12-month MAE is about 8 m. The forecast
   bands are wide. The risk and health scores are relative indicators within this record, not calibrated probabilities.
6. **What-if effects are small under the default assumption.** The extraction effect uses an assumed share
   (`EXTRACTION_SHARE_OF_TREND = 0.5`) of the observed deepening trend, which is about 0.7 m a year. A 30% cut therefore
   moves the 12-month level by only about 0.1 m. That is a consequence of the missing extraction data, not a finding.
   Calibrate this with extraction records before relying on the scenarios.
7. **Deep-learning scope.** Only LSTM and GRU were implemented. TFT and PatchTST were not, because of compute time on
   this machine and the small monthly dataset.
8. **Not implemented:** React frontend (the dashboard is plain HTML and JavaScript), PostgreSQL/MongoDB (results are stored as
   files), Leaflet (Plotly's built-in map is used), and the report and UML documents. The architecture and use-case diagrams
   are in `docs/architecture.md`.

## Suggested report structure (final-year project)

1. Introduction: groundwater depletion and the need for decision support
2. Literature review: forecasting models for hydrology, explainable AI, risk indices
3. Data and preprocessing (Section 4 of the brief)
4. Methodology: models, chronological validation, metrics, risk index, anomaly detection, SHAP
5. Results: model comparison, ablation, horizon errors, risk and alerts, what-if (labelled), explanations
6. Discussion: why ARIMA was selected, the regime-change concern, and the missing drivers
7. Limitations and future work (see the limitations above)
8. Conclusion
