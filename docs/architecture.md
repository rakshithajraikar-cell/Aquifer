# Architecture, data flow and use cases

## 1. Pipeline architecture

The stages follow the architecture given in the project brief. Each stage maps to a module.

```mermaid
flowchart TD
    A[Data Sources<br/>karnataka_groundwater.csv] --> B[Data Preprocessing<br/>aquifer_ai/data.py]
    B --> C[Feature Engineering<br/>aquifer_ai/features.py]
    C --> D[Exploratory Data Analysis<br/>reports/data_investigation_report.txt, data_quality]
    D --> E[Anomaly Detection<br/>aquifer_ai/anomaly.py]
    E --> F[Model Training<br/>aquifer_ai/models.py]
    F --> G[Model Comparison<br/>aquifer_ai/evaluate.py]
    G --> H[Best Model Selection<br/>validation block only]
    H --> I[Groundwater Forecasting<br/>1, 3, 6, 12 months]
    I --> J[Risk Assessment<br/>aquifer_ai/risk.py]
    J --> K[Explainable AI<br/>aquifer_ai/explain.py]
    I --> L[Scenario Simulation<br/>aquifer_ai/scenarios.py]
    J --> M[Recommendations<br/>aquifer_ai/risk.py]
    K --> N[Interactive Dashboard<br/>api.py + static/index.html]
    L --> N
    M --> N
    J --> N
```

## 2. Data-flow diagram

```mermaid
flowchart LR
    subgraph Offline pipeline: run_pipeline.py
      R[(Raw CSV<br/>~45 MB, all stations)] --> P1[Clean one station<br/>monthly mean, interpolate, flag]
      P1 --> P2[Chronological split<br/>70 / 15 / 15]
      P2 --> P3[Tune & rank on validation]
      P3 --> P4[Refit on train+val<br/>score test]
      P4 --> P5[Refit on all data<br/>12-month forecast]
      P5 --> P6[Risk, alerts, recovery,<br/>health, SHAP, scenarios]
      P6 --> REP[(reports/*.csv, summary.json,<br/>figures/, models/*.joblib)]
    end
    subgraph Online service: uvicorn api:app
      REP --> API[FastAPI JSON endpoints]
      API --> UI[Browser dashboard]
      UI -- slider / cut % --> WI[/api/whatif recalculates<br/>from saved forecast/]
      WI --> UI
    end
```

## 3. Use-case diagram

```mermaid
flowchart LR
    Analyst((Hydrogeologist / analyst))
    Planner((Water-management planner))
    Researcher((Researcher / examiner))
    subgraph Aquifer system
      UC1[View current level & trend]
      UC2[View 1–12 month forecasts with bands]
      UC3[See risk index & early-warning alerts]
      UC4[Simulate extraction cuts 10/20/30%]
      UC5[Read SHAP explanations]
      UC6[Read recharge & conservation recommendations]
      UC7[Compare models & ablation]
      UC8[Browse monitoring wells on map]
      UC9[Re-run pipeline on new data]
    end
    Analyst --> UC1
    Analyst --> UC2
    Analyst --> UC3
    Analyst --> UC5
    Planner --> UC3
    Planner --> UC4
    Planner --> UC6
    Planner --> UC8
    Researcher --> UC7
    Researcher --> UC5
    Analyst --> UC9
```

## 4. Modules at a glance

| Module | Responsibility |
|---|---|
| `aquifer_ai/config.py` | All thresholds, weights and assumptions in one place |
| `aquifer_ai/data.py` | Load, clean, flag missing months, chronological split |
| `aquifer_ai/features.py` | Lag, rolling, change and calendar features (past values only) |
| `aquifer_ai/models.py` | Naive, seasonal naive, ARIMA, SARIMA, Random Forest, XGBoost, LSTM, GRU |
| `aquifer_ai/evaluate.py` | MAE, RMSE, MAPE, R², one-step and multi-horizon errors |
| `aquifer_ai/anomaly.py` | Sudden jumps, seasonal residuals, repeated values, Isolation Forest |
| `aquifer_ai/recovery.py` | Seasonal rise, response time, recession time, recharge ratio |
| `aquifer_ai/risk.py` | Groundwater Risk Index, early-warning alerts, recommendations |
| `aquifer_ai/health.py` | Aquifer Health Score (project-developed indicator) |
| `aquifer_ai/explain.py` | SHAP, permutation importance, partial dependence |
| `aquifer_ai/scenarios.py` | What-if extraction scenarios (assumption-driven) |
| `aquifer_ai/stations.py` | Latest reading per monitoring well for the map |
| `api.py` | FastAPI endpoints over the artifacts |
| `static/index.html` | Dashboard (Plotly, no build step) |
