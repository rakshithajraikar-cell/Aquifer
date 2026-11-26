
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("data/selected_station.csv")
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

# Build lag features
for lag in [1, 2, 3, 6, 12]:
    df[f"lag_{lag}"] = df["ground_water_level"].shift(lag)

df["month_number"] = df["Month"].dt.month
df["year"] = df["Month"].dt.year
df["target"] = df["ground_water_level"]

# Keep the same rows for both models
df = df.dropna().reset_index(drop=True)

features = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "month_number", "year"
]

# Same chronological split for both models
split_index = int(len(df) * 0.8)
test = df.iloc[split_index:]

actual = test["target"]

# Baseline: use the previous month's level
baseline_predictions = test["lag_1"]

# Load the saved XGBoost model
saved = joblib.load("models/xgboost_groundwater.joblib")
xgb_model = saved["model"]
xgb_predictions = xgb_model.predict(test[features])

# Compare performance
for name, predictions in [
    ("Baseline", baseline_predictions),
    ("XGBoost", xgb_predictions)
]:
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
