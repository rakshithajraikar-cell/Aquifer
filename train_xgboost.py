
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import joblib

# 1. Load groundwater data
df = pd.read_csv("data/selected_station.csv")
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

# 2. Create features from previous months
for lag in [1, 2, 3, 6, 12]:
    df[f"lag_{lag}"] = df["ground_water_level"].shift(lag)

# Calendar information for the month being predicted
df["month_number"] = df["Month"].dt.month
df["year"] = df["Month"].dt.year

# 3. Predict this month's level using previous months
df["target"] = df["ground_water_level"]
df = df.dropna().reset_index(drop=True)

features = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "month_number", "year"
]

X = df[features]
y = df["target"]

# 4. Split by time: earlier data for training,
# later data for testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 5. Train the XGBoost model
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# 6. Evaluate predictions
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nCorrected XGBoost Model Evaluation")
print("----------------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# 7. Save the model
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {"model": model, "features": features},
    "models/xgboost_groundwater.joblib"
)

print("\nModel saved successfully!")

# 8. Plot actual versus predicted levels
plt.figure(figsize=(12, 5))
plt.plot(
    df["Month"].iloc[split_index:],
    y_test,
    label="Actual"
)
plt.plot(
    df["Month"].iloc[split_index:],
    predictions,
    label="XGBoost prediction"
)

plt.title("One-Month-Ahead Groundwater Forecast")
plt.xlabel("Month")
plt.ylabel("Groundwater level (units to be verified)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("data/xgboost_test_results.png", dpi=300)
plt.show()
