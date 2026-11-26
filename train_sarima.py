
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import joblib

# 1. Load groundwater data
df = pd.read_csv("data/selected_station.csv")
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").set_index("Month")

series = df["ground_water_level"].asfreq("MS")

# 2. Split chronologically
split_index = int(len(series) * 0.8)
train = series.iloc[:split_index]
test = series.iloc[split_index:]

print("Training records:", len(train))
print("Testing records:", len(test))

# 3. Build a simpler SARIMA model
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

# 4. Train
results = model.fit(disp=False)

# 5. Forecast the test period
predictions = results.forecast(steps=len(test))
predictions.index = test.index

# 6. Evaluate
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

print("\nUpdated SARIMA Model Evaluation")
print("--------------------------------")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# 7. Save model
Path("models").mkdir(exist_ok=True)
joblib.dump(results, "models/sarima_groundwater.joblib")
print("\nUpdated SARIMA model saved!")

# 8. Plot test predictions
plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label="Actual test data")
plt.plot(predictions.index, predictions, label="SARIMA prediction")

plt.title("Updated SARIMA Groundwater Forecast")
plt.xlabel("Month")
plt.ylabel("Groundwater level (units to be verified)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("data/sarima_test_results.png", dpi=300)
plt.show()
