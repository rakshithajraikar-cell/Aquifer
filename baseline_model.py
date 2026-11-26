
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load groundwater data
df = pd.read_csv("data/selected_station.csv")
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

# Actual level for each month
df["actual"] = df["ground_water_level"]

# Baseline: predict this month using the previous month's level
df["prediction"] = df["ground_water_level"].shift(1)

# Remove the first row because it has no previous month
df = df.dropna().reset_index(drop=True)

# Use the same chronological 80/20 split
split_index = int(len(df) * 0.8)
test = df.iloc[split_index:]

# Evaluate baseline on the test period
mae = mean_absolute_error(test["actual"], test["prediction"])
rmse = np.sqrt(
    mean_squared_error(test["actual"], test["prediction"])
)

print("\nBaseline Model Evaluation")
print("-------------------------")
print("Testing records:", len(test))
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
