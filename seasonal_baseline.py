
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load groundwater data
df = pd.read_csv("data/selected_station.csv")
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

# 2. Predict using the level from 12 months earlier
df["prediction"] = df["ground_water_level"].shift(12)
df["actual"] = df["ground_water_level"]

# 3. Remove rows without 12 months of history
df = df.dropna().reset_index(drop=True)

# 4. Use the same chronological 80/20 test split
split_index = int(len(df) * 0.8)
test = df.iloc[split_index:]

# 5. Evaluate predictions
mae = mean_absolute_error(test["actual"], test["prediction"])
rmse = np.sqrt(
    mean_squared_error(test["actual"], test["prediction"])
)

print("\nSeasonal Naive Model Evaluation")
print("--------------------------------")
print("Testing records:", len(test))
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
