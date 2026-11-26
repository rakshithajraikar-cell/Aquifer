
import pandas as pd
import matplotlib.pyplot as plt

# Load the selected groundwater dataset
df = pd.read_csv("data/selected_station.csv")
df["Month"] = pd.to_datetime(df["Month"])

df = df.sort_values("Month").reset_index(drop=True)
levels = df["ground_water_level"]

print("\nDATA QUALITY REPORT")
print("-------------------")

print("Total records:", len(df))
print("First month:", df["Month"].min())
print("Last month:", df["Month"].max())
print("Missing groundwater values:", levels.isna().sum())
print("Duplicate months:", df["Month"].duplicated().sum())

print("\nGroundwater level statistics:")
print(levels.describe())

print("\nLowest 10 groundwater readings:")
print(df.nsmallest(10, "ground_water_level")[["Month", "ground_water_level"]])

print("\nHighest 10 groundwater readings:")
print(df.nlargest(10, "ground_water_level")[["Month", "ground_water_level"]])

# Calculate month-to-month changes
df["monthly_change"] = levels.diff()

print("\nLargest absolute monthly changes:")
print(
    df.loc[df["monthly_change"].abs().nlargest(10).index,
           ["Month", "ground_water_level", "monthly_change"]]
    .sort_values("Month")
)

# Plot the monthly changes
plt.figure(figsize=(12, 5))
plt.plot(df["Month"], df["ground_water_level"])
plt.title("Groundwater Data Quality Check")
plt.xlabel("Month")
plt.ylabel("Groundwater level (units to be verified)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/groundwater_quality_check.png", dpi=300)
plt.show()
