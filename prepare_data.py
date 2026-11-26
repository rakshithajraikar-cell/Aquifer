
import pandas as pd
from pathlib import Path

input_file = Path("data/karnataka_groundwater.csv")
output_file = Path("data/selected_station.csv")

print("Loading groundwater dataset...")
df = pd.read_csv(input_file, low_memory=False)

# Clean column formats
df["Monitoring Date"] = pd.to_datetime(
    df["Monitoring Date"],
    format="%d-%m-%Y",
    errors="coerce"
)
df["Ground Water Level"] = pd.to_numeric(
    df["Ground Water Level"],
    errors="coerce"
)

# Select one station
station = df[df["Station Code"].astype(str).str.strip() == "70202"].copy()

# Remove invalid readings
station = station.dropna(
    subset=["Monitoring Date", "Ground Water Level"]
)

# Sort by date
station = station.sort_values("Monitoring Date")

# Report data quality before handling duplicates
print("\nStation: 70202")
print("District:", station["District"].iloc[0])
print("Block:", station["Block"].iloc[0])
print("First observation:", station["Monitoring Date"].min().date())
print("Last observation:", station["Monitoring Date"].max().date())
print("Records before duplicate check:", len(station))

duplicates = station["Monitoring Date"].duplicated().sum()
print("Duplicate dates:", duplicates)

# Keep one reading per month.
# If multiple readings occur in a month, use their mean.
station["Month"] = station["Monitoring Date"].dt.to_period("M")

monthly = station.groupby("Month").agg(
    ground_water_level=("Ground Water Level", "mean"),
    observations=("Ground Water Level", "count")
).reset_index()

monthly["Month"] = monthly["Month"].dt.to_timestamp()
monthly = monthly.sort_values("Month")

# Identify missing calendar months
full_months = pd.date_range(
    monthly["Month"].min(),
    monthly["Month"].max(),
    freq="MS"
)
missing_months = full_months.difference(monthly["Month"])

print("Monthly records:", len(monthly))
print("Missing months:", len(missing_months))

if len(missing_months) > 0:
    print("First missing months:", list(missing_months[:12].date))

output_file.parent.mkdir(parents=True, exist_ok=True)
monthly.to_csv(output_file, index=False)

print("\nSaved selected station data to:", output_file)
print("\nFirst five monthly rows:")
print(monthly.head().to_string(index=False))
print("\nLast five monthly rows:")
print(monthly.tail().to_string(index=False))
