
import pandas as pd

file_path = "data/karnataka_groundwater.csv"

print("Reading groundwater dataset...")
df = pd.read_csv(file_path, low_memory=False)

# Convert the monitoring date into a date format
df["Monitoring Date"] = pd.to_datetime(
    df["Monitoring Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Convert groundwater readings into numbers
df["Ground Water Level"] = pd.to_numeric(
    df["Ground Water Level"],
    errors="coerce"
)

# Remove rows with missing station, date, or groundwater level
df = df.dropna(
    subset=["Station Code", "Monitoring Date", "Ground Water Level"]
)

# Summarize each station's history
summary = df.groupby(
    ["Station Code", "District", "Block"]
).agg(
    records=("Ground Water Level", "count"),
    first_date=("Monitoring Date", "min"),
    last_date=("Monitoring Date", "max"),
    unique_months=("Monitoring Date", lambda x: x.dt.to_period("M").nunique())
).reset_index()

# Prefer stations with more records and longer histories
summary = summary.sort_values(
    ["unique_months", "records"],
    ascending=False
)

print("\nTotal usable records:", len(df))
print("\nTop 10 stations with the longest monthly coverage:")
print(summary.head(10).to_string(index=False))

summary.to_csv("data/station_summary.csv", index=False)
print("\nSaved station summary to data/station_summary.csv")
