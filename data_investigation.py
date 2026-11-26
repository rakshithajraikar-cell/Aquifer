
import pandas as pd
from pathlib import Path

# Load groundwater dataset
df = pd.read_csv("data/selected_station.csv")
df["Month"] = pd.to_datetime(df["Month"])

# Find consecutive repeated readings
df["run"] = (
    df["ground_water_level"]
    .ne(df["ground_water_level"].shift())
    .cumsum()
)

repeated = df.groupby("run").agg(
    reading=("ground_water_level", "first"),
    start=("Month", "first"),
    end=("Month", "last"),
    months=("ground_water_level", "size")
).reset_index(drop=True)

repeated = repeated[repeated["months"] >= 3]

# Prepare investigation report
report = f"""
GROUNDWATER DATA INVESTIGATION REPORT

Dataset: Karnataka Groundwater 1971-2020
Station Code: 70202
Location: Chitradurga, Karnataka

Total Records: {len(df)}
Start Date: {df['Month'].min().date()}
End Date: {df['Month'].max().date()}

DATA QUALITY CHECK
Missing Values: {df.isnull().sum().sum()}
Duplicate Dates: {df['Month'].duplicated().sum()}

READING 15.0 INVESTIGATION
Total Occurrences: {(df['ground_water_level'] == 15.0).sum()}

CONSECUTIVE REPEATED READINGS
{repeated.to_string(index=False)}

IMPORTANT FINDING
The reading 15.0 remained unchanged for
20 consecutive months from December 1989
to July 1991.

The original readings are preserved.
No values have been modified or removed.

GROUNDWATER MEASUREMENT UNIT: UNVERIFIED
Awaiting official clarification from NWIC.
"""

# Save report
Path("reports").mkdir(exist_ok=True)

with open("reports/data_investigation_report.txt", "w") as file:
    file.write(report)

print("Investigation report created successfully!")
print("Location: reports/data_investigation_report.txt")
