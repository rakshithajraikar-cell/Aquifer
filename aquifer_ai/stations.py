"""Station overview for the map: latest reading per monitoring well."""
from __future__ import annotations

import pandas as pd

from . import config


def station_overview(raw_csv=config.RAW_CSV) -> pd.DataFrame:
    raw = pd.read_csv(
        raw_csv,
        low_memory=False,
        dtype={"Station Code": str},
        usecols=["Station Code", "District", "Block", "Latitude", "Longitude",
                 "Monitoring Date", "Ground Water Level"],
    )
    raw["Station Code"] = raw["Station Code"].str.strip()
    raw["date"] = pd.to_datetime(raw["Monitoring Date"], format="%d-%m-%Y", errors="coerce")
    raw["value"] = pd.to_numeric(raw["Ground Water Level"], errors="coerce")
    raw = raw.dropna(subset=["Station Code", "date", "value", "Latitude", "Longitude"])

    latest_date = raw.groupby("Station Code")["date"].max().rename("latest_date")
    meta = raw.groupby("Station Code").agg(
        district=("District", "first"), block=("Block", "first"),
        latitude=("Latitude", "first"), longitude=("Longitude", "first"),
        readings=("value", "size"), first_date=("date", "min"),
    )
    rows = raw.merge(latest_date, left_on="Station Code", right_index=True)
    latest = rows[rows["date"] == rows["latest_date"]].groupby("Station Code")["value"].mean()
    out = meta.join(latest.rename("latest_level")).join(latest_date)
    out["level_percentile_among_stations"] = out["latest_level"].rank(pct=True) * 100
    out = out.reset_index().rename(columns={"Station Code": "station"})
    out["latest_date"] = out["latest_date"].dt.strftime("%Y-%m-%d")
    out["first_date"] = out["first_date"].dt.strftime("%Y-%m-%d")
    return out
