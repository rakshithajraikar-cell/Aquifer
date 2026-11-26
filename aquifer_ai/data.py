"""Load and clean the monthly groundwater series for one monitoring station."""
from __future__ import annotations

import pandas as pd

from . import config


def load_station_monthly(station: str = config.DEFAULT_STATION,
                         raw_csv=config.RAW_CSV) -> tuple[pd.DataFrame, dict]:
    """Return (monthly_frame, metadata).

    monthly_frame has a monthly-start DatetimeIndex and columns:
      level     mean of the day readings in that month (after interpolation)
      n_obs     number of raw readings averaged in that month (0 if none)
      missing   True where no reading existed and level was interpolated
    """
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

    st = raw[raw["Station Code"] == station].dropna(subset=["date", "value"]).copy()
    if st.empty:
        raise ValueError(f"No valid readings for station {station}")

    meta = {
        "station": station,
        "district": st["District"].iloc[0],
        "block": st["Block"].iloc[0],
        "latitude": float(st["Latitude"].iloc[0]),
        "longitude": float(st["Longitude"].iloc[0]),
        "raw_readings": int(len(st)),
    }

    st["month"] = st["date"].dt.to_period("M").dt.to_timestamp()
    grouped = st.groupby("month")["value"].agg(level="mean", n_obs="size")
    meta["months_with_several_readings"] = int((grouped["n_obs"] > 1).sum())

    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="MS")
    monthly = grouped.reindex(full_index)
    monthly.index.name = "month"
    monthly["n_obs"] = monthly["n_obs"].fillna(0).astype(int)
    monthly["missing"] = monthly["level"].isna()
    monthly["level"] = monthly["level"].interpolate(limit_direction="both")
    meta["missing_months_interpolated"] = int(monthly["missing"].sum())
    meta["first_month"] = str(monthly.index.min().date())
    meta["last_month"] = str(monthly.index.max().date())
    return monthly, meta


def chronological_split(n: int) -> dict[str, slice]:
    """Index slices for train / validation / test. Earlier data always comes first."""
    train_end = int(n * config.TRAIN_FRAC)
    val_end = int(n * (config.TRAIN_FRAC + config.VAL_FRAC))
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }
