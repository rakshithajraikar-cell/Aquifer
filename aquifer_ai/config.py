"""Central configuration for the Aquifer decision-support system.

Values marked ASSUMPTION are project choices, not findings. Change them here,
then re-run `python run_pipeline.py` so every report is regenerated.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

RAW_CSV = DATA_DIR / "karnataka_groundwater.csv"
DEFAULT_STATION = "70202"  # Chitradurga, Karnataka (see reports/data_investigation_report.txt)

# ASSUMPTION: the dataset does not state its unit. The CGWB convention for
# this kind of data is depth to water level in metres below ground level, so
# a HIGHER number means a DEEPER (worse) water table. Verify before presenting.
LEVEL_IS_DEPTH = True

# Chronological split. Nothing is shuffled. Test data is only used once,
# for the final report; model selection uses the validation block.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test = remaining 15%

MAX_HORIZON = 12  # months; the data is monthly so 7-day and 30-day horizons are not possible
HORIZONS_REPORTED = (1, 3, 6, 12)

LAGS = (1, 2, 3, 6, 12)
LOOKBACK = 24  # months of input for LSTM / GRU
SEED = 42

# Risk index weights (components that have data). Rainfall and extraction
# components are not computed because those series are not in the dataset.
RISK_WEIGHTS = {"level": 0.35, "decline": 0.25, "seasonal": 0.15, "recharge": 0.25}
RISK_CLASS_EDGES = (25, 50, 75)  # Low < 25 <= Moderate < 50 <= High < 75 <= Critical
CRITICAL_PERCENTILE = 90  # ASSUMPTION: depth above this historical percentile = critical

# Aquifer Health Score weights. Extraction and rainfall are not available, so
# their weights are dropped and the rest are renormalised.
HEALTH_WEIGHTS = {
    "current_level": 0.25,
    "long_term_trend": 0.20,
    "recharge": 0.15,
    "anomaly_frequency": 0.10,
    "forecast_condition": 0.30,
}
# ASSUMPTION: long-term deepening rate that maps to a health score of 0.
TREND_WORST_M_PER_DECADE = 1.0

# What-if simulator. ASSUMPTION: share of the observed long-term deepening
# attributed to groundwater extraction. No extraction data exists, so this is
# a user-adjustable parameter, not a fitted coefficient.
EXTRACTION_SHARE_OF_TREND = 0.5
