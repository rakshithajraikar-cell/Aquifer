# Testing documentation

## What is tested

Run the unit tests from the project root:

```bash
pytest -q tests
```

| Test | Property checked | Why it matters |
|---|---|---|
| `test_features_do_not_use_current_value` | Changing the level at month *t* does not change any feature describing month *t* | Prevents target leakage, which would inflate every model's accuracy |
| `test_chronological_split_is_ordered_and_disjoint` | Train, validation and test blocks are contiguous, ordered and cover the series | The brief forbids shuffling time-series data |
| `test_seasonal_naive_repeats_last_year` | Seasonal naive reproduces the last 12 months, then repeats them | Confirms the baseline is implemented as defined |
| `test_naive_forecast_is_last_value` | Naive forecast equals the last observed value | Baseline sanity check |
| `test_tree_forecaster_is_recursive_and_finite` | Recursive multi-step forecasts are finite and have the requested length | Multi-horizon forecasts must not produce NaN or truncated output |
| `test_risk_index_bounded_and_classes` | Risk Index lies in 0–100, has a valid class, and weights sum to 1 | Risk score is reported to users as a 0–100 index |
| `test_risk_class_edges` | Class boundaries are inclusive on the upper side | Ensures a score on an edge is classified consistently |
| `test_scenarios_reduce_depth_when_extraction_is_cut` | Larger extraction cuts give shallower (lower-depth) curves; 0% cut equals the baseline | Verifies the direction of the what-if effect |
| `test_recovery_events_on_synthetic_monsoon_cycle` | Recovery detection returns the known rise on a constructed series | Checks the recovery logic against a case with a known answer |
| `test_regression_metrics_known_values` | MAE and RMSE match hand-computed values | Metric correctness |

## Validation checks performed on the real data

These were run once during development. Re-run `python run_pipeline.py` to reproduce them.

- **Time-based split**: training ends 2006-03, validation 2006-04 to 2013-07, test 2013-08 to 2020-12. No shuffling.
- **Model selection on validation only**: test data is not used to choose a model. It is scored once, after refitting on train+validation.
- **Multi-horizon evaluation**: forecasts use the model's own predictions for up to 12 months, which is how a deployed system works.
- **Missing months**: none in the selected station, so nothing needed interpolation. The pipeline would flag any it does interpolate.
- **Repeated values**: the flat run of 15.0 (Dec 1989 to Jul 1991, 20 months) is flagged as a possible sensor or data issue.

## Not yet tested

- The React frontend (the dashboard is plain HTML and JavaScript, checked by loading the API, not by automated browser tests).
- Load testing of the API; it reads small JSON and CSV files, so no performance test was run.
- Multi-station runs; the pipeline models one station at a time.
