# Traffic Demand Prediction

This project predicts traffic `demand` for each row in `test.csv`.

The task is a supervised tabular regression problem with strong location and time signals. The main model is `CatBoostRegressor`, which works well here because the dataset contains high-cardinality and low-cardinality categorical features such as `geohash`, `RoadType`, `LargeVehicles`, `Landmarks`, and `Weather`.

## Dataset Files

Place the files here:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

Expected shapes:

```text
train.csv: 77,299 rows x 11 columns
test.csv: 41,778 rows x 10 columns
sample_submission.csv: 5 rows x 2 columns
```

The actual downloaded dataset uses the column name `NumberofLanes`. The code also accepts `NumberOfLanes` and normalizes it internally.

## Install Dependencies

From this project folder:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional heavier packages are listed in:

```text
requirements-optional.txt
```

## Run Training

Default full training:

```bash
python main.py
```

Fast smoke test:

```bash
python main.py --quick
```

Current model:

```bash
python main.py --iterations 1500 --output_path outputs/submission_current.csv
```

No day-shift calibration:

```bash
python main.py --iterations 1500 --no_day_shift --output_path outputs/submission_no_day_shift.csv
```

Baseline blend:

```bash
python main.py --iterations 1500 --no_day_shift --blend_baseline --blend_alpha 0.65 --output_path outputs/submission_blend_065.csv
```

Future-window validation:

```bash
python main.py --run_future_validation --iterations 1500 --no_day_shift --blend_baseline --alpha_search
```

Experiment runner:

```bash
python run_experiments.py
```

Hour-calibrated blend:

```bash
python main.py --iterations 1500 --no_day_shift --blend_baseline --blend_alpha 0.65 --hour_calibration --output_path outputs/submission_blend_hour_calibrated.csv
```

Custom paths:

```bash
python main.py \
  --train_path data/train.csv \
  --test_path data/test.csv \
  --sample_path data/sample_submission.csv \
  --output_path outputs/submission.csv
```

## Outputs

The pipeline writes:

```text
outputs/submission.csv
outputs/oof_predictions.csv
outputs/feature_importance.csv
outputs/future_window_validation_summary.csv
outputs/future_window_validation_by_hour.csv
outputs/future_window_validation_predictions.csv
outputs/blend_alpha_search.csv
outputs/experiment_summary.csv
outputs/hour_calibration_table.csv
```

`submission.csv` contains exactly:

```text
Index,demand
```

It is validated before saving.

## Model

Primary model:

```text
CatBoostRegressor
```

Fallback model if CatBoost is unavailable:

```text
sklearn HistGradientBoostingRegressor with ordinal-encoded categorical features
```

CatBoost is recommended for this dataset because it handles categorical features directly and usually performs strongly on medium-sized tabular data.

## Features Created

Timestamp features:

- `hour`
- `minute`
- `second`
- `total_minutes`
- `is_morning_peak`
- `is_evening_peak`
- `is_night`
- `hour_sin`
- `hour_cos`

Day features:

- numeric `day`
- categorical `day_cat`
- `day_sin`
- `day_cos`

Geohash features:

- `geohash_3`
- `geohash_4`
- `geohash_5`
- `geohash_6`
- geohash frequency features

Interaction categorical features:

- `geohash_hour`
- `geohash_day`
- `day_hour`
- `road_lanes`
- `weather_hour`
- `landmark_hour`

## Validation

Random KFold is still available, but it is not the main decision signal for this competition.

Why random KFold is misleading:

- The hidden test is a future time window.
- Train contains day 48 full day and day 49 only from `00:00` to `02:00`.
- Test contains day 49 from `02:15` to `13:45`.
- Random KFold mixes similar historical rows across train and validation, which can make target-stat features look much stronger than they are on the platform.

The main local decision signal should be future-window validation.

Future-window validation simulates the platform by holding out a future window from day 48:

```text
fake observed window: day 48, 00:00 to 02:00
fake future window:   day 48, 02:15 to 13:45
```

Run it with:

```bash
python main.py --run_future_validation --quick --no_day_shift
```

The regular training pipeline uses:

```text
KFold(n_splits=5, shuffle=True, random_state=42)
```

For each fold, it prints R2. At the end it prints:

- mean R2
- standard deviation R2
- competition-like score: `max(0, 100 * mean_r2)`

Treat this random KFold score as a fit/debug signal, not as a leaderboard estimate.

## Statistical Baseline

The deterministic baseline predicts demand from previous-day patterns using this fallback family:

- previous day same `geohash + timestamp`
- previous day same `geohash + hour`
- previous day same `geohash_6 + timestamp`
- previous day same `geohash_5 + timestamp`
- previous day same `geohash`
- previous day same `timestamp`
- previous day same `hour`
- global train mean

The baseline is saved as:

```text
outputs/statistical_baseline_test.csv
```

## CatBoost + Baseline Blending

Enable blending with:

```bash
python main.py --iterations 1500 --no_day_shift --blend_baseline --blend_alpha 0.65 --output_path outputs/submission_blend_065.csv
```

The final prediction is:

```text
final_pred = alpha * catboost_pred + (1 - alpha) * baseline_pred
```

Choose `alpha` using future-window validation, not random KFold:

```bash
python main.py --run_future_validation --quick --no_day_shift --blend_baseline --alpha_search
```

## Hour Calibration

Hour calibration learns conservative correction ratios from fake future-window validation.

It clips raw hour ratios between `0.85` and `1.15`, then applies only 35% of the correction:

```text
smooth_ratio = 1.0 + 0.35 * (clipped_ratio - 1.0)
```

Run:

```bash
python main.py --iterations 1500 --no_day_shift --blend_baseline --blend_alpha 0.65 --hour_calibration --output_path outputs/submission_blend_hour_calibrated.csv
```

## Submission Format

The final file must be:

```text
outputs/submission.csv
```

Required columns:

```text
Index,demand
```

Rules checked by the validator:

- same number of rows as `test.csv`
- exact columns `Index` and `demand`
- no missing predictions
- `Index` values match `test.csv`
- no unnamed extra columns
- `demand` is numeric

## Project Structure

```text
traffic_demand_prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── outputs/
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── statistical_baseline.py
│   ├── future_window_validation.py
│   ├── calibration.py
│   ├── error_analysis.py
│   └── validate_submission.py
├── main.py
├── future_window_validation.py
├── run_experiments.py
├── requirements.txt
└── README.md
```
