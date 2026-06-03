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

The training pipeline uses:

```text
KFold(n_splits=5, shuffle=True, random_state=42)
```

For each fold, it prints R2. At the end it prints:

- mean R2
- standard deviation R2
- competition-like score: `max(0, 100 * mean_r2)`

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
│   └── validate_submission.py
├── main.py
├── requirements.txt
└── README.md
```
