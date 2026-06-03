import numpy as np
import pandas as pd

from .config import ID_COL, LANE_COL, TARGET_COL
from .statistical_baseline import build_statistical_baseline_features


BASE_CATEGORICAL_COLUMNS = [
    "geohash",
    "timestamp",
    "geohash_3",
    "geohash_4",
    "geohash_5",
    "geohash_6",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "day_cat",
    "hour_cat",
    "geohash_hour",
    "geohash_day",
    "day_hour",
    "road_lanes",
    "weather_hour",
    "landmark_hour",
]

TARGET_AGGREGATION_SPECS = [
    ("stat_geohash_mean", ["geohash"]),
    ("stat_geohash_5_mean", ["geohash_5"]),
    ("stat_geohash_6_mean", ["geohash_6"]),
    ("stat_timestamp_mean", ["timestamp"]),
    ("stat_hour_mean", ["hour"]),
    ("stat_geohash_timestamp_mean", ["geohash", "timestamp"]),
    ("stat_geohash_hour_mean", ["geohash", "hour"]),
    ("stat_geohash5_timestamp_mean", ["geohash_5", "timestamp"]),
    ("stat_day_hour_mean", ["day_cat", "hour"]),
    ("stat_road_lanes_mean", ["RoadType", LANE_COL]),
    ("stat_weather_hour_mean", ["Weather", "hour"]),
    ("stat_landmark_hour_mean", ["Landmarks", "hour"]),
]

PREVIOUS_DAY_SPECS = [
    ("prev_day_geohash_timestamp_demand", ["day", "geohash", "timestamp"]),
    ("prev_day_geohash_hour_demand", ["day", "geohash", "hour"]),
    ("prev_day_geohash_demand", ["day", "geohash"]),
    ("prev_day_geohash5_timestamp_demand", ["day", "geohash_5", "timestamp"]),
    ("prev_day_timestamp_demand", ["day", "timestamp"]),
    ("prev_day_hour_demand", ["day", "hour"]),
]


def _extract_time_parts(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    values = series.fillna("0:0").astype(str)
    parts = values.str.extract(
        r"(?:^|\s)(?P<hour>\d{1,2}):(?P<minute>\d{1,2})(?::(?P<second>\d{1,2}))?"
    )

    hour = pd.to_numeric(parts["hour"], errors="coerce")
    minute = pd.to_numeric(parts["minute"], errors="coerce")
    second = pd.to_numeric(parts["second"], errors="coerce")

    missing_mask = hour.isna()
    if missing_mask.any():
        parsed = pd.to_datetime(values[missing_mask], errors="coerce")
        hour.loc[missing_mask] = parsed.dt.hour
        minute.loc[missing_mask] = parsed.dt.minute
        second.loc[missing_mask] = parsed.dt.second

    hour = hour.fillna(0).astype(int).clip(0, 23)
    minute = minute.fillna(0).astype(int).clip(0, 59)
    second = second.fillna(0).astype(int).clip(0, 59)
    return hour, minute, second


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hour, minute, second = _extract_time_parts(df["timestamp"])
    df["hour"] = hour
    df["minute"] = minute
    df["second"] = second
    df["total_minutes"] = df["hour"] * 60 + df["minute"]
    df["is_morning_peak"] = df["hour"].between(7, 10).astype(int)
    df["is_evening_peak"] = df["hour"].between(17, 21).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["hour_cat"] = df["hour"].astype(str)
    return df


def _add_day_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    df["day"] = df["day"].fillna(df["day"].median())
    max_day = int(df["day"].max()) if not df["day"].isna().all() else 7
    period = 7 if 1 <= max_day <= 7 else max(max_day, 7)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / period)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / period)
    df["day_cat"] = df["day"].astype(int).astype(str)
    return df


def _add_geohash_features(df: pd.DataFrame) -> pd.DataFrame:
    df["geohash"] = df["geohash"].fillna("missing").astype(str)
    for length in [3, 4, 5, 6]:
        df[f"geohash_{length}"] = df["geohash"].str[:length]
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["geohash_hour"] = df["geohash"].astype(str) + "_" + df["hour"].astype(str)
    df["geohash_day"] = df["geohash"].astype(str) + "_" + df["day_cat"].astype(str)
    df["day_hour"] = df["day_cat"].astype(str) + "_" + df["hour"].astype(str)
    df["road_lanes"] = df["RoadType"].astype(str) + "_" + df[LANE_COL].astype(str)
    df["weather_hour"] = df["Weather"].astype(str) + "_" + df["hour"].astype(str)
    df["landmark_hour"] = df["Landmarks"].astype(str) + "_" + df["hour"].astype(str)
    return df


def _add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["geohash", "geohash_4", "geohash_5", "geohash_6"]:
        freq = df[col].value_counts(dropna=False)
        df[f"{col}_freq"] = df[col].map(freq).fillna(0).astype(float)
    return df


def _add_target_aggregation_features(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    train_part = df.loc[train_mask].copy()
    global_mean = float(train_part[TARGET_COL].mean())

    for feature_name, keys in TARGET_AGGREGATION_SPECS:
        agg = (
            train_part.groupby(keys, dropna=False)[TARGET_COL]
            .agg(["sum", "count"])
            .reset_index()
        )
        agg = agg.rename(
            columns={
                "sum": f"_{feature_name}_sum",
                "count": f"{feature_name}_count",
            }
        )
        df = df.merge(agg, on=keys, how="left")

        sum_col = f"_{feature_name}_sum"
        count_col = f"{feature_name}_count"
        mean_values = df[sum_col] / df[count_col]

        loo_values = (df[sum_col] - df[TARGET_COL]) / (df[count_col] - 1)
        df[feature_name] = mean_values
        df.loc[train_mask, feature_name] = np.where(
            df.loc[train_mask, count_col] > 1,
            loo_values.loc[train_mask],
            global_mean,
        )

        df[feature_name] = df[feature_name].fillna(global_mean)
        df[count_col] = df[count_col].fillna(0).astype(float)
        df = df.drop(columns=[sum_col])

    return df


def _map_previous_day_mean(
    df: pd.DataFrame,
    train_part: pd.DataFrame,
    keys: list[str],
) -> pd.Series:
    grouped = train_part.groupby(keys, dropna=False)[TARGET_COL].mean()
    lookup = pd.DataFrame(index=df.index)
    for key in keys:
        lookup[key] = df["day"] - 1 if key == "day" else df[key]
    lookup_keys = pd.MultiIndex.from_frame(lookup[keys])
    return pd.Series(grouped.reindex(lookup_keys).to_numpy(), index=df.index)


def _add_previous_day_features(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    train_part = df.loc[train_mask].copy()
    for feature_name, keys in PREVIOUS_DAY_SPECS:
        df[feature_name] = _map_previous_day_mean(df, train_part, keys)
    return df


def _safe_ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return default
    return float(np.clip(numerator / denominator, 0.5, 1.5))


def _add_day_shift_features(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    train_part = df.loc[train_mask].copy()
    unique_days = sorted(train_part["day"].dropna().unique())
    if len(unique_days) < 2:
        df["day_shift_global_ratio"] = 1.0
        return df

    previous_day = unique_days[-2]
    current_day = unique_days[-1]
    current_slots = set(train_part.loc[train_part["day"].eq(current_day), "timestamp"])
    previous = train_part[
        train_part["day"].eq(previous_day) & train_part["timestamp"].isin(current_slots)
    ]
    current = train_part[train_part["day"].eq(current_day)]

    global_ratio = _safe_ratio(current[TARGET_COL].mean(), previous[TARGET_COL].mean())
    df["day_shift_global_ratio"] = global_ratio

    for group_col in ["geohash", "geohash_5", "timestamp", "hour"]:
        prev_mean = previous.groupby(group_col, dropna=False)[TARGET_COL].mean()
        curr_mean = current.groupby(group_col, dropna=False)[TARGET_COL].mean()
        ratio = (curr_mean / prev_mean).replace([np.inf, -np.inf], np.nan).clip(0.5, 1.5)
        feature_name = f"day_shift_{group_col}_ratio"
        df[feature_name] = df[group_col].map(ratio).fillna(global_ratio).astype(float)

    if "prev_day_geohash_timestamp_demand" in df.columns:
        df["calibrated_prev_day_geohash_timestamp_demand"] = (
            df["prev_day_geohash_timestamp_demand"] * df["day_shift_geohash_ratio"]
        )
    return df


def _add_statistical_baseline(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    global_mean = float(df.loc[train_mask, TARGET_COL].mean())
    fallback_columns = [
        "calibrated_prev_day_geohash_timestamp_demand",
        "prev_day_geohash_timestamp_demand",
        "prev_day_geohash_hour_demand",
        "prev_day_geohash_demand",
        "stat_geohash_timestamp_mean",
        "stat_geohash_hour_mean",
        "stat_geohash_mean",
        "stat_geohash_5_mean",
        "stat_timestamp_mean",
        "stat_hour_mean",
    ]

    baseline = pd.Series(np.nan, index=df.index, dtype=float)
    for col in fallback_columns:
        if col in df.columns:
            baseline = baseline.where(baseline.notna(), df[col])
    df["statistical_baseline"] = baseline.fillna(global_mean)
    return df


def _clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather", "timestamp"]:
        df[col] = df[col].fillna("missing").astype(str)

    df[LANE_COL] = pd.to_numeric(df[LANE_COL], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")

    for col in [LANE_COL, "Temperature"]:
        df[col] = df[col].fillna(df[col].median())

    return df


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    use_target_stats: bool = True,
    use_day_shift: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], list[str]]:
    print("\nBuilding features...")
    original_train_rows = len(train)
    original_test_rows = len(test)
    y = train[TARGET_COL].copy()

    train_features = train.copy()
    test_features = test.copy()
    test_features[TARGET_COL] = np.nan
    train_features["_is_train"] = 1
    test_features["_is_train"] = 0

    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    train_mask = combined["_is_train"].eq(1)
    combined = _clean_missing_values(combined)
    combined = _add_time_features(combined)
    combined = _add_day_features(combined)
    combined = _add_geohash_features(combined)
    combined = _add_interaction_features(combined)
    combined = _add_frequency_features(combined)
    if use_target_stats:
        combined = _add_target_aggregation_features(combined, train_mask)
        combined = _add_previous_day_features(combined, train_mask)
        if use_day_shift:
            combined = _add_day_shift_features(combined, train_mask)
        combined = _add_statistical_baseline(combined, train_mask)

    categorical_cols = [col for col in BASE_CATEGORICAL_COLUMNS if col in combined.columns]
    for col in categorical_cols:
        combined[col] = combined[col].fillna("missing").astype(str)

    numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in [ID_COL, "_is_train"]]
    for col in numeric_cols:
        combined[col] = combined[col].replace([np.inf, -np.inf], np.nan)
        combined[col] = combined[col].fillna(combined[col].median())

    drop_cols = [ID_COL, "_is_train", TARGET_COL]
    feature_cols = [col for col in combined.columns if col not in drop_cols]

    train_processed = combined.loc[combined["_is_train"].eq(1), feature_cols].reset_index(drop=True)
    test_processed = combined.loc[combined["_is_train"].eq(0), feature_cols].reset_index(drop=True)

    if use_target_stats:
        baseline_train, baseline_test = build_statistical_baseline_features(train, test)
        train_processed = pd.concat([train_processed, baseline_train], axis=1)
        test_processed = pd.concat([test_processed, baseline_test], axis=1)
        feature_cols = train_processed.columns.tolist()

    assert len(train_processed) == original_train_rows
    assert len(test_processed) == original_test_rows
    assert not y.isna().any()

    categorical_cols = [col for col in categorical_cols if col in feature_cols]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Final feature count: {len(feature_cols)}")
    print(f"Train feature shape: {train_processed.shape}")
    print(f"Test feature shape: {test_processed.shape}")

    return train_processed, test_processed, y, feature_cols, categorical_cols
