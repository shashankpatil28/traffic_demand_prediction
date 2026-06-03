from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID_COL, TARGET_COL


BASELINE_FEATURE_COLUMNS = [
    "base_prev_geo_ts",
    "base_prev_geo_hour",
    "base_prev_geo6_ts",
    "base_prev_geo5_ts",
    "base_prev_geo",
    "base_prev_ts",
    "base_prev_hour",
    "base_global_mean",
    "base_available_count",
    "baseline_pred",
]

BASELINE_WEIGHTS = {
    "base_prev_geo_ts": 0.45,
    "base_prev_geo_hour": 0.20,
    "base_prev_geo5_ts": 0.15,
    "base_prev_ts": 0.10,
    "base_prev_hour": 0.10,
}


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["timestamp"].fillna("0:0").astype(str).str.extract(
        r"(?:^|\s)(?P<hour>\d{1,2}):(?P<minute>\d{1,2})(?::(?P<second>\d{1,2}))?"
    )
    out["hour"] = pd.to_numeric(parts["hour"], errors="coerce").fillna(0).astype(int)
    out["minute"] = pd.to_numeric(parts["minute"], errors="coerce").fillna(0).astype(int)
    out["total_minutes"] = out["hour"] * 60 + out["minute"]
    return out


def add_geohash_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["geohash"] = out["geohash"].fillna("missing").astype(str)
    out["geohash_5"] = out["geohash"].str[:5]
    out["geohash_6"] = out["geohash"].str[:6]
    return out


def prepare_baseline_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_columns(df)
    out = add_geohash_columns(out)
    out["day"] = pd.to_numeric(out["day"], errors="coerce")
    return out


def _map_previous_day_mean(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.Series:
    keys = ["day", *group_cols]
    grouped = train_df.groupby(keys, dropna=False)[TARGET_COL].mean()
    lookup = pd.DataFrame(index=pred_df.index)
    lookup["day"] = pred_df["day"] - 1
    for col in group_cols:
        lookup[col] = pred_df[col]
    index = pd.MultiIndex.from_frame(lookup[keys])
    return pd.Series(grouped.reindex(index).to_numpy(), index=pred_df.index)


def _build_baseline_feature_frame(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    train_work = prepare_baseline_frame(train_df)
    pred_work = prepare_baseline_frame(pred_df)
    global_mean = float(train_work[TARGET_COL].mean())

    features = pd.DataFrame(index=pred_work.index)
    features["base_prev_geo_ts"] = _map_previous_day_mean(
        train_work, pred_work, ["geohash", "timestamp"]
    )
    features["base_prev_geo_hour"] = _map_previous_day_mean(
        train_work, pred_work, ["geohash", "hour"]
    )
    features["base_prev_geo6_ts"] = _map_previous_day_mean(
        train_work, pred_work, ["geohash_6", "timestamp"]
    )
    features["base_prev_geo5_ts"] = _map_previous_day_mean(
        train_work, pred_work, ["geohash_5", "timestamp"]
    )
    features["base_prev_geo"] = _map_previous_day_mean(train_work, pred_work, ["geohash"])
    features["base_prev_ts"] = _map_previous_day_mean(train_work, pred_work, ["timestamp"])
    features["base_prev_hour"] = _map_previous_day_mean(train_work, pred_work, ["hour"])
    features["base_global_mean"] = global_mean

    signal_cols = [col for col in features.columns if col != "base_global_mean"]
    features["base_available_count"] = features[signal_cols].notna().sum(axis=1).astype(float)
    features["baseline_pred"] = _weighted_baseline(features, global_mean)
    return features[BASELINE_FEATURE_COLUMNS].reset_index(drop=True)


def _weighted_baseline(features: pd.DataFrame, global_mean: float) -> np.ndarray:
    values = np.zeros(len(features), dtype=float)
    weights = np.zeros(len(features), dtype=float)

    for col, weight in BASELINE_WEIGHTS.items():
        available = features[col].notna().to_numpy()
        col_values = features[col].fillna(0).to_numpy(dtype=float)
        values += np.where(available, col_values * weight, 0)
        weights += np.where(available, weight, 0)

    return np.divide(
        values,
        weights,
        out=np.full(len(features), global_mean, dtype=float),
        where=weights > 0,
    )


def build_statistical_baseline_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build previous-day baseline features without using prediction targets."""
    train_features = _build_baseline_feature_frame(train_df, train_df)
    test_features = _build_baseline_feature_frame(train_df, test_df)
    return train_features, test_features


def predict_statistical_baseline(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
    return _build_baseline_feature_frame(train_df, pred_df)["baseline_pred"].to_numpy()


def save_statistical_baseline_test(
    test_df: pd.DataFrame,
    baseline_pred: np.ndarray,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "statistical_baseline_test.csv"
    pd.DataFrame({ID_COL: test_df[ID_COL].values, "baseline_pred": baseline_pred}).to_csv(
        path,
        index=False,
    )
    print(f"Saved statistical baseline predictions: {path}")
    return path
