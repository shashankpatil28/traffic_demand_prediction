from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_r2(group: pd.DataFrame) -> float:
    if len(group) < 2 or group["actual_demand"].nunique() <= 1:
        return np.nan
    return float(r2_score(group["actual_demand"], group["predicted_demand"]))


def _summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False):
        actual = group["actual_demand"]
        pred = group["predicted_demand"]
        rows.append(
            {
                group_col: value,
                "count": len(group),
                "actual_mean": actual.mean(),
                "pred_mean": pred.mean(),
                "bias": pred.mean() - actual.mean(),
                "mae": mean_absolute_error(actual, pred),
                "rmse": mean_squared_error(actual, pred) ** 0.5,
                "r2": _safe_r2(group),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def save_error_analysis(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"actual_demand", "predicted_demand"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Missing error analysis columns: {sorted(missing)}")

    report_specs = [
        ("hour", "error_by_hour.csv"),
        ("Weather", "error_by_weather.csv"),
        ("RoadType", "error_by_roadtype.csv"),
    ]
    for group_col, filename in report_specs:
        if group_col in predictions_df.columns:
            _summary(predictions_df, group_col).to_csv(output_dir / filename, index=False)

    if "geohash_5" in predictions_df.columns:
        geo_report = _summary(predictions_df, "geohash_5").head(50)
        geo_report.to_csv(output_dir / "error_by_geohash5_top50.csv", index=False)

    if "hour" in predictions_df.columns:
        _summary(predictions_df, "hour").to_csv(
            output_dir / "prediction_bias_by_hour.csv",
            index=False,
        )

    print(f"Saved error analysis reports in: {output_dir}")
