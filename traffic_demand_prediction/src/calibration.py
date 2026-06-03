import numpy as np
import pandas as pd


def fit_hour_calibration(validation_predictions_df: pd.DataFrame) -> pd.DataFrame:
    required = {"hour", "actual_demand", "predicted_demand"}
    missing = required - set(validation_predictions_df.columns)
    if missing:
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")

    rows = []
    global_count = max(len(validation_predictions_df), 1)
    for hour, group in validation_predictions_df.groupby("hour"):
        actual_mean = float(group["actual_demand"].mean())
        pred_mean = float(group["predicted_demand"].mean())
        raw_ratio = actual_mean / pred_mean if pred_mean > 0 else 1.0
        clipped_ratio = float(np.clip(raw_ratio, 0.85, 1.15))
        count = len(group)
        count_shrink = min(1.0, count / max(500.0, global_count / 48))
        conservative_ratio = 1.0 + 0.35 * (clipped_ratio - 1.0)
        smooth_ratio = 1.0 + count_shrink * (conservative_ratio - 1.0)
        rows.append(
            {
                "hour": int(hour),
                "count": count,
                "actual_mean": actual_mean,
                "pred_mean": pred_mean,
                "raw_ratio": raw_ratio,
                "clipped_ratio": clipped_ratio,
                "smooth_ratio": smooth_ratio,
            }
        )

    return pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)


def apply_hour_calibration(pred_df: pd.DataFrame, calibration_df: pd.DataFrame) -> np.ndarray:
    required = {"hour", "raw_pred"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns for calibration: {sorted(missing)}")

    ratio_map = calibration_df.set_index("hour")["smooth_ratio"]
    ratios = pred_df["hour"].map(ratio_map).fillna(1.0).to_numpy(dtype=float)
    return pred_df["raw_pred"].to_numpy(dtype=float) * ratios
