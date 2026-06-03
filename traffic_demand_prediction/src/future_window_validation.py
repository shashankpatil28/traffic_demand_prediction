from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .calibration import fit_hour_calibration
from .config import CATBOOST_PARAMS, ID_COL, QUICK_CATBOOST_PARAMS, RANDOM_SEED, TARGET_COL
from .error_analysis import save_error_analysis
from .features import build_features
from .statistical_baseline import predict_statistical_baseline
from .train import _catboost_available, _make_fallback_model


FUTURE_START_MINUTE = 2 * 60 + 15
FUTURE_END_MINUTE = 13 * 60 + 45
ALPHA_VALUES = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80]


def _add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["timestamp"].astype(str).str.extract(
        r"(?:^|\s)(?P<hour>\d{1,2}):(?P<minute>\d{1,2})"
    )
    out["hour"] = pd.to_numeric(parts["hour"], errors="coerce").fillna(0).astype(int)
    out["minute"] = pd.to_numeric(parts["minute"], errors="coerce").fillna(0).astype(int)
    out["total_minutes"] = out["hour"] * 60 + out["minute"]
    out["geohash_5"] = out["geohash"].fillna("missing").astype(str).str[:5]
    return out


def _select_validation_day(train: pd.DataFrame) -> int:
    day_values = sorted(pd.to_numeric(train["day"], errors="coerce").dropna().unique())
    if 48 in day_values:
        return 48
    if len(day_values) < 1:
        raise ValueError("No valid day values found for future-window validation.")
    return int(day_values[0])


def _make_fake_future_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = _add_time_parts(train)
    validation_day = _select_validation_day(working)
    day_mask = pd.to_numeric(working["day"], errors="coerce").eq(validation_day)
    observed_mask = day_mask & working["total_minutes"].between(0, 2 * 60)
    future_mask = day_mask & working["total_minutes"].between(
        FUTURE_START_MINUTE,
        FUTURE_END_MINUTE,
    )

    fake_observed = working.loc[observed_mask].copy()
    fake_valid = working.loc[future_mask].copy()

    # Leakage prevention: fake future target rows are excluded from training and
    # from every lookup table used to build target stats, baselines, and calibration.
    fake_train = fake_observed.copy()

    drop_cols = ["hour", "minute", "total_minutes", "geohash_5"]
    fake_train = fake_train.drop(columns=drop_cols)
    fake_observed = fake_observed.drop(columns=drop_cols)
    fake_valid = fake_valid.drop(columns=drop_cols)

    if fake_train.empty or fake_valid.empty:
        raise ValueError("Could not create fake future-window validation split.")
    return (
        fake_train.reset_index(drop=True),
        fake_observed.reset_index(drop=True),
        fake_valid.reset_index(drop=True),
    )


def _fit_predict_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_cols: list[str],
    iterations: int,
    quick: bool,
    random_seed: int,
) -> np.ndarray:
    if _catboost_available():
        from catboost import CatBoostRegressor

        params = QUICK_CATBOOST_PARAMS if quick else CATBOOST_PARAMS
        params = {**params, "random_seed": random_seed}
        if not quick:
            params = {**params, "iterations": iterations}
        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            cat_features=categorical_cols,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
        )
        return model.predict(X_valid)

    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
    model = _make_fallback_model(categorical_cols, numeric_cols)
    model.fit(X_train, y_train)
    return model.predict(X_valid)


def _metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, pred)),
    }


def _by_hour_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hour, group in predictions.groupby("hour"):
        metrics = _metrics(group["actual_demand"], group["predicted_demand"])
        rows.append(
            {
                "hour": int(hour),
                "count": len(group),
                "r2": metrics["r2"] if len(group) > 1 else np.nan,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "actual_mean": group["actual_demand"].mean(),
                "pred_mean": group["predicted_demand"].mean(),
                "bias": group["predicted_demand"].mean() - group["actual_demand"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("hour")


def _alpha_search(
    y_true: pd.Series,
    catboost_pred: np.ndarray,
    baseline_pred: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for alpha in ALPHA_VALUES:
        blended = alpha * catboost_pred + (1 - alpha) * baseline_pred
        row = {"alpha": alpha, **_metrics(y_true, blended)}
        rows.append(row)
    results = pd.DataFrame(rows).sort_values("r2", ascending=False)
    results.to_csv(output_dir / "blend_alpha_search.csv", index=False)
    print("\nBlend alpha search:")
    print(results.to_string(index=False))
    print(f"Best alpha by future-window validation: {results.iloc[0]['alpha']}")
    return results


def run_future_window_validation(
    train_path: str,
    output_dir: str,
    use_target_stats: bool = True,
    use_day_shift: bool = True,
    iterations: int = 1500,
    random_seed: int = 42,
    quick: bool = False,
    blend_baseline: bool = False,
    blend_alpha: float = 0.65,
    alpha_search: bool = False,
    hour_calibration: bool = False,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    fake_train, fake_observed, fake_valid = _make_fake_future_split(train)
    fake_valid_features = fake_valid.drop(columns=[TARGET_COL]).copy()
    y_valid = fake_valid[TARGET_COL].reset_index(drop=True)

    X_train, X_valid, y_train, _, categorical_cols = build_features(
        fake_train,
        fake_valid_features,
        use_target_stats=use_target_stats,
        use_day_shift=use_day_shift,
    )
    catboost_pred = _fit_predict_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_cols,
        iterations=iterations,
        quick=quick,
        random_seed=random_seed,
    )
    baseline_pred = predict_statistical_baseline(fake_train, fake_valid_features)

    alpha_results = None
    if alpha_search:
        alpha_results = _alpha_search(y_valid, catboost_pred, baseline_pred, output_path)
        blend_alpha = float(alpha_results.iloc[0]["alpha"])

    final_pred = catboost_pred
    if blend_baseline:
        final_pred = blend_alpha * catboost_pred + (1 - blend_alpha) * baseline_pred

    valid_work = _add_time_parts(fake_valid)
    predictions = pd.DataFrame(
        {
            ID_COL: fake_valid[ID_COL].values,
            "day": fake_valid["day"].values,
            "timestamp": fake_valid["timestamp"].values,
            "hour": valid_work["hour"].values,
            "geohash": fake_valid["geohash"].values,
            "geohash_5": valid_work["geohash_5"].values,
            "RoadType": fake_valid["RoadType"].values,
            "Weather": fake_valid["Weather"].values,
            "actual_demand": y_valid.values,
            "catboost_pred": catboost_pred,
            "baseline_pred": baseline_pred,
            "predicted_demand": final_pred,
        }
    )

    if hour_calibration:
        from .calibration import apply_hour_calibration

        calibration_input = predictions[["hour", "actual_demand", "predicted_demand"]]
        calibration_df = fit_hour_calibration(calibration_input)
        calibration_df.to_csv(output_path / "hour_calibration_table.csv", index=False)
        calibrated = apply_hour_calibration(
            pd.DataFrame({"hour": predictions["hour"], "raw_pred": predictions["predicted_demand"]}),
            calibration_df,
        )
        predictions["predicted_demand"] = calibrated

    predictions["error"] = predictions["predicted_demand"] - predictions["actual_demand"]
    predictions["abs_error"] = predictions["error"].abs()

    summary = {
        "train_rows": len(fake_train),
        "fake_observed_rows": len(fake_observed),
        "fake_validation_rows": len(fake_valid),
        "validation_start_minute": FUTURE_START_MINUTE,
        "validation_end_minute": FUTURE_END_MINUTE,
        "use_target_stats": use_target_stats,
        "use_day_shift": use_day_shift,
        "blend_baseline": blend_baseline,
        "blend_alpha": blend_alpha if blend_baseline else np.nan,
        "hour_calibration": hour_calibration,
        **_metrics(y_valid, predictions["predicted_demand"].to_numpy()),
    }

    summary_df = pd.DataFrame([summary])
    by_hour = _by_hour_metrics(predictions)
    summary_df.to_csv(output_path / "future_window_validation_summary.csv", index=False)
    by_hour.to_csv(output_path / "future_window_validation_by_hour.csv", index=False)
    predictions.to_csv(output_path / "future_window_validation_predictions.csv", index=False)
    save_error_analysis(predictions, output_path)

    print("\nFuture-window validation")
    print("------------------------")
    print(f"Train rows: {summary['train_rows']}")
    print(f"Fake observed rows: {summary['fake_observed_rows']}")
    print(f"Fake validation rows: {summary['fake_validation_rows']}")
    print("Validation time range: 02:15 to 13:45")
    print(f"Overall R2: {summary['r2']:.6f}")
    print(f"RMSE: {summary['rmse']:.6f}")
    print(f"MAE: {summary['mae']:.6f}")
    print("\nBy-hour metrics:")
    print(by_hour[["hour", "count", "r2", "mae", "actual_mean", "pred_mean", "bias"]].to_string(index=False))

    return summary
