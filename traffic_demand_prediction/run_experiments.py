from pathlib import Path

import pandas as pd

from src.config import OUTPUT_DIR, TRAIN_PATH
from src.future_window_validation import _make_fake_future_split, run_future_window_validation
from src.statistical_baseline import predict_statistical_baseline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _baseline_only_experiment(train: pd.DataFrame) -> dict:
    fake_train, _, fake_valid = _make_fake_future_split(train)
    valid_features = fake_valid.drop(columns=["demand"]).copy()
    pred = predict_statistical_baseline(fake_train, valid_features)
    y = fake_valid["demand"]
    return {
        "experiment_name": "baseline_only",
        "use_target_stats": False,
        "use_day_shift": False,
        "blend_baseline": False,
        "blend_alpha": None,
        "hour_calibration": False,
        "future_window_r2": r2_score(y, pred),
        "future_window_rmse": mean_squared_error(y, pred) ** 0.5,
        "future_window_mae": mean_absolute_error(y, pred),
        "notes": "deterministic previous-day baseline only",
    }


def main() -> None:
    output_dir = OUTPUT_DIR
    experiment_root = output_dir / "experiments"
    experiment_root.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    rows = []

    experiments = [
        {
            "experiment_name": "current_full",
            "use_target_stats": True,
            "use_day_shift": True,
            "blend_baseline": False,
            "blend_alpha": 0.65,
            "hour_calibration": False,
            "notes": "current target stats plus day shift",
        },
        {
            "experiment_name": "no_day_shift",
            "use_target_stats": True,
            "use_day_shift": False,
            "blend_baseline": False,
            "blend_alpha": 0.65,
            "hour_calibration": False,
            "notes": "target stats without day-shift ratios",
        },
        {
            "experiment_name": "no_target_stats",
            "use_target_stats": False,
            "use_day_shift": False,
            "blend_baseline": False,
            "blend_alpha": 0.65,
            "hour_calibration": False,
            "notes": "base categorical/time/geohash/frequency features only",
        },
        {
            "experiment_name": "catboost_baseline_blend",
            "use_target_stats": True,
            "use_day_shift": False,
            "blend_baseline": True,
            "blend_alpha": 0.65,
            "hour_calibration": False,
            "notes": "CatBoost plus deterministic baseline blend",
        },
        {
            "experiment_name": "blend_hour_calibrated",
            "use_target_stats": True,
            "use_day_shift": False,
            "blend_baseline": True,
            "blend_alpha": 0.65,
            "hour_calibration": True,
            "notes": "blend plus conservative hour calibration",
        },
    ]

    print("Running future-window experiment suite in quick mode.")
    print("Use main.py commands for full final submissions.")
    rows.append(_baseline_only_experiment(train))

    for exp in experiments:
        print(f"\nExperiment runner: {exp['experiment_name']}")
        exp_dir = experiment_root / exp["experiment_name"]
        summary = run_future_window_validation(
            train_path=str(TRAIN_PATH),
            output_dir=str(exp_dir),
            use_target_stats=exp["use_target_stats"],
            use_day_shift=exp["use_day_shift"],
            iterations=1500,
            quick=True,
            blend_baseline=exp["blend_baseline"],
            blend_alpha=exp["blend_alpha"],
            alpha_search=exp["blend_baseline"],
            hour_calibration=exp["hour_calibration"],
        )
        rows.append(
            {
                "experiment_name": exp["experiment_name"],
                "use_target_stats": exp["use_target_stats"],
                "use_day_shift": exp["use_day_shift"],
                "blend_baseline": exp["blend_baseline"],
                "blend_alpha": summary.get("blend_alpha"),
                "hour_calibration": exp["hour_calibration"],
                "future_window_r2": summary["r2"],
                "future_window_rmse": summary["rmse"],
                "future_window_mae": summary["mae"],
                "notes": exp["notes"],
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("future_window_r2", ascending=False)
    summary_path = output_dir / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\nExperiment summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved experiment summary: {summary_path}")


if __name__ == "__main__":
    main()
