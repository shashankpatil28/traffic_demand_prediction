import argparse
from pathlib import Path

import pandas as pd
from src.config import (
    N_SPLITS,
    OOF_PATH,
    OUTPUT_DIR,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSION_PATH,
    TEST_PATH,
    TRAIN_PATH,
)
from src.data_utils import load_data, print_data_checks
from src.features import build_features
from sklearn.metrics import r2_score

from src.calibration import apply_hour_calibration, fit_hour_calibration
from src.future_window_validation import run_future_window_validation
from src.predict import (
    blend_with_baseline,
    clip_predictions,
    create_submission,
    report_baseline_checks,
)
from src.statistical_baseline import (
    add_time_columns,
    predict_statistical_baseline,
    save_statistical_baseline_test,
)
from src.train import make_oof_frame, run_time_holdout_check, train_cv_models
from src.validate_submission import validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic Demand Prediction training pipeline")
    parser.add_argument("--train_path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test_path", type=Path, default=TEST_PATH)
    parser.add_argument("--sample_path", type=Path, default=SAMPLE_SUBMISSION_PATH)
    parser.add_argument("--output_path", type=Path, default=SUBMISSION_PATH)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override CatBoost iterations for non-quick runs.",
    )
    parser.add_argument(
        "--baseline_weight",
        type=float,
        default=0.0,
        help="Weight for blending model predictions with the statistical baseline.",
    )
    parser.add_argument("--blend_baseline", action="store_true")
    parser.add_argument("--blend_alpha", type=float, default=0.65)
    parser.add_argument("--hour_calibration", action="store_true")
    parser.add_argument("--run_future_validation", action="store_true")
    parser.add_argument("--alpha_search", action="store_true")
    parser.add_argument(
        "--time_check",
        action="store_true",
        help="Run an additional day-based holdout diagnostic.",
    )
    parser.add_argument(
        "--no_target_stats",
        action="store_true",
        help="Disable target-derived aggregate and previous-day demand features.",
    )
    parser.add_argument(
        "--no_day_shift",
        action="store_true",
        help="Disable day-49 early calibration ratio features while keeping target stats.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast smoke-test version with fewer folds and iterations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.run_future_validation:
        run_future_window_validation(
            train_path=str(args.train_path),
            output_dir=str(OUTPUT_DIR),
            use_target_stats=not args.no_target_stats,
            use_day_shift=not args.no_day_shift,
            iterations=args.iterations or 1500,
            quick=args.quick,
            blend_baseline=args.blend_baseline,
            blend_alpha=args.blend_alpha,
            alpha_search=args.alpha_search,
            hour_calibration=args.hour_calibration,
        )
        return

    train, test, sample_submission = load_data(
        train_path=args.train_path,
        test_path=args.test_path,
        sample_path=args.sample_path,
    )
    print_data_checks(train, test)

    use_target_stats = not args.no_target_stats
    use_day_shift = not args.no_day_shift
    X, X_test, y, feature_cols, categorical_cols = build_features(
        train,
        test,
        use_target_stats=use_target_stats,
        use_day_shift=use_day_shift,
    )
    report_baseline_checks(X, y, train)

    if args.time_check:
        run_time_holdout_check(
            train=train,
            quick=args.quick,
            use_target_stats=use_target_stats,
            use_day_shift=use_day_shift,
        )

    oof_predictions, test_predictions, _ = train_cv_models(
        X=X,
        y=y,
        X_test=X_test,
        categorical_cols=categorical_cols,
        output_dir=OUTPUT_DIR,
        n_splits=args.n_splits,
        quick=args.quick,
        iterations=args.iterations,
    )

    catboost_test_predictions = test_predictions.copy()
    pd.DataFrame(
        {
            "Index": test["Index"].values,
            "catboost_pred": catboost_test_predictions,
        }
    ).to_csv(OUTPUT_DIR / "catboost_test_predictions.csv", index=False)

    baseline_pred = predict_statistical_baseline(train, test)
    if args.blend_baseline:
        save_statistical_baseline_test(test, baseline_pred, OUTPUT_DIR)

    if args.blend_baseline:
        test_predictions = args.blend_alpha * test_predictions + (1 - args.blend_alpha) * baseline_pred
        pd.DataFrame(
            {
                "Index": test["Index"].values,
                "catboost_pred": catboost_test_predictions,
                "baseline_pred": baseline_pred,
                "blended_pred": test_predictions,
            }
        ).to_csv(OUTPUT_DIR / "blended_test_predictions.csv", index=False)
    elif args.baseline_weight > 0 and "statistical_baseline" in X.columns:
        blended_oof = blend_with_baseline(
            model_predictions=oof_predictions,
            baseline_values=X["statistical_baseline"],
            baseline_weight=args.baseline_weight,
        )
        print(f"Blended OOF R2: {r2_score(y, blended_oof):.6f}")
        oof_predictions = blended_oof

    if args.baseline_weight > 0 and not args.blend_baseline and "statistical_baseline" in X_test.columns:
        test_predictions = blend_with_baseline(
            model_predictions=test_predictions,
            baseline_values=X_test["statistical_baseline"],
            baseline_weight=args.baseline_weight,
        )

    if args.hour_calibration:
        run_future_window_validation(
            train_path=str(args.train_path),
            output_dir=str(OUTPUT_DIR),
            use_target_stats=use_target_stats,
            use_day_shift=use_day_shift,
            iterations=args.iterations or 1500,
            quick=args.quick,
            blend_baseline=args.blend_baseline,
            blend_alpha=args.blend_alpha,
            alpha_search=False,
            hour_calibration=False,
        )
        validation_predictions = pd.read_csv(OUTPUT_DIR / "future_window_validation_predictions.csv")
        calibration_df = fit_hour_calibration(
            validation_predictions.rename(columns={"predicted_demand": "predicted_demand"})[
                ["hour", "actual_demand", "predicted_demand"]
            ]
        )
        calibration_df.to_csv(OUTPUT_DIR / "hour_calibration_table.csv", index=False)
        test_with_hours = add_time_columns(test)
        test_predictions = apply_hour_calibration(
            pd.DataFrame({"hour": test_with_hours["hour"], "raw_pred": test_predictions}),
            calibration_df,
        )
        calibrated_submission = create_submission(test, clip_predictions(test_predictions, y))
        calibrated_path = OUTPUT_DIR / "submission_hour_calibrated.csv"
        calibrated_submission.to_csv(calibrated_path, index=False)
        print(f"Saved hour-calibrated submission: {calibrated_path}")

    test_predictions = clip_predictions(test_predictions, y)
    submission = create_submission(test, test_predictions)
    validate_submission(submission, test, sample_submission)

    submission.to_csv(args.output_path, index=False)
    print(f"Saved submission: {args.output_path}")

    oof_frame = make_oof_frame(train, y, oof_predictions)
    oof_frame.to_csv(OOF_PATH, index=False)
    print(f"Saved OOF predictions: {OOF_PATH}")
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
