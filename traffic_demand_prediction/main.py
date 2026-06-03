import argparse
from pathlib import Path

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

from src.predict import (
    blend_with_baseline,
    clip_predictions,
    create_submission,
    report_baseline_checks,
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
        "--baseline_weight",
        type=float,
        default=0.0,
        help="Weight for blending model predictions with the statistical baseline.",
    )
    parser.add_argument(
        "--time_check",
        action="store_true",
        help="Run an additional day-based holdout diagnostic.",
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

    train, test, sample_submission = load_data(
        train_path=args.train_path,
        test_path=args.test_path,
        sample_path=args.sample_path,
    )
    print_data_checks(train, test)

    X, X_test, y, feature_cols, categorical_cols = build_features(train, test)
    report_baseline_checks(X, y, train)

    if args.time_check:
        run_time_holdout_check(
            train=train,
            quick=args.quick,
        )

    oof_predictions, test_predictions, _ = train_cv_models(
        X=X,
        y=y,
        X_test=X_test,
        categorical_cols=categorical_cols,
        output_dir=OUTPUT_DIR,
        n_splits=args.n_splits,
        quick=args.quick,
    )

    if args.baseline_weight > 0 and "statistical_baseline" in X.columns:
        blended_oof = blend_with_baseline(
            model_predictions=oof_predictions,
            baseline_values=X["statistical_baseline"],
            baseline_weight=args.baseline_weight,
        )
        print(f"Blended OOF R2: {r2_score(y, blended_oof):.6f}")
        oof_predictions = blended_oof

    if args.baseline_weight > 0 and "statistical_baseline" in X_test.columns:
        test_predictions = blend_with_baseline(
            model_predictions=test_predictions,
            baseline_values=X_test["statistical_baseline"],
            baseline_weight=args.baseline_weight,
        )

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
