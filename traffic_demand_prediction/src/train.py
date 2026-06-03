import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .config import (
    CATBOOST_PARAMS,
    FEATURE_IMPORTANCE_PATH,
    ID_COL,
    QUICK_CATBOOST_PARAMS,
    RANDOM_SEED,
    TARGET_COL,
)


def _catboost_available() -> bool:
    try:
        import catboost  # noqa: F401

        return True
    except ImportError:
        return False


def _make_fallback_model(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", SimpleImputer(strategy="median"), numeric_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=500,
        random_state=RANDOM_SEED,
        l2_regularization=0.05,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def train_cv_models(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_cols: list[str],
    output_dir,
    n_splits: int = 5,
    quick: bool = False,
    iterations: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    print("\nTraining model with cross validation...")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_splits = min(n_splits, len(X))
    if quick:
        print("Quick mode enabled: using 2 folds and fewer CatBoost iterations.")
        n_splits = min(2, n_splits)

    oof_predictions = np.zeros(len(X), dtype=float)
    test_predictions = np.zeros(len(X_test), dtype=float)
    fold_scores: list[float] = []
    feature_importance_frames: list[pd.DataFrame] = []

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    use_catboost = _catboost_available()
    print(f"Model backend: {'CatBoostRegressor' if use_catboost else 'sklearn fallback'}")

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X, y), start=1):
        print(f"\nFold {fold}/{n_splits}")
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        if use_catboost:
            from catboost import CatBoostRegressor

            params = QUICK_CATBOOST_PARAMS if quick else CATBOOST_PARAMS
            if iterations is not None and not quick:
                params = {**params, "iterations": iterations}
            model = CatBoostRegressor(**params)
            model.fit(
                X_train,
                y_train,
                cat_features=categorical_cols,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
            )
            valid_pred = model.predict(X_valid)
            fold_test_pred = model.predict(X_test)

            importance = pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": model.get_feature_importance(),
                    "fold": fold,
                }
            )
            feature_importance_frames.append(importance)
        else:
            numeric_cols = [col for col in X.columns if col not in categorical_cols]
            model = _make_fallback_model(categorical_cols, numeric_cols)
            model.fit(X_train, y_train)
            valid_pred = model.predict(X_valid)
            fold_test_pred = model.predict(X_test)

        score = r2_score(y_valid, valid_pred)
        fold_scores.append(score)
        oof_predictions[valid_idx] = valid_pred
        test_predictions += fold_test_pred / n_splits
        print(f"Fold {fold} R2: {score:.6f}")

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    competition_score = max(0.0, 100 * mean_score)
    print("\nCross-validation summary")
    print(f"Mean R2: {mean_score:.6f}")
    print(f"Std R2: {std_score:.6f}")
    print(f"Competition-like score: {competition_score:.4f}")

    if feature_importance_frames:
        feature_importance = (
            pd.concat(feature_importance_frames, ignore_index=True)
            .groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
        feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        print(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")

    return oof_predictions, test_predictions, fold_scores


def run_time_holdout_check(
    train: pd.DataFrame,
    quick: bool = True,
    use_target_stats: bool = True,
    use_day_shift: bool = True,
) -> None:
    print("\nRunning time-aware holdout diagnostic...")
    day_numeric = pd.to_numeric(train["day"], errors="coerce")
    max_day = day_numeric.max()
    train_mask = day_numeric.lt(max_day)
    valid_mask = day_numeric.eq(max_day)

    if train_mask.sum() == 0 or valid_mask.sum() < 100:
        print("Skipped time-aware holdout: not enough day structure.")
        return

    from .features import build_features

    holdout_train = train.loc[train_mask].copy()
    holdout_valid = train.loc[valid_mask].copy()
    holdout_valid_features = holdout_valid.drop(columns=[TARGET_COL]).copy()
    X_train, X_valid, y_train, _, holdout_categorical_cols = build_features(
        holdout_train,
        holdout_valid_features,
        use_target_stats=use_target_stats,
        use_day_shift=use_day_shift,
    )
    y_valid = holdout_valid[TARGET_COL].reset_index(drop=True)

    if _catboost_available():
        from catboost import CatBoostRegressor

        params = QUICK_CATBOOST_PARAMS if quick else {**CATBOOST_PARAMS, "iterations": 1000}
        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            cat_features=holdout_categorical_cols,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
        )
        valid_pred = model.predict(X_valid)
    else:
        numeric_cols = [col for col in X_train.columns if col not in holdout_categorical_cols]
        model = _make_fallback_model(holdout_categorical_cols, numeric_cols)
        model.fit(X_train, y_train)
        valid_pred = model.predict(X_valid)

    model_score = r2_score(y_valid, valid_pred)
    print(f"Time-aware holdout R2 on day {int(max_day)}: {model_score:.6f}")
    if "statistical_baseline" in X_valid.columns:
        baseline_score = r2_score(y_valid, X_valid["statistical_baseline"])
        print(f"Time-aware baseline R2 on day {int(max_day)}: {baseline_score:.6f}")


def make_oof_frame(train: pd.DataFrame, y: pd.Series, oof_predictions: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL: train[ID_COL].values,
            "actual_demand": y.values,
            "predicted_demand": oof_predictions,
        }
    )
