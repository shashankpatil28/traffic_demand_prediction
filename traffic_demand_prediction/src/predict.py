import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .config import ID_COL, TARGET_COL


def clip_predictions(predictions: np.ndarray, y: pd.Series) -> np.ndarray:
    y_min = float(y.min())
    y_max = float(y.max())
    print(f"\nClipping predictions to target range: [{y_min:.6f}, {y_max:.6f}]")
    return np.clip(predictions, y_min, y_max)


def create_submission(test: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    if len(predictions) != len(test):
        raise ValueError(
            f"Prediction length mismatch. Got {len(predictions)}, expected {len(test)}."
        )
    if np.isnan(predictions).any():
        raise ValueError("Predictions contain NaN values.")

    return pd.DataFrame(
        {
            ID_COL: test[ID_COL].values,
            TARGET_COL: predictions,
        }
    )


def blend_with_baseline(
    model_predictions: np.ndarray,
    baseline_values: pd.Series | np.ndarray | None,
    baseline_weight: float,
) -> np.ndarray:
    if baseline_values is None or baseline_weight <= 0:
        return model_predictions

    baseline_weight = float(np.clip(baseline_weight, 0, 1))
    baseline = np.asarray(baseline_values, dtype=float)
    if len(baseline) != len(model_predictions):
        raise ValueError("Baseline length does not match model prediction length.")
    if np.isnan(baseline).any():
        raise ValueError("Baseline contains NaN values.")

    print(f"\nBlending predictions: model={1 - baseline_weight:.2f}, baseline={baseline_weight:.2f}")
    return (1 - baseline_weight) * model_predictions + baseline_weight * baseline


def report_baseline_checks(X: pd.DataFrame, y: pd.Series, train: pd.DataFrame) -> None:
    if "statistical_baseline" not in X.columns:
        return

    baseline = X["statistical_baseline"].to_numpy()
    print("\nStatistical baseline checks")
    print(f"Overall baseline R2: {r2_score(y, baseline):.6f}")

    day_numeric = pd.to_numeric(train["day"], errors="coerce")
    max_day = day_numeric.max()
    max_day_mask = day_numeric.eq(max_day)
    if max_day_mask.sum() > 1:
        print(
            f"Latest-day baseline R2 on day {int(max_day)}: "
            f"{r2_score(y.loc[max_day_mask], baseline[max_day_mask]):.6f}"
        )
