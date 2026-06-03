import pandas as pd

from .config import ID_COL, REQUIRED_SUBMISSION_COLUMNS, TARGET_COL


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    print("\nValidating submission...")
    if submission.shape != (len(test), 2):
        raise ValueError(
            f"Submission shape is {submission.shape}, expected {(len(test), 2)}."
        )

    if list(submission.columns) != REQUIRED_SUBMISSION_COLUMNS:
        raise ValueError(
            f"Submission columns are {list(submission.columns)}, expected {REQUIRED_SUBMISSION_COLUMNS}."
        )

    unnamed_cols = [col for col in submission.columns if str(col).startswith("Unnamed")]
    if unnamed_cols:
        raise ValueError(f"Submission contains unnamed columns: {unnamed_cols}")

    if not submission[ID_COL].equals(test[ID_COL]):
        raise ValueError("Submission Index column does not exactly match test Index column.")

    if submission[TARGET_COL].isna().any():
        raise ValueError("Submission demand column contains missing values.")

    if not pd.api.types.is_numeric_dtype(submission[TARGET_COL]):
        raise ValueError("Submission demand column must be numeric.")

    if list(sample_submission.columns) != REQUIRED_SUBMISSION_COLUMNS:
        raise ValueError("Sample submission columns do not match expected format.")

    print("Submission validation passed.")
