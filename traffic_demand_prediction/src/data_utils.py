from pathlib import Path

import pandas as pd

from .config import (
    ID_COL,
    LANE_COL,
    REQUIRED_SUBMISSION_COLUMNS,
    REQUIRED_TEST_COLUMNS,
    REQUIRED_TRAIN_COLUMNS,
    TARGET_COL,
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "NumberOfLanes" in df.columns and LANE_COL not in df.columns:
        df = df.rename(columns={"NumberOfLanes": LANE_COL})
    return df


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def _require_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_data(
    train_path: Path,
    test_path: Path,
    sample_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading data...")
    for path in [train_path, test_path, sample_path]:
        _require_file(path)

    train = _normalize_columns(pd.read_csv(train_path))
    test = _normalize_columns(pd.read_csv(test_path))
    sample_submission = pd.read_csv(sample_path)

    _require_columns(train, REQUIRED_TRAIN_COLUMNS, "train.csv")
    _require_columns(test, REQUIRED_TEST_COLUMNS, "test.csv")
    _require_columns(sample_submission, REQUIRED_SUBMISSION_COLUMNS, "sample_submission.csv")

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample submission shape: {sample_submission.shape}")
    print(f"Train columns: {list(train.columns)}")
    print(f"Test columns: {list(test.columns)}")
    return train, test, sample_submission


def print_data_checks(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\nMissing values in train:")
    print(train.isna().sum())
    print("\nMissing values in test:")
    print(test.isna().sum())
    print("\nTarget statistics:")
    print(train[TARGET_COL].describe())

    if train[TARGET_COL].isna().any():
        raise ValueError("Target column contains missing values.")
    if train[ID_COL].duplicated().any():
        print("Warning: duplicate Index values found in train.")
    if test[ID_COL].duplicated().any():
        print("Warning: duplicate Index values found in test.")
