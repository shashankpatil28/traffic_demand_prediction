from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
OOF_PATH = OUTPUT_DIR / "oof_predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"

RANDOM_SEED = 42
N_SPLITS = 5
TARGET_COL = "demand"
ID_COL = "Index"

# The downloaded dataset uses NumberofLanes. Some problem statements write
# NumberOfLanes, so data loading normalizes either spelling to NumberofLanes.
LANE_COL = "NumberofLanes"

REQUIRED_TRAIN_COLUMNS = [
    ID_COL,
    "geohash",
    "day",
    "timestamp",
    TARGET_COL,
    "RoadType",
    LANE_COL,
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather",
]

REQUIRED_TEST_COLUMNS = [c for c in REQUIRED_TRAIN_COLUMNS if c != TARGET_COL]
REQUIRED_SUBMISSION_COLUMNS = [ID_COL, TARGET_COL]

CATBOOST_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "R2",
    "random_seed": RANDOM_SEED,
    "od_type": "Iter",
    "od_wait": 200,
    "l2_leaf_reg": 5,
    "random_strength": 1,
    "bagging_temperature": 0.5,
    "verbose": 300,
    "allow_writing_files": False,
}

QUICK_CATBOOST_PARAMS = {
    **CATBOOST_PARAMS,
    "iterations": 50,
    "learning_rate": 0.08,
    "od_wait": 20,
    "verbose": 25,
}
