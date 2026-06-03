import argparse
from pathlib import Path

from src.config import TRAIN_PATH
from src.future_window_validation import run_future_window_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fake future-window validation experiments.")
    parser.add_argument("--train_path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--quick", action="store_true", help="Use fast CatBoost settings.")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--no_target_stats", action="store_true")
    parser.add_argument("--no_day_shift", action="store_true")
    parser.add_argument("--blend_baseline", action="store_true")
    parser.add_argument("--blend_alpha", type=float, default=0.65)
    parser.add_argument("--alpha_search", action="store_true")
    parser.add_argument("--hour_calibration", action="store_true")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("outputs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_future_window_validation(
        train_path=str(args.train_path),
        output_dir=str(args.output_path),
        use_target_stats=not args.no_target_stats,
        use_day_shift=not args.no_day_shift,
        iterations=args.iterations,
        quick=args.quick,
        blend_baseline=args.blend_baseline,
        blend_alpha=args.blend_alpha,
        alpha_search=args.alpha_search,
        hour_calibration=args.hour_calibration,
    )


if __name__ == "__main__":
    main()
