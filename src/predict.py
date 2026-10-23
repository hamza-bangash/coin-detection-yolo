
from pathlib import Path
import argparse

from ultralytics import YOLO


def main():

    parser = argparse.ArgumentParser(
        description="Run coin detection using a trained YOLO model."
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to your locally stored model weights (.pt).",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image or directory of images.",
    )
    parser.add_argument(
        "--output",
        default="runs/coin-predictions",
        help="Directory for saving prediction results.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (default: 0.25).",
    )
    args = parser.parse_args()

    weights = Path(args.weights).expanduser()
    source = Path(args.source).expanduser()

    if not weights.is_file():
        parser.error(f"Model weights file not found: {weights}")

    if not source.exists():
        parser.error(f"Image source not found: {source}")

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1.")

    model = YOLO(str(weights))

    model.predict(
        source=str(source),
        conf=args.conf,
        save=True,
        project=args.output,
        name="results",
        exist_ok=True,
    )

    print(f"Predictions saved to: {args.output}/results")


if __name__ == "__main__":
    main()
