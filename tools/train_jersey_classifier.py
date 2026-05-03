from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


LAYOUT = """
Expected YOLO classification layout:

  datasets/jersey_cls/
    train/
      0/
      1/
      11/
      23/
      unknown/
    val/
      0/
      1/
      11/
      23/
      unknown/

Use tools/review_jersey_crops.py to build this dataset from OCR crops.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a jersey-number classification model from reviewed jersey crops.")
    parser.add_argument("--data", default="datasets/jersey_cls")
    parser.add_argument("--base-model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/classify/runs/jersey_cls")
    parser.add_argument("--name", default="jersey_number_cls")
    parser.add_argument("--show-layout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_layout:
        print(LAYOUT)
        return
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"Dataset not found: {data}\n{LAYOUT}")
    model = YOLO(args.base_model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
