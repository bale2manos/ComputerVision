from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-class YOLO ball detector.")
    parser.add_argument("--data", default="datasets/ball/data.yaml", help="YOLO data.yaml path.")
    parser.add_argument("--base-model", default="yolo11s.pt", help="Base YOLO weights.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or 0.")
    parser.add_argument("--project", default="runs/ball_train", help="Ultralytics project directory.")
    parser.add_argument("--name", default="yolo11s_ball", help="Run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=20,
        plots=True,
    )


if __name__ == "__main__":
    main()
